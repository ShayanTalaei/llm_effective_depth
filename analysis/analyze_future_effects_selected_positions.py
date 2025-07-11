"""
Future Effects Analysis for Understanding Layer Dependencies

This script analyzes how interventions at one layer affect the processing in all subsequent layers.
Unlike layer skipping which completely removes a layer, this uses more surgical interventions:

1. Layer intervention: Replace layer output with layer input (effectively skipping computation)
2. MLP intervention: Zero out MLP contributions  
3. Attention intervention: Zero out attention contributions

Key insights:
- Which layers have the most impact on future processing
- How much do attention vs MLP components contribute
- Whether early layers have cascading effects through many future layers

This helps understand the "computational graph" of the transformer and which layers
are most critical for the overall computation.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
import os
import argparse
import nnsight
nnsight.CONFIG.API.APIKEY =  os.environ["NDIF_TOKEN"]
import torch
import random
import time
from nnsight import LanguageModel
from typing import Optional

from lib.models import get_model, create_model
from lib.nnsight_tokenize import tokenize
from lib.datasets import STACK_V2, GSM8K
from tqdm import tqdm
from lib.ndif_cache import ndif_cache_wrapper


def plot_layer_diffs(dall):
    """
    Plot a heatmap showing how much each layer intervention affects each future layer.
    
    Args:
        dall: 2D tensor where dall[i,j] = effect of intervening at layer i on layer j
    
    The heatmap shows:
    - X-axis: Which layer is affected (future layers)
    - Y-axis: Which layer was intervened on (skipped)
    - Color intensity: How much the intervention changed the future layer's processing
    """
    fig, ax = plt.subplots(figsize=(10,3))
    im = ax.imshow(dall.float().cpu().numpy(), vmin=0, vmax=1, interpolation="nearest")
    plt.ylabel("Layer skipped")
    plt.xlabel("Effect @ layer")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.2, pad=0.1)
    cbar = fig.colorbar(im, cax=cax, label='Relative change')
    return fig


def plot_logit_diffs(dall, layers_to_test=None):
    """Plot how much each layer intervention affects the final output."""
    fig = plt.figure(figsize=(6,3))
    dall = dall.squeeze()
    
    # Use custom layer indices if provided
    if layers_to_test is not None:
        x_labels = layers_to_test
        plt.bar(range(len(x_labels)), dall)
        plt.xticks(range(len(x_labels)), x_labels)
        plt.xlim(-0.5, len(x_labels)-0.5)
    else:
        plt.bar(list(range(dall.shape[0])), dall)
        plt.xlim(-1, dall.shape[0])
    
    plt.xlabel("Layer")
    plt.ylabel("Output change norm")
    return fig


def plot_layer_diffs_per_position(position_results, layers_to_test=None):
    """
    Plot separate heatmaps for each position showing layer intervention effects.
    
    Args:
        position_results: Dict mapping position -> (layer_diffs, out_diffs)
        layers_to_test: List of layer indices tested (for labeling)
    
    Returns:
        Dict mapping position -> figure
    """
    figures = {}
    
    for position, (layer_diffs, out_diffs) in position_results.items():
        fig, ax = plt.subplots(figsize=(10,3))
        im = ax.imshow(layer_diffs.float().cpu().numpy(), vmin=0, vmax=1, interpolation="nearest")
        
        # Set labels
        if layers_to_test is not None:
            ax.set_yticks(range(len(layers_to_test)))
            ax.set_yticklabels(layers_to_test)
        
        ax.set_ylabel("Layer intervened")
        ax.set_xlabel("Effect @ layer")
        ax.set_title(f"Layer Effects at Position {position}")
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.1)
        cbar = fig.colorbar(im, cax=cax, label='Relative change')
        
        figures[position] = fig
    
    return figures


def plot_logit_diffs_per_position(position_results, layers_to_test=None):
    """
    Plot separate bar charts for each position showing final output effects.
    
    Args:
        position_results: Dict mapping position -> (layer_diffs, out_diffs)
        layers_to_test: List of layer indices tested (for labeling)
    
    Returns:
        Dict mapping position -> figure
    """
    figures = {}
    
    for position, (layer_diffs, out_diffs) in position_results.items():
        fig = plt.figure(figsize=(6,3))
        out_diffs = out_diffs.squeeze()
        
        # Use custom layer indices if provided
        if layers_to_test is not None:
            x_labels = layers_to_test
            plt.bar(range(len(x_labels)), out_diffs)
            plt.xticks(range(len(x_labels)), x_labels)
            plt.xlim(-0.5, len(x_labels)-0.5)
        else:
            plt.bar(list(range(out_diffs.shape[0])), out_diffs)
            plt.xlim(-1, out_diffs.shape[0])
        
        plt.xlabel("Layer")
        plt.ylabel("Output change norm")
        plt.title(f"Output Effects at Position {position}")
        
        figures[position] = fig
    
    return figures


def merge_io(intervened, orig, t: Optional[int] = None, no_skip_front: int = 1):
    """
    Merge intervened and original activations at a specific time position.
    
    This function allows us to:
    1. Keep the original computation for early tokens (no_skip_front)
    2. Use intervened computation from position t onwards
    3. Or use intervened computation for all tokens after no_skip_front if t is None
    
    Args:
        intervened: The modified activations (e.g., zeros for ablation)
        orig: The original activations
        t: Time position to start intervention (None = intervene from no_skip_front to end)
        no_skip_front: Number of initial tokens to never intervene on
    """
    outs = [orig[:, :no_skip_front]]
    if t is not None:
        outs.append(intervened[:, no_skip_front:t].to(orig.device))
        outs.append(orig[:, t:])    
    else:
        outs.append(intervened[:, no_skip_front:].to(orig.device))
    
    return torch.cat(outs, dim=1)


def intervene_layer(layer, t: Optional[int], part: str, no_skip_front: int):
    """
    Apply intervention to a specific part of a layer.
    
    Three types of interventions:
    1. "layer": Replace layer output with layer input (skip the entire layer)
    2. "mlp": Zero out the MLP contribution  
    3. "attention": Zero out the attention contribution
    
    Args:
        layer: The transformer layer to intervene on
        t: Time position to start intervention
        part: Which part to intervene on ("layer", "mlp", or "attention")
        no_skip_front: Number of initial tokens to never intervene on
    """
    if part == "layer":
        # Skip entire layer: output = input (no computation applied)
        layer.output = merge_io(layer.inputs[0][0], layer.output[0], t, no_skip_front),
    elif part == "mlp":
        # Zero out MLP: remove the feedforward contribution
        layer.mlp.output = merge_io(torch.zeros_like(layer.mlp.output), layer.mlp.output, t, no_skip_front)
    elif part == "attention":
        # Zero out attention: remove the self-attention contribution
        layer.self_attn.output = merge_io(torch.zeros_like(layer.self_attn.output[0]), layer.self_attn.output[0], t, no_skip_front), layer.self_attn.output[1]
    else:
        raise ValueError(f"Invalid part: {part}")
    

def get_future(data, t: Optional[int]):
    """Extract the future part of a sequence from position t onwards."""
    if t is not None:
        return data[:, t:]
    else:
        return data
    

# @ndif_cache_wrapper
def test_effect(llm, prompt, positions, part, layers_to_test=None, no_skip_front=1):
    """
    Test the effect of intervening on specific layers at specific positions.
    
    Execution model:
    1. Collect baseline trace (no intervention)
    2. For each position P and layer L: collect intervention trace
    3. When session exits, all traces are batched and sent to NDIF
    4. Compare baseline vs intervention results
    
    Total traces: 1 + len(positions) × len(layers_to_test)
    
    Args:
        llm: Language model to analyze
        prompt: Text prompt to analyze
        positions: List of token positions to intervene at
        part: Which part to intervene on ("layer", "mlp", or "attention")
        layers_to_test: List of layer indices to test (None = all layers)
        no_skip_front: Number of initial tokens to never intervene on
    
    Returns:
        dall: Max relative change in layer representations across all positions
        dall_out: Max relative change in final output across all positions
        position_results: Dict mapping position -> (layer_diffs, out_diffs) for per-position analysis
    """
    start_time = time.time()
    all_diffs = []
    all_out_diffs = []
    position_results = {}
    
    # Use all layers if none specified
    if layers_to_test is None:
        layers_to_test = list(range(len(llm.model.layers)))
    
    num_layers_total = len(llm.model.layers)
    num_layers_to_test = len(layers_to_test)
    total_traces = 1 + len(positions) * num_layers_to_test
    print(f"Starting analysis: {len(positions)} positions × {num_layers_to_test} layers = {total_traces} total traces")
    print(f"Testing layers: {layers_to_test}")
    print(f"Testing positions: {positions}")
    print("Note: All traces will be batched and sent to NDIF when session exits")

    with llm.session(remote=llm.remote) as session:
        session_start = time.time()
        with torch.no_grad():
            residual_log = []
            
            # First pass: Record normal processing without any intervention
            print("Collecting baseline trace...")
            with llm.trace(prompt) as tracer:
                for i, layer in enumerate(llm.model.layers):
                    if i == 0:
                        residual_log.clear()
                    # Record the residual stream change at each layer
                    residual_log.append(layer.output[0].detach().cpu().float() - layer.inputs[0][0].detach().cpu().float())

                residual_log = torch.cat(residual_log, dim=0)
                outputs = llm.output.logits.detach().float().softmax(dim=-1).cpu()
                
                        # For each position where we want to test intervention effects
            for pos_idx, t in enumerate(positions):
                print(f"Collecting traces for position {pos_idx+1}/{len(positions)} (token {t})")
                diffs = []
                out_diffs = []

                # For each layer we want to test, test what happens if we intervene at that layer
                for test_idx, lskip in enumerate(layers_to_test):
                    with llm.trace(prompt) as tracer:
                        new_logs = []

                        # Apply intervention at layer lskip, position t
                        intervene_layer(llm.model.layers[lskip], t, part, no_skip_front)

                        # Record the new residual stream changes at all layers
                        for i, layer in enumerate(llm.model.layers):
                            new_logs.append((layer.output[0].detach().cpu().float() - layer.inputs[0][0].detach().cpu().float()))

                        new_logs = torch.cat(new_logs, dim=0).float()

                        # Calculate relative change in future layer representations
                        # Only look at positions from t onwards (the "future" affected by intervention)
                        future_orig = get_future(residual_log, t)
                        future_new = get_future(new_logs, t)
                        
                        relative_diffs = (future_orig - future_new).norm(dim=-1) / future_orig.norm(dim=-1).clamp(min=1e-6)

                        # Take max across all future positions for each layer
                        diffs.append(relative_diffs.max(dim=-1).values)
                        
                        # Also measure change in final output
                        future_orig_out = get_future(outputs, t)
                        future_new_out = get_future(llm.output.logits.detach(), t).float().softmax(dim=-1).cpu()
                        
                        out_diffs.append((future_orig_out - future_new_out).norm(dim=-1).max(dim=-1).values)

                    if (test_idx + 1) % 8 == 0:  # Log every 8 layers to avoid spam
                        print(f"  Collected {test_idx+1}/{num_layers_to_test} intervention traces")

                # Store per-position results (need to save for use outside trace)
                pos_layer_diffs = torch.stack(diffs, dim=0).save()  # [num_tested_layers, num_all_layers]
                pos_out_diffs = torch.stack(out_diffs, dim=0).save()  # [num_tested_layers]
                position_results[t] = (pos_layer_diffs, pos_out_diffs)

                all_diffs.append(pos_layer_diffs)
                all_out_diffs.append(pos_out_diffs)

            # Take maximum effect across all tested positions
            dall = torch.stack(all_diffs, dim=0).max(dim=0).values.save()
            dall_out = torch.stack(all_out_diffs, dim=0).max(dim=0).values.save()
        
        print(f"Session collected {total_traces} traces, sending to NDIF...")
        # When session exits here, all traces are sent to NDIF as a batch
        
    session_time = time.time() - session_start
    total_time = time.time() - start_time
    print(f"NDIF execution completed in {session_time:.2f}s")
    print(f"Total analysis time: {total_time:.2f}s")
    return dall, dall_out, position_results



def test_future_max_effect(llm, prompt, N_CHUNKS=2, part="layer", custom_positions=None, layers_to_test=None):
    """
    Test future effects by sampling multiple positions throughout the sequence.
    
    Rather than testing every position (expensive), sample N_CHUNKS positions
    spaced throughout the sequence to get a representative sample.
    
    Args:
        custom_positions: List of specific positions to test (overrides N_CHUNKS sampling)
        layers_to_test: List of layer indices to test (None = all layers)
    """
    _, tokens = tokenize(llm, prompt)

    # Use custom positions if provided, otherwise sample
    if custom_positions is not None:
        positions = custom_positions
        print(f"Using custom positions: {positions}")
    else:
        # Sample positions throughout the sequence (avoiding very start/end)
        positions = list(range(8, len(tokens)-4, 8))
        random.shuffle(positions)
        positions = positions[:N_CHUNKS]
        print(f"Sampled {N_CHUNKS} positions: {positions}")
    
    return test_effect(llm, prompt, positions, part, layers_to_test)


def run(llm, model_name, dataset_name, run_name, n_examples=10, custom_positions=None, layers_to_test=None):
    """
    Run the complete future effects analysis.
    
    Tests three types of interventions:
    1. Layer skipping ("layer")
    2. MLP ablation ("mlp") 
    3. Attention ablation ("attention")
    
    For each intervention type, generates visualizations showing:
    - How much each layer affects future layer processing
    - How much each layer affects final output
    
    Args:
        custom_positions: List of specific positions to test (e.g., [30, 100, 300, 900])
        layers_to_test: List of layer indices to test (e.g., [2, 6, 10, 14, 18, 22, 26, 30])
    """
    N_EXAMPLES = n_examples

    random.seed(123)

    target_dir = os.path.join("out/future_effects", run_name)

    os.makedirs(target_dir, exist_ok=True)

    # Test all three types of interventions
    for what in ["layer"]: #, "mlp", "attention" To be added later
        dall = []
        d_max = torch.zeros([1])      # Max effect on future layers
        dout_max = torch.zeros([1])   # Max effect on final output
        all_position_results = {}     # Accumulate per-position results
        
        if dataset_name == "gsm8k":
            dataset = GSM8K()
        elif dataset_name == "stack_v2":
            dataset = STACK_V2()
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        
        # Accumulate results across multiple examples
        for idx, prompt in enumerate(dataset):
            print(f"Example {idx+1}/{N_EXAMPLES}: {prompt[:100]}...")
            diff_now, diff_out, position_results = test_future_max_effect(
                llm, prompt, part=what, custom_positions=custom_positions, layers_to_test=layers_to_test
            )
            d_max = torch.max(d_max, diff_now)
            dout_max = torch.max(dout_max, diff_out)

            dall.append(diff_now)
            
            # Accumulate per-position results
            for pos, (layer_diffs, out_diffs) in position_results.items():
                if pos not in all_position_results:
                    all_position_results[pos] = {'layer_diffs': [], 'out_diffs': []}
                all_position_results[pos]['layer_diffs'].append(layer_diffs)
                all_position_results[pos]['out_diffs'].append(out_diffs)
            
            if idx == N_EXAMPLES - 1:
                break

        print("--------------------------------")

        # Save aggregated plots
        fig = plot_layer_diffs(d_max)
        fig.savefig(os.path.join(target_dir, f"{model_name}_future_max_effect_{what}_aggregated.pdf"), bbox_inches="tight")

        fig = plot_logit_diffs(dout_max, layers_to_test)
        fig.savefig(os.path.join(target_dir, f"{model_name}_future_max_effect_out_{what}_aggregated.pdf"), bbox_inches="tight")

        # Generate per-position plots
        print("Generating per-position plots...")
        
        # Take max across examples for each position
        final_position_results = {}
        for pos, results in all_position_results.items():
            # Stack results from all examples and take max
            layer_diffs_max = torch.stack(results['layer_diffs']).max(dim=0).values
            out_diffs_max = torch.stack(results['out_diffs']).max(dim=0).values
            final_position_results[pos] = (layer_diffs_max, out_diffs_max)
        
        # Generate and save per-position layer effect plots
        layer_figs = plot_layer_diffs_per_position(final_position_results, layers_to_test)
        for pos, fig in layer_figs.items():
            fig.savefig(os.path.join(target_dir, f"{model_name}_future_effect_{what}_pos{pos}.pdf"), bbox_inches="tight")
        
        # Generate and save per-position output effect plots
        output_figs = plot_logit_diffs_per_position(final_position_results, layers_to_test)
        for pos, fig in output_figs.items():
            fig.savefig(os.path.join(target_dir, f"{model_name}_future_effect_out_{what}_pos{pos}.pdf"), bbox_inches="tight")
        
        print(f"Saved {len(layer_figs)} per-position layer plots and {len(output_figs)} per-position output plots")


def main():
    parser = argparse.ArgumentParser(description='Analyze future effects of layer interventions')
    parser.add_argument('--model_name', type=str, help='Name of the model to analyze')
    parser.add_argument('--dataset_name', type=str, choices=['gsm8k', 'stack_v2'], 
                       help='Dataset to use for analysis')
    parser.add_argument('--run_name', type=str, help='Name for this run (creates subfolder in output directory)')
    parser.add_argument('--n_examples', type=int, default=10, help='Number of examples to analyze')
    parser.add_argument('--positions', type=str, help='Comma-separated list of token positions to test (e.g., "30,100,300,900")')
    parser.add_argument('--layers', type=str, help='Comma-separated list of layer indices to test (e.g., "2,6,10,14,18,22,26,30")')
    args = parser.parse_args()

    # Parse custom positions
    custom_positions = None
    if args.positions:
        custom_positions = [int(x.strip()) for x in args.positions.split(',')]
        print(f"Using custom positions: {custom_positions}")
    
    # Parse custom layers
    layers_to_test = None
    if args.layers:
        layers_to_test = [int(x.strip()) for x in args.layers.split(',')]
        print(f"Using custom layers: {layers_to_test}")

    llm = create_model(args.model_name, force_local=False)
    llm.eval()

    run(llm, args.model_name, args.dataset_name, args.run_name, args.n_examples, custom_positions, layers_to_test)

if __name__ == "__main__":
    main()
