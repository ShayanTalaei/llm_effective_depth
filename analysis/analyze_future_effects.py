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


def plot_logit_diffs(dall):
    """Plot how much each layer intervention affects the final output."""
    fig = plt.figure(figsize=(6,3))
    dall = dall.squeeze()
    plt.bar(list(range(dall.shape[0])), dall)
    plt.xlim(-1, dall.shape[0])
    plt.xlabel("Layer")
    plt.ylabel("Output change norm")
    return fig


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
    

@ndif_cache_wrapper
def test_effect(llm, prompt, positions, part, no_skip_front=1):
    """
    Test the effect of intervening on each layer at specific positions.
    
    For each layer and each position:
    1. Run normal forward pass and record all layer states
    2. Run intervened forward pass and record all layer states
    3. Compare how much each future layer changed due to the intervention
    
    Args:
        llm: Language model to analyze
        prompt: Text prompt to analyze
        positions: List of token positions to intervene at
        part: Which part to intervene on ("layer", "mlp", or "attention")
        no_skip_front: Number of initial tokens to never intervene on
    
    Returns:
        dall: Max relative change in layer representations across all positions
        dall_out: Max relative change in final output across all positions
    """
    all_diffs = []
    all_out_diffs = []

    with llm.session(remote=llm.remote) as session:
        with torch.no_grad():
            residual_log = []
            
            # First pass: Record normal processing without any intervention
            with llm.trace(prompt) as tracer:
                for i, layer in enumerate(llm.model.layers):
                    if i == 0:
                        residual_log.clear()
                    # Record the residual stream change at each layer
                    residual_log.append(layer.output[0].detach().cpu().float() - layer.inputs[0][0].detach().cpu().float())

                residual_log = torch.cat(residual_log, dim=0)
                outputs = llm.output.logits.detach().float().softmax(dim=-1).cpu()
                
            # For each position where we want to test intervention effects
            for t in positions:
                diffs = []
                out_diffs = []

                # For each layer, test what happens if we intervene at that layer
                for lskip in range(len(llm.model.layers)):
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

                all_diffs.append(torch.stack(diffs, dim=0))
                all_out_diffs.append(torch.stack(out_diffs, dim=0))

            # Take maximum effect across all tested positions
            dall = torch.stack(all_diffs, dim=0).max(dim=0).values.save()
            dall_out = torch.stack(all_out_diffs, dim=0).max(dim=0).values.save()
    return dall, dall_out


def test_future_max_effect(llm, prompt, N_CHUNKS=4, part = "layer"):
    """
    Test future effects by sampling multiple positions throughout the sequence.
    
    Rather than testing every position (expensive), sample N_CHUNKS positions
    spaced throughout the sequence to get a representative sample.
    """
    all_diffs = []
    all_out_diffs = []

    _, tokens = tokenize(llm, prompt)

    # Sample positions throughout the sequence (avoiding very start/end)
    positions = list(range(8, len(tokens)-4, 8))
    random.shuffle(positions)
    positions = positions[:N_CHUNKS]
    
    return test_effect(llm, prompt, positions, part)


def run(llm, model_name, dataset_name, run_name, n_examples=10):
    """
    Run the complete future effects analysis.
    
    Tests three types of interventions:
    1. Layer skipping ("layer")
    2. MLP ablation ("mlp") 
    3. Attention ablation ("attention")
    
    For each intervention type, generates visualizations showing:
    - How much each layer affects future layer processing
    - How much each layer affects final output
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
        
        if dataset_name == "gsm8k":
            dataset = GSM8K()
        elif dataset_name == "stack_v2":
            dataset = STACK_V2()
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        
        # Accumulate results across multiple examples
        for idx, prompt in enumerate(dataset):
            print(prompt)
            diff_now, diff_out = test_future_max_effect(llm, prompt, part=what)
            d_max = torch.max(d_max, diff_now)
            dout_max = torch.max(dout_max, diff_out)

            dall.append(diff_now)
            if idx == N_EXAMPLES - 1:
                break

        print("--------------------------------")

        # Save heatmap: intervention layer vs affected layer  
        fig = plot_layer_diffs(d_max)
        fig.savefig(os.path.join(target_dir, f"{model_name}_future_max_effect_{what}.pdf"), bbox_inches="tight")

        # Save bar plot: intervention layer vs final output change
        fig = plot_logit_diffs(dout_max)
        fig.savefig(os.path.join(target_dir, f"{model_name}_future_max_effect_out_{what}.pdf"), bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description='Analyze future effects of layer interventions')
    parser.add_argument('--model_name', type=str, help='Name of the model to analyze')
    parser.add_argument('--dataset_name', type=str, choices=['gsm8k', 'stack_v2'], 
                       help='Dataset to use for analysis')
    parser.add_argument('--run_name', type=str, help='Name for this run (creates subfolder in output directory)')
    parser.add_argument('--n_examples', type=int, default=10, help='Number of examples to analyze')
    args = parser.parse_args()

    llm = create_model(args.model_name, force_local=False)
    llm.eval()

    run(llm, args.model_name, args.dataset_name, args.run_name, args.n_examples)

if __name__ == "__main__":
    main()
