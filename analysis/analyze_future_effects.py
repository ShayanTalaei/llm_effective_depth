import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
import os
import nnsight
nnsight.CONFIG.API.APIKEY =  os.environ["NDIF_TOKEN"]
import torch
import random
from nnsight import LanguageModel
from typing import Optional

from lib.models import get_model, create_model
from lib.nnsight_tokenize import tokenize
from lib.datasets import GSM8K
from tqdm import tqdm
from lib.ndif_cache import ndif_cache_wrapper


def plot_layer_diffs(dall):
    fig, ax = plt.subplots(figsize=(10,3))
    im = ax.imshow(dall.float().cpu().numpy(), vmin=0, vmax=1, interpolation="nearest")
    plt.ylabel("Layer skipped")
    plt.xlabel("Effect @ layer")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.2, pad=0.1)
    cbar = fig.colorbar(im, cax=cax, label='Relative change')
    return fig


def plot_logit_diffs(dall):
    fig = plt.figure(figsize=(6,3))
    dall = dall.squeeze()
    plt.bar(list(range(dall.shape[0])), dall)
    plt.xlim(-1, dall.shape[0])
    plt.xlabel("Layer")
    plt.ylabel("Output change norm")
    return fig


def merge_io(intervened, orig, t: Optional[int] = None, no_skip_front: int = 1):
    outs = [orig[:, :no_skip_front]]
    if t is not None:
        outs.append(intervened[:, no_skip_front:t].to(orig.device))
        outs.append(orig[:, t:])    
    else:
        outs.append(intervened[:, no_skip_front:].to(orig.device))
    
    return torch.cat(outs, dim=1)


def intervene_layer(layer, t: Optional[int], part: str, no_skip_front: int):
    if part == "layer":
        layer.output = merge_io(layer.inputs[0][0], layer.output[0], t, no_skip_front),
    elif part == "mlp":
        layer.mlp.output = merge_io(torch.zeros_like(layer.mlp.output), layer.mlp.output, t, no_skip_front)
    elif part == "attention":
        layer.self_attn.output = merge_io(torch.zeros_like(layer.self_attn.output[0]), layer.self_attn.output[0], t, no_skip_front), layer.self_attn.output[1]
    else:
        raise ValueError(f"Invalid part: {part}")
    

def get_future(data, t: Optional[int]):
    if t is not None:
        return data[:, t:]
    else:
        return data
    

@ndif_cache_wrapper
def test_effect(llm, prompt, positions, part, no_skip_front=1):
    all_diffs = []
    all_out_diffs = []

    with llm.session(remote=llm.remote) as session:
        with torch.no_grad():
            residual_log = []
            with llm.trace(prompt) as tracer:
                for i, layer in enumerate(llm.model.layers):
                    if i == 0:
                        residual_log.clear()
                    residual_log.append(layer.output[0].detach().cpu().float() - layer.inputs[0][0].detach().cpu().float())

                residual_log = torch.cat(residual_log, dim=0)
                outputs = llm.output.logits.detach().float().softmax(dim=-1).cpu()
                
            for t in positions:
            # with session.iter(positions) as t:
                diffs = []
                out_diffs = []

                for lskip in range(len(llm.model.layers)):
                # with session.iter(range(len(llm.model.layers))) as lskip:
                    with llm.trace(prompt) as tracer:
                        new_logs = []

                        intervene_layer(llm.model.layers[lskip], t, part, no_skip_front)

                        for i, layer in enumerate(llm.model.layers):
                            new_logs.append((layer.output[0].detach().cpu().float() - layer.inputs[0][0].detach().cpu().float()))#.cpu())

                        new_logs = torch.cat(new_logs, dim=0).float()

                        relative_diffs = (get_future(residual_log, t) - get_future(new_logs, t)).norm(dim=-1) / get_future(residual_log, t).norm(dim=-1).clamp(min=1e-6)

                        diffs.append(relative_diffs.max(dim=-1).values)
                        out_diffs.append((get_future(llm.output.logits.detach(), t).float().softmax(dim=-1).cpu() - get_future(outputs, t)).norm(dim=-1).max(dim=-1).values)

                all_diffs.append(torch.stack(diffs, dim=0))
                all_out_diffs.append(torch.stack(out_diffs, dim=0))

            dall = torch.stack(all_diffs, dim=0).max(dim=0).values.save()
            dall_out = torch.stack(all_out_diffs, dim=0).max(dim=0).values.save()
    return dall, dall_out


def test_future_max_effect(llm, prompt, N_CHUNKS=4, part = "layer"):
    all_diffs = []
    all_out_diffs = []

    _, tokens = tokenize(llm, prompt)

    positions = list(range(8, len(tokens)-4, 8))
    random.shuffle(positions)
    positions = positions[:N_CHUNKS]
    # positions = [10]
    
    return test_effect(llm, prompt, positions, part)


def run(llm, model_name):
    N_EXAMPLES = 10

    random.seed(123)

    target_dir = "out/future_effects"

    os.makedirs(target_dir, exist_ok=True)

    for what in ["layer", "mlp", "attention"]:
        dall = []
        d_max = torch.zeros([1])
        dout_max = torch.zeros([1])
        for idx, prompt in enumerate(GSM8K()):
            print(prompt)
            diff_now, diff_out = test_future_max_effect(llm, prompt, part=what)
            d_max = torch.max(d_max, diff_now)
            dout_max = torch.max(dout_max, diff_out)

            dall.append(diff_now)
            if idx == N_EXAMPLES - 1:
                break

        print("--------------------------------")

        fig = plot_layer_diffs(d_max)
        fig.savefig(os.path.join(target_dir, f"{model_name}_future_max_effect_{what}.pdf"), bbox_inches="tight")

        fig = plot_logit_diffs(dout_max)
        fig.savefig(os.path.join(target_dir, f"{model_name}_future_max_effect_out_{what}.pdf"), bbox_inches="tight")


def main():
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        raise ValueError("Please provide a model name")

    llm = create_model(model_name, force_local=False)
    llm.eval()

    run(llm, model_name)

if __name__ == "__main__":
    main()
