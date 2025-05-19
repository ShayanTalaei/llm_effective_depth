import os
import sys
import shutil
import lib

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <checkpoint>")
    sys.exit(1)

model_name = sys.argv[1]

target_dir = "out/dm_math"
os.makedirs(target_dir, exist_ok=True)

shutil.rmtree("save/tst", ignore_errors=True)
sys.argv = f"main.py -log tb -name tst -restore {model_name} -amp 1 -compile 0 -reset 1".split(" ")

# Pretend we are in the main directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir+"/../")

from main import initialize
import torch
from typing import List, Tuple, Dict, Any, Optional
import torch.utils
from layers.transformer import generate_causal_attention_mask, AttentionMask
import random
import numpy as np
import math
import ast
from tqdm import tqdm

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Note: checkpoints have all arguments saved
helper, task = initialize()

task.model.eval()


model_name = ("moeut" if "moeut" in task.helper.args.task else "transformer") + "_" + ("nolm" if task.helper.args.dm_math.loss_on_target_only else "lm")

saved_inputs = []
def layer_hook(module, inputs, outputs):
    global saved_inputs
    saved_inputs.append(outputs[0].detach())

def model_pre_hook(module, inputs):
    global saved_inputs
    saved_inputs.clear()
    saved_inputs.append(inputs[0].detach())

for layer in task.model.transformer.layers:
    layer._forward_hooks.clear()
    layer.register_forward_hook(layer_hook)

task.model.transformer._forward_pre_hooks.clear()
task.model.transformer.register_forward_pre_hook(model_pre_hook)

def run_model(tokens):
    inputs = torch.tensor(tokens, dtype=torch.long).cuda()[:, None]
    with torch.no_grad():
        res, logs = task.run_model({
            "data": inputs
        }, 0)
        activations = torch.stack(saved_inputs, dim=0)
        saved_inputs.clear()

    activations = activations[:, :, :res.outputs.shape[0]]
    return activations, res, logs

def tokens_to_str(tokens):
    return [task.train_set.vocabulary.to_string(i) for i in tokens]


def tokenize(input_text: str) -> Tuple[torch.Tensor, List[str]]:
    inputs = task.train_set.vocabulary(input_text)
    return inputs, tokens_to_str(inputs)


def partial_run(activations: torch.Tensor, from_layer: int):
    global saved_inputs
    mask = AttentionMask(None, generate_causal_attention_mask(activations.shape[1], activations.device))
    x = activations

    from_repeat = from_layer // len(task.model.transformer.layers)
    from_layer = from_layer % len(task.model.transformer.layers)
    # with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):

    for r in range(from_repeat, getattr(task.model.transformer, "n_repeats", 1)):
        for lid in range(from_layer, len(task.model.transformer.layers)):
            if lid == 0 and hasattr(task.model.transformer, "repeat_norm"):
                x = task.model.transformer.repeat_norm(x)

            layer = task.model.transformer.layers[lid]
            x, _ = layer(x, mask)

        from_layer = 0

    x = task.model.out_norm(x)

    saved_inputs.clear()
    if hasattr(task.model.transformer, "collect_losses"):
        task.model.transformer.collect_losses(x.device)

    return task.model.lm_head(x)


@torch.no_grad()
def measure_erasal(q, a, counterfactual: torch.Tensor, compare_tokens = 1, skip_n = 0, force_bad=False):
    inputs = q+a
    nlayers = len(task.model.transformer.layers) * getattr(task.model.transformer, "n_repeats", 1)

    ntokens = len(inputs)

    ref_act, outs, _ = run_model(inputs)
    ref_logits = outs.outputs.transpose(0, 1)

    ref_ids = ref_logits[0, -len(a):].argmax(-1)

    answer_ok = all(i==j for i, j in zip(ref_ids.tolist(), a))
    if not (answer_ok or force_bad):
        return None

    def cut_logits(logits):
        res = logits[:, -(compare_tokens+skip_n):-skip_n] if skip_n > 0 else logits[:, -compare_tokens:]
        res = res.softmax(-1)
        return res

    ref_logits = cut_logits(ref_logits)
    res = torch.zeros(nlayers, ntokens-1, device=ref_act.device, dtype=torch.float32)

    for l in range(0, nlayers):
        act = ref_act[l].clone()

        erased = counterfactual[l].mean(dim=1, keepdim=True)

        batch_in = act.repeat(ntokens - 2, 1, 1)

        for t in range(1, ntokens-1):
            batch_in[t-1, t, :] = erased

        logits = partial_run(batch_in, l)
        last_logits = cut_logits(logits)
        res[l, 1:] = (ref_logits - last_logits).norm(dim=-1).max(dim=-1).values

    if force_bad:
        return res, answer_ok
    else:
        return res



dm_math_prompts = [
    ("arithmetic_1", ("Q: What is ((-14)/(-6))/(1162/(-4980))? A: ", "-10")),
    ("arithmetic_2", ("Q: What is the value of -24 + -10 + 22 + 1 + 1 + 25? A: ", "-15")),
    ("arithmetic_3", ("Q: What is 1 - 3 - (63 + 7 - 60)? A: ", "-12")),
    ("arithmetic_4", ("Q: (10/3)/((-8)/16)*(-90)/(-75) A: ", "-8")),
]


def plot_erasal(ref: Tuple[str, str]):
    ref_q, ref_q_tokens = tokenize(ref[0])
    ref_a, ref_a_tokens = tokenize(ref[1])

    cin, _ = tokenize("Q: Simplify 5 + -2*(sqrt(180)*1*2 - sqrt(180)) + 4 + (sqrt(180)*2 + -4 - sqrt(180)) + 4. A: -6*sqrt(5) + 9")
    counter, _, _ = run_model(cin)

    diffmap, answer_ok = measure_erasal(ref_q, ref_a, counter, len(ref_a_tokens), force_bad=True)
    fig, ax = plt.subplots()
    im = ax.imshow(diffmap.cpu().numpy(), vmin=0)
    ax.invert_yaxis()
    # plt.yticks(range(len(task.model.transformer.layers)), [f"{i}" for i in reversed(range(len(task.model.transformer.layers)))], fontsize=8)
    xlabels = ref_q_tokens+ref_a_tokens
    plt.xticks(range(len(xlabels)-1), xlabels[:-1], rotation=45, ha='right',rotation_mode="anchor", fontsize=5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.1, pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    return fig, answer_ok


for name, (q, a) in tqdm(dm_math_prompts):
    fig, aok = plot_erasal((q, a))
    plt.show()
    fig.savefig(f"{target_dir}/erasal_{model_name}_{name}.pdf", bbox_inches="tight")
    if not aok:
        with open(f"{target_dir}/erasal_bad_{model_name}_{name}_fail.txt", "w") as f:
            f.write(f"{q}\n{a}\n")
