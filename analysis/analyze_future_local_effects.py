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
from analyze_future_effects import plot_layer_diffs, plot_logit_diffs, merge_io, get_future



@ndif_cache_wrapper
def test_local_effect(llm, prompt, positions, no_skip_front=1):
    all_diffs = []

    with llm.session(remote=llm.remote) as session:
        with torch.no_grad():
            contribution_log = []
            raw_out_logs = []
            with llm.trace(prompt) as tracer:
                for i, layer in enumerate(llm.model.layers):
                    if i == 0:
                        contribution_log.clear()
                        raw_out_logs.clear()
                    contribution_log.append(layer.output[0].detach() - layer.inputs[0][0].detach())
                    raw_out_logs.append(layer.output[0].detach())

                contribution_log = torch.cat(contribution_log, dim=0)
            
            contribution_log_cpu = contribution_log.cpu().float()
            for t in positions:
                diffs = []

                for lskip in range(len(llm.model.layers)):
                    with llm.trace(prompt) as tracer:
                        new_logs = []

                        for i, layer in enumerate(llm.model.layers):
                            new_logs.append((layer.output[0].detach().cpu().float() - layer.inputs[0][0].detach().cpu().float()))#.cpu())
                            if i >= lskip:
                                layer.output = merge_io(raw_out_logs[i] - contribution_log[lskip], raw_out_logs[i], t, no_skip_front),

                        new_logs = torch.cat(new_logs, dim=0).float()

                        relative_diffs = (get_future(contribution_log_cpu, t) - get_future(new_logs, t)).norm(dim=-1) / get_future(contribution_log_cpu, t).norm(dim=-1).clamp(min=1e-6)

                        diffs.append(relative_diffs.max(dim=-1).values)

                all_diffs.append(torch.stack(diffs, dim=0))

            dall = torch.stack(all_diffs, dim=0).max(dim=0).values.save()
    return dall


def test_future_max_local_effect(llm, prompt, N_CHUNKS=4):
    all_diffs = []
    all_out_diffs = []

    _, tokens = tokenize(llm, prompt)

    positions = list(range(8, len(tokens)-4, 8))
    random.shuffle(positions)
    positions = positions[:N_CHUNKS]
    # positions = [10]
    
    return test_local_effect(llm, prompt, positions)



def run(llm, model_name):
    N_EXAMPLES = 10

    random.seed(123)

    target_dir = "out/future_local_effects"

    os.makedirs(target_dir, exist_ok=True)

    dall = []
    d_max = torch.zeros([1])
    for idx, prompt in enumerate(GSM8K()):
        print(prompt)
        diff_now = test_future_max_local_effect(llm, prompt)
        d_max = torch.max(d_max, diff_now)

        dall.append(diff_now)
        if idx == N_EXAMPLES - 1:
            break

    print("--------------------------------")

    fig = plot_layer_diffs(d_max)
    fig.savefig(os.path.join(target_dir, f"{model_name}_future_max_effect_local.pdf"), bbox_inches="tight")

    # fig = plot_logit_diffs(dout_max)
    # fig.savefig(os.path.join(target_dir, f"{model_name}_future_max_effect_out_local.pdf"), bbox_inches="tight")


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
