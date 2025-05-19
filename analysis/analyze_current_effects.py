from lib.matplotlib_config import sort_zorder
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
import os
import nnsight
nnsight.CONFIG.API.APIKEY =  os.environ["NDIF_TOKEN"]
import torch
import torch.nn.functional as F
import random
import datasets
from nnsight import LanguageModel

from lib.models import create_model
from lib.nnsight_tokenize import tokenize
from lib.datasets import GSM8K


from analyze_future_effects import plot_layer_diffs, plot_logit_diffs, test_effect


def test_current_max_effect(llm, prompt, N_CHUNKS=4, part = "layer"):
    return test_effect(llm, prompt, [None], part)


def main():
    random.seed(12345)

    N_EXAMPLES = 20
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        raise ValueError("Please provide a model name")

    llm = create_model(model_name)
    target_dir = "out/current_effects"

    llm.eval()

    os.makedirs(target_dir, exist_ok=True)

    for what in ["layer", "mlp", "attention"]:
        dall = []
        d_max = None
        dout_max = torch.zeros([1])
        for idx, prompt in enumerate(GSM8K()):
            print(prompt)
            diff_now, diff_out = test_current_max_effect(llm, prompt, part=what)
            d_max = torch.max(d_max, diff_now) if d_max is not None else diff_now
            dout_max = torch.max(dout_max, diff_out)

            dall.append(diff_now)
            if idx == N_EXAMPLES - 1:
                break

        fig = plot_layer_diffs(d_max)
        fig.savefig(os.path.join(target_dir, f"{model_name}_current_max_effect_{what}.pdf"), bbox_inches="tight")

        fig = plot_logit_diffs(dout_max)
        fig.savefig(os.path.join(target_dir, f"{model_name}_current_max_effect_out_{what}.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
