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
from analyze_future_local_effects import test_local_effect




def run(llm, model_name):
    N_EXAMPLES = 10

    random.seed(123)

    target_dir = "out/curret_local_effects"

    os.makedirs(target_dir, exist_ok=True)

    dall = []
    d_max = torch.zeros([1])
    for idx, prompt in enumerate(GSM8K()):
        print(prompt)
        diff_now = test_local_effect(llm, prompt, [None])
        d_max = torch.max(d_max, diff_now)

        dall.append(diff_now)
        if idx == N_EXAMPLES - 1:
            break

    print("--------------------------------")

    fig = plot_layer_diffs(d_max)
    fig.savefig(os.path.join(target_dir, f"{model_name}_current_max_effect_local.pdf"), bbox_inches="tight")


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
