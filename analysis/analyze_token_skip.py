from lib.matplotlib_config import sort_zorder
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import sys
import nnsight
nnsight.CONFIG.API.APIKEY =  os.environ["NDIF_TOKEN"]
import torch
import torch.nn.functional as F
import random
import datasets
from nnsight import LanguageModel
from lib.matplotlib_config import replace_bos

from lib.models import create_model
from lib.nnsight_tokenize import tokenize
from lib.datasets import GSM8K
from lib.token_skip import measure_token_skip, plot_logit_effect


manual_prompts = [
    ("math", ("5 + 7 + 5 + 3 + 1 + 7 = ", "28")),
    ("question", ("The spouse of the performer of Imagine is", " Yoko Ono"))
]


def main():
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        raise ValueError("Please provide a model name")

    # model_name = "llama_3.1_70b"`
    llm = create_model(model_name)
    target_dir = "out/token_skip"

    llm.eval()

    os.makedirs(target_dir, exist_ok=True)

   
    results, parsed = measure_token_skip(llm, manual_prompts)

    for name, ls in results.items():
        fig = plot_logit_effect(ls, parsed[name][1])
        fig.savefig(os.path.join(target_dir, f"{model_name}_skip_token_logit_effect_{name}.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()