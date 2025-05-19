import os
import nnsight
nnsight.CONFIG.API.APIKEY =  os.environ["NDIF_TOKEN"]
from nnsight import LanguageModel
import matplotlib.pyplot as plt
import tqdm

from lib.matplotlib_config import sort_zorder
from lib.models import create_model
from lib.igrad import plot_igrads, get_igrads_multiple
from lib.datasets import MQuake, Math
import random
from lib.nnsight_tokenize import tokenize
import torch
from analyze_future_effects import test_effect, plot_layer_diffs
from lib.nnsight_tokenize import tokenize
import sys

def plot_output_change(llm, hops, max_effects, format="{} hops"):
    plt.figure(figsize=(6,3))
    bars = []
    for h, maxval in zip(hops, max_effects):
        maxval = maxval.squeeze()
        bars.append(plt.bar([x for x in range(maxval.shape[0])], maxval.float().cpu().numpy(), label=format.format(h)))
    # plt.xticks([x for x in range(len(llm.model.layers)) if x % 2 == 0], [f"L{i}" for i in range(len(llm.model.layers)) if i % 2 == 0], fontsize=12)
    plt.legend()
    sort_zorder(bars)
    plt.xlim(-1, len(llm.model.layers))
    plt.xlabel("Layer")
    return plt


def effect_layer_score(effects):
    return [(torch.arange(m.shape[0]) * m.squeeze()).sum().item()/m.sum().item() for m in effects]


def plot_per_depth_summary(llm, hops, max_effects, format="{} hops"):
    plt.figure(figsize=(6,3))
    scores = effect_layer_score(max_effects)
    plt.bar([x for x in range(len(hops))], scores)
    plt.xticks([x for x in range(len(hops))], hops)
    plt.ylabel("Depth score")
    return plt

def plot_per_depth_summary_both(hops, max_layer_effects, max_logit_effects, format="{} hops"):
    plt.figure(figsize=(6,2))
    scores_layer = effect_layer_score(max_layer_effects)
    scores_logit = effect_layer_score(max_logit_effects)
    plt.bar([x*3 for x in range(len(hops))], scores_layer, label="Layer effect")
    plt.bar([x*3+1 for x in range(len(hops))], scores_logit, label="Logit effect")
    plt.xticks([x*3 + 0.5 for x in range(len(hops))], hops)
    plt.ylabel("Depth score")
    plt.legend(loc='lower right')
    return plt


def main():
    random.seed(12345)
    N_EXAMPLES = 20
    
    Dataset = MQuake
    N_POINTS_PER_ANSWER = None
    

    if len(sys.argv) < 3:
        raise ValueError("Please provide a model name and dataset name")

    model_name = sys.argv[1]
    dataset_name = sys.argv[2].lower()

    if dataset_name == "mquake":
        Dataset = MQuake
        N_POINTS_PER_ANSWER = None
    elif dataset_name == "math":
        Dataset = Math
        N_POINTS_PER_ANSWER = 4
    else:
        raise ValueError("Invalid dataset name")

    llm = create_model(model_name, force_local=False)
    target_dir = "out/skip_depth2"

    llm.eval()

    os.makedirs(target_dir, exist_ok=True)


    hops = Dataset.levels()
    max_effects = []
    max_douts = []
    mean_effects = []
    mean_douts = []

    with tqdm.tqdm(total=len(hops) * N_EXAMPLES) as pbar:
        for nhops in hops:
            max_dout = torch.zeros([1])
            max_dall = torch.zeros([1])

            mean_dout = 0
            mean_dall = 0
            n = 0
            
            for i, d in enumerate(Dataset(nhops)):
                if i >= N_EXAMPLES:
                    break
                
                prompt = d[0] + d[1]
                _, tokens = tokenize(llm, prompt)
                _, atokens = tokenize(llm, d[1], add_special_tokens=False)

                a_start_pos = len(tokens) - len(atokens)
                if N_POINTS_PER_ANSWER is None:
                    pos = [a_start_pos]
                else:
                    pos = list(range(0, len(atokens)-4, 4))
                    random.shuffle(pos)
                    pos = pos[:N_POINTS_PER_ANSWER]

                    pos = [a_start_pos + x for x in pos]

                dall, dall_out = test_effect(llm, prompt, pos, "layer")

                max_dout = torch.max(max_dout, dall_out)
                max_dall = torch.max(max_dall, dall)

                mean_dout += dall_out
                mean_dall += dall
                n += 1

                pbar.update(1)

            mean_dout /= n
            mean_dall /= n

            dall_reduce = max_dall.sum(-1) / torch.arange(len(llm.model.layers)-1, -1, -1).clamp(min=1)
            dall_avg_reduce = mean_dall.sum(-1) / torch.arange(len(llm.model.layers)-1, -1, -1).clamp(min=1)

            fig = plot_layer_diffs(max_dall)
            fig.savefig(os.path.join(target_dir, f"{Dataset.__name__}_{model_name}_max_effect_on_last__{nhops}.pdf"), bbox_inches="tight")

            max_douts.append(max_dout)
            max_effects.append(dall_reduce)
            mean_douts.append(mean_dout)
            mean_effects.append(dall_avg_reduce)


    plot_output_change(llm, hops, max_effects, format=Dataset.level_format())
    plt.ylabel("Max output change")
    plt.ylim(0,1)
    plt.savefig(os.path.join(target_dir, f"{Dataset.__name__}_{model_name}_max_skip_logit_change_hops.pdf"), bbox_inches="tight")

    plot_per_depth_summary(llm, hops, max_effects, format=Dataset.level_format())
    plt.savefig(os.path.join(target_dir, f"{Dataset.__name__}_{model_name}_max_skip_logit_change_hops_summary.pdf"), bbox_inches="tight")

    plot_output_change(llm, hops, mean_effects, format=Dataset.level_format())
    plt.ylabel("Mean output change")
    plt.ylim(0,1)
    plt.savefig(os.path.join(target_dir, f"{Dataset.__name__}_{model_name}_mean_skip_logit_change_hops.pdf"), bbox_inches="tight")

    plot_per_depth_summary(llm, hops, mean_effects, format=Dataset.level_format())
    plt.savefig(os.path.join(target_dir, f"{Dataset.__name__}_{model_name}_mean_skip_logit_change_hops_summary.pdf"), bbox_inches="tight")


    plot_output_change(llm, hops, max_douts, format=Dataset.level_format())
    plt.ylabel("Max relative change")
    plt.ylim(0,1)
    plt.savefig(os.path.join(target_dir, f"{Dataset.__name__}_{model_name}_max_layer_change_hops.pdf"), bbox_inches="tight")

    plot_per_depth_summary(llm, hops, max_douts, format=Dataset.level_format())
    plt.savefig(os.path.join(target_dir, f"{Dataset.__name__}_{model_name}_max_layer_change_hops_summary.pdf"), bbox_inches="tight")

    plot_output_change(llm, hops, mean_douts, format=Dataset.level_format())
    plt.ylabel("Mean relative change")
    plt.ylim(0,1)
    plt.savefig(os.path.join(target_dir, f"{Dataset.__name__}_{model_name}_mean_layer_change_hops.pdf"), bbox_inches="tight")

    plot_per_depth_summary(llm, hops, mean_douts, format=Dataset.level_format())
    plt.savefig(os.path.join(target_dir, f"{Dataset.__name__}_{model_name}_mean_layer_change_hops_summary.pdf"), bbox_inches="tight")

    plot_per_depth_summary_both(hops, max_effects, max_douts, format=Dataset.level_format())
    plt.savefig(os.path.join(target_dir, f"{Dataset.__name__}_{model_name}_max_skip_hops_summary_both.pdf"), bbox_inches="tight")

    plot_per_depth_summary_both(hops, mean_effects, mean_douts, format=Dataset.level_format())
    plt.savefig(os.path.join(target_dir, f"{Dataset.__name__}_{model_name}_mean_skip_hops_summary_both.pdf"), bbox_inches="tight")

    
if __name__ == "__main__":
    main()