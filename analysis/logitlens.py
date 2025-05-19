import matplotlib.pyplot as plt

import os
import nnsight
nnsight.CONFIG.API.APIKEY =  os.environ["NDIF_TOKEN"]
import torch
import random

from lib.models import create_model
from lib.datasets import GSM8K
from lib.ndif_cache import ndif_cache_wrapper
import torch.nn.functional as F

import sys

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} model")
    exit(1)
else:
    model_name = sys.argv[1]

N_EXAMPLES = 10
# model_name = "llama_3.1_8b"
# model_name = "llama_3.1_8b_instruct"
# model_name = "llama_3.1_70b"
# model_name = "llama_3.1_70b_instruct"
# model_name = "llama_3.1_405b_instruct"
target_dir = "out/logitlens"

random.seed(123123)

llm = create_model(model_name)
llm.eval()

os.makedirs(target_dir, exist_ok=True)


@ndif_cache_wrapper
def run_logitlens(llm, prompts, K=5):
    res_kl_divs = []
    res_topks = []

    with llm.session(remote=llm.remote) as session:
        with torch.no_grad():
            for prompt in prompts:
                kl_divs = []
                topks = []
                layer_logs = []        
                with llm.trace(prompt):
                    for l in range(len(llm.model.layers)):
                        tap = llm.model.layers[l].inputs[0][0]
                        layer_logs.append(llm.lm_head(llm.model.norm(tap)).detach().float())
                    out_logits = llm.output.logits

                lout = out_logits.float().log_softmax(-1)
                otopl = lout.topk(K, dim=-1).indices

                for l in range(len(llm.model.layers)):
                    llayer = layer_logs[l].log_softmax(-1)

                    kl_divs.append((llayer.exp() * (llayer - lout)).sum(-1).mean().detach())

                    itopl = llayer.topk(K, dim=-1).indices
                    topks.append(itopl.save())

                topks.append(otopl.save())

                res_kl_divs.append(torch.stack(kl_divs, dim=0).save())
                res_topks.append(topks)

    res_topk_overlaps = []
    for topks in res_topks:
        real_topk = topks[-1]
        other_topks = topks[:-1]

        real_oh = F.one_hot(real_topk, llm.model.embed_tokens.weight.shape[0]).sum(-2)
        overlaps = [(F.one_hot(ot.to(real_oh.device), llm.model.embed_tokens.weight.shape[0]).sum(-2).unsqueeze(-2).float() @ real_oh.unsqueeze(-1).float() / K).mean() for ot in other_topks]
        overlaps = torch.stack(overlaps, dim=0)
        res_topk_overlaps.append(overlaps)

    return [d.cpu() for d in res_kl_divs], res_topk_overlaps


N_EXAMPLES = 20

accu_kl_div = 0
accu_topk_overlaps = 0

for i, prompt in enumerate(GSM8K()):
    if i >= N_EXAMPLES:
        break
    kl_divs, topk_overlaps = run_logitlens(llm, [prompt])
    accu_kl_div = accu_kl_div + kl_divs[0]
    accu_topk_overlaps = accu_topk_overlaps + topk_overlaps[0]

accu_kl_div = accu_kl_div / N_EXAMPLES
accu_topk_overlaps = accu_topk_overlaps / N_EXAMPLES

plt.figure(figsize=(5,2))
plt.bar(range(len(llm.model.layers)), accu_kl_div.cpu().numpy())
plt.ylabel("KL Divergence")
plt.xlabel("Layer")
plt.savefig(os.path.join(target_dir, f"{model_name}_logitlens_kl_div.pdf"), bbox_inches="tight")

plt.figure(figsize=(5,2))
plt.bar(range(len(llm.model.layers)), accu_topk_overlaps.cpu().numpy())
plt.ylabel("Overlap")
plt.xlabel("Layer")
plt.savefig(os.path.join(target_dir, f"{model_name}_logitlens_topk_overlaps.pdf"), bbox_inches="tight")
