import os
import nnsight
nnsight.CONFIG.API.APIKEY = os.environ["NDIF_TOKEN"]
import torch
from nnsight import LanguageModel
from lib.igrad import plot_igrads, get_igrads_multiple
import gc
from dm_math_skip import dm_math_prompts
import sys

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <ckpt_path>")
    exit(1)
else:
    ckpt_path = sys.argv[1]

ckpt_path = os.path.expanduser(ckpt_path)
out_name = os.path.splitext(os.path.basename(ckpt_path))[0]

target_dir = "out/dm_math_igrad"
os.makedirs(target_dir, exist_ok=True)



def run_test(llm, prefix):
    igrads, tokens = get_igrads_multiple(llm, [q for _, q in dm_math_prompts], what="layer", cache_name=prefix)
    for (name, (q, a)), igrad, token in zip(dm_math_prompts, igrads, tokens):
        fig = plot_igrads(igrad, token)
        fig.savefig(os.path.join(target_dir, f"dm_math_igrad_{prefix}_{name}.pdf"), bbox_inches="tight", pad_inches=0)

print("Testing instruct model.")
llm = LanguageModel("meta-llama/Llama-3.2-3B-Instruct", device_map="auto", dispatch=True)
llm.eval()
llm.remote = False
run_test(llm, "instruct")
del llm

print("Testing original model.")
gc.collect()
torch.cuda.empty_cache()
llm = LanguageModel("meta-llama/Llama-3.2-3B", device_map="auto", dispatch=True)
llm.eval()
llm.remote = False
run_test(llm, "orig")



print("Loading tuned model.")
gc.collect()
torch.cuda.empty_cache()
new_weights = torch.load(ckpt_path, map_location="cpu")
new_weights["lm_head.weight"] = new_weights["model.embed_tokens.weight"]
llm.load_state_dict(new_weights)
del new_weights
new_weights = None

print("Testing tuned model.")
run_test(llm, f"{out_name}_tuned")