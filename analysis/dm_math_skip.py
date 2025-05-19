import os
import nnsight
nnsight.CONFIG.API.APIKEY = os.environ["NDIF_TOKEN"]
import torch
from nnsight import LanguageModel
from lib.token_skip import measure_token_skip, plot_logit_effect
import gc
import sys


dm_math_prompts = [
    ("arithmetic_1", ("Q: What is ((-14)/(-6))/(1162/(-4980))? A: ", "-10")),
    ("arithmetic_2", ("Q: What is the value of -24 + -10 + 22 + 1 + 1 + 25? A: ", "-15")),
    ("arithmetic_3", ("Q: What is 1 - 3 - (63 + 7 - 60)? A: ", "-12")),
    ("arithmetic_4", ("Q: (10/3)/((-8)/16)*(-90)/(-75) A: ", "-8")),
]

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <ckpt_path>")
        exit(1)
    else:
        ckpt_path = sys.argv[1]

    ckpt_path = os.path.expanduser(ckpt_path)
    out_name = os.path.splitext(os.path.basename(ckpt_path))[0]

    ckpt_path = os.path.expanduser(ckpt_path)

    target_dir = "out/dm_math_skip"
    os.makedirs(target_dir, exist_ok=True)


    def run_test(llm, prefix):
        results, parsed = measure_token_skip(llm, dm_math_prompts, cache_name=prefix)
        for name, ls in results.items():
            fig = plot_logit_effect(ls, parsed[name][1])
            fig.savefig(os.path.join(target_dir, f"dm_math_skip_{prefix}_{name}.pdf"), bbox_inches="tight", pad_inches=0)

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
    run_test(llm, f"tuned_{out_name}")

if __name__ == "__main__":
    main()
