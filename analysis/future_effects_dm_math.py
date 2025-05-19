import gc
import torch
from nnsight import LanguageModel
import os
from analyze_future_effects import test_effect, plot_layer_diffs, plot_logit_diffs, tokenize
import random
import sys


if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <ckpt_path>")
    exit(1)
else:
    ckpt_path = sys.argv[1]

ckpt_path = os.path.expanduser(ckpt_path)

target_dir = "out/dm_math_future_effects"
os.makedirs(target_dir, exist_ok=True)


prompts = [
    ("What is ((-14)/(-6))/(1162/(-4980))?", "-10"),
    ("What is the value of -24 + -10 + 22 + 1 + 1 + 25?", "-15"),
    ("What is 1 - 3 - (63 + 7 - 60)?", "-12"),
    ("(10/3)/((-8)/16)*(-90)/(-75)", "-8"),
    ("Evaluate (-230)/(-23) + 19 - (0 - -4).", "25"),
    ("What is -2*899/(-124) + (-10 - 9/3)?", "3/2"),
    ("Simplify -2 + -4*(5 + (sqrt(2057) + 2)**2 - (4*(-2 + sqrt(2057)))**2) + 1.", "-2992*sqrt(17) + 123639"),
    ("Simplify 3*2*-3*(sqrt(832) + sqrt(832)*2 - sqrt(832))**2*1*-1*-1.", "-59904"),
    ("What is 20/10*(-60)/32*(-504)/70?", "27"),
    ("Evaluate ((-196)/(-112) + 258/(-152))/(2/6).", "3/19"),
    ("Simplify ((-1 + 6*((sqrt(1100) - (-1 + sqrt(1100)))*2 + 3) + 1)*-4)**2.", "14400"),
    ("((-7)/(-9))/(11109/(-504) + 21) - 4/(-50)", "-2/3"),
    ("Calculate 13 + -1 + (-61)/((-3904)/1736) + 2/(-16).", "39"),
    ("Evaluate (-441)/(-3)*-21*(-21)/1323.", "49"),
    ("What is -11 + -6 + 39 + -8 + -54?", "-40"),
    ("What is -22 - ((-3)/(-7) - 93015/4081)?", "4/11"),
    ("Simplify 5 + -2*(sqrt(180)*1*2 - sqrt(180)) + 4 + (sqrt(180)*2 + -4 - sqrt(180)) + 4.","-6*sqrt(5) + 9")
]


def test_future_max_effect(llm, prompt, N_CHUNKS=4, part = "layer", cache_name=None):
    _, tokens = tokenize(llm, prompt)

    positions = list(range(3, len(tokens)-1))
    random.shuffle(positions)
    positions = positions[:N_CHUNKS]
    # positions = [10]

    return test_effect(llm, prompt, positions, part, cache_name=cache_name)

def run_test(llm, prefix):
    random.seed(21345)
    d_max = None
    dout_max = torch.zeros([1])
    for idx, (q,a) in enumerate(prompts):
        prompt = f"Q: {q} A: {a}"
        print(prompt)
        diff_now, diff_out = test_future_max_effect(llm, prompt, part="layer", cache_name=prefix)
        d_max = torch.max(d_max, diff_now) if d_max is not None else diff_now
        dout_max = torch.max(dout_max, diff_out)


    print("--------------------------------")

    fig = plot_layer_diffs(d_max)
    fig.savefig(os.path.join(target_dir, f"{prefix}_future_max_effect.pdf"), bbox_inches="tight")

    fig = plot_logit_diffs(dout_max)
    fig.savefig(os.path.join(target_dir, f"{prefix}_future_max_effect_out.pdf"), bbox_inches="tight")



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

print("Testing original model.")
run_test(llm, "original")


print("Loading tuned model.")
gc.collect()
torch.cuda.empty_cache()
new_weights = torch.load(ckpt_path, map_location="cpu")
new_weights["lm_head.weight"] = new_weights["model.embed_tokens.weight"]
llm.load_state_dict(new_weights)
del new_weights
new_weights = None


print("Testing tuned model.")
run_test(llm, "tuned")

