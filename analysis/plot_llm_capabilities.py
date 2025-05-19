import requests
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, HfFileSystem
import json
from lib import matplotlib_config
import math


URL = "https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/releases/v1.13.0/groups/core_scenarios.json"
MODEL_URL = "https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/releases/v1.13.0/schema.json"

target_dir = "out/helm"
os.makedirs(target_dir, exist_ok=True)

data = requests.get(URL).json()
models = requests.get(MODEL_URL).json()

def fix_name(name):
    if name.startswith("meta/"):
        name = "meta-llama"+name[4:]

    if name.endswith("-turbo"):
        name = name[:-6]

    return name

model_to_kindof_huggingface_map = {m["display_name"]: fix_name(m["name"]) for m in models["models"]}

def parse_model_size(model_name):
    size_block = model_name.split("(")[1]
    assert size_block.endswith(")") and size_block[-2] == "B"
    return int(size_block[:-2])


def get_no_of_layers(model_name):
    name = model_to_kindof_huggingface_map[model_name]

    fs = HfFileSystem()
    files = fs.ls(name, detail=False)
    config_path = None
    for file in files:
        if file.endswith(".json"):
            if config_path is not None:
                raise RuntimeError(f"Found multiple config files in {name}: {config_path} and {file}")
            
            if file.startswith(name):
                file = file[len(name)+1:]
            config_path = file
            break

    config_path = hf_hub_download(
        repo_id=name,
        filename=config_path,
        # optional: where to put it instead of the default cache
        cache_dir="./cache/huggingface_hub/"
    )

    with open(config_path, "r") as f:
        config = json.load(f)

    if "num_hidden_layers" in config:
        return config["num_hidden_layers"]
    else:
        return config["n_layers"]

# As the 2402 models completely disappeared from the internet, and no source of their size is available,
# after spending 3 hours googling, I assume that they have the same size as the 2407 models. At this
# point, only the God himself knows how big they were.
uninformative_name_map = {
    "Mistral Small (2402)": "Mistral (22B)",
    "Mistral Large (2402)": "Mistral (123B)",
    "Mistral NeMo (2402)": "Mistral (12B)",
}


models = OrderedDict()
models["Llama 2"] = "Llama 2"
models["Llama 3.1 Instruct"] = "Llama 3.1 Instruct Turbo"
models["Qwen1.5 "] = "Qwen1.5"
models["Qwen2.5 Instruct"] = "Qwen2.5 Instruct Turbo"
# models["Mistral"] = "Mistral"

model_data = []
for model_name, model_helm_name in models.items():
    points = {}
    for row in data[0]["rows"]:
        table_model_name = row[0]['value']
        table_model_name = uninformative_name_map.get(table_model_name, table_model_name)

        if table_model_name.startswith(model_helm_name+" ("):
            size = parse_model_size(table_model_name)
            n_layers = get_no_of_layers(table_model_name)
            score = row[1]['value']
            points[size] = (score, n_layers)
    
    if points:
        x = list(sorted(points.keys()))
        y = [points[xi][0] for xi in x]
        n_layers = [points[xi][1] for xi in x]
        model_data.append((model_name, x, y, n_layers))

plt.figure(figsize=(4,3))
for model_name, x, y, _ in model_data:
    plt.plot(x, y, label=model_name, marker="o", markersize=6)
plt.legend()
plt.xscale("log")
plt.xlabel("Model Size (B)")
plt.ylabel("HELM Lite score")
plt.ylim(0, 1)
plt.savefig(os.path.join(target_dir, "helm_lite_score.pdf"), bbox_inches="tight", pad_inches=0.0)


plt.figure(figsize=(8,3))
for model_name, x, y, n_layers in model_data:
    line, = plt.plot(n_layers, y, label=model_name)
    plt.scatter(n_layers, y, s=[math.sqrt(a) * 5 for a in x], color=line.get_color())

plt.legend()
# plt.xscale("log")
plt.xlabel("Number of layers")
plt.ylabel("HELM Lite score")
plt.savefig(os.path.join(target_dir, "helm_lite_score_vs_layers.pdf"), bbox_inches="tight", pad_inches=0.0)

