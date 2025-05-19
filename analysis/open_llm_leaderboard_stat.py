import requests
from huggingface_hub import hf_hub_download, HfFileSystem
import json
import pandas as pd
import math
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

URL = "https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/leaderboard/formatted"
data = requests.get(URL).json()

target_dir = "out/open_llm_leaderboard"
os.makedirs(target_dir, exist_ok=True)

data_per_name = {d["model"]["name"]: d for d in data}

fs = HfFileSystem()

def filter(d):
    f = {
        "is_merged": False,
        "is_moe": False,
        "is_not_available_on_hub": True
    }

    m = {
        "precision": "bfloat16",
        "type": "pretrained",
    }

    for n, v in m.items():
        if d["model"][n] != v:
            return False

    for n, v in f.items():
        if d["features"][n] != v:
            return False
    
    return True

d2 = [d for d in data if filter(d)]

def get_model_data(name):
    files = fs.ls(name, detail=False)
    config_path = None
    if f"{name}/config.json" in files:
        config_path = "config.json"
    else:
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


    res = {}
    if "num_hidden_layers" in config:
        res["n_layers"] = config["num_hidden_layers"]
    elif "n_layers" in config:
        res["n_layers"] = config["n_layers"]
    else:
        return None


    if "hidden_size" in config:
        res["d_model"] = config["hidden_size"]
    elif "dim" in config:
        res["d_model"] = config["dim"]
    elif "d_model" in config:
        res["d_model"] = config["d_model"]
    else:
        return None

    if "intermediate_size" in config:
        res["d_ff"] = config["intermediate_size"]
    elif "expansion_ratio" in config:
        res["d_ff"] = config["expansion_ratio"] * res["d_model"]
    elif "falcon" in name:
        res["d_ff"] = 4  * res["d_model"]
    else:
        return None

    if "activation" in config:
        res["act"] = config["activation"]
    elif "hidden_act" in config:
        res["act"] = config["hidden_act"]
    elif "hidden_activation" in config:
        res["act"] = config["hidden_activation"]
    elif "ffn_type" in config:
        res["act"] = config["ffn_type"]
    elif ("falcon" in name) or ("expansion_ratio" in config and config["expansion_ratio"] == 4):
        res["act"] = "relu"
    else:
        return None
    
    res["n_params"] = data_per_name[name]["metadata"]["params_billions"] * 1e9
    
    return res


def compensate_for_gated(d):
    if d["act"] in {"silu", "swiglu"}:
        # Compensate for the fact that gated activation. Pretend it's the closes non-gated
        d=dict(d)
        d["d_ff"] = d["d_ff"] * 1.5
    return d

model_data = {d["model"]["name"]: get_model_data(d["model"]["name"]) for d in d2}

data_fixed = {k: compensate_for_gated(v) for k, v in model_data.items() if v is not None}


dffs = []
dmodels = []
nlayers = []
perf = []
nparams = []
names = []
for k, v in data_fixed.items():
    dffs.append(v["d_ff"])
    dmodels.append(v["d_model"])
    nlayers.append(v["n_layers"])
    perf.append(data_per_name[k]["model"]["average_score"])
    nparams.append(v["n_params"])
    names.append(k)

print("Number of models:", len(names))


df = pd.DataFrame({
    "name": names,
    "dff": dffs, #[d/maxdff for d in dffs],
    "dmodel": dmodels, #[d/maxdmodel for d in dmodels],
    "nlayers": nlayers,
    "log_nparams": [math.log(n) for n in nparams],
    "perf": [p for p in perf]
})

df.to_csv(os.path.join(target_dir, "data.csv"), index=False)

# Define predictors and response
# X = df[['dff', 'dmodel', 'nlayers']]
# X = df[['dff', 'nlayers', 'log_nparams']]
X = df[['nlayers']]
X = sm.add_constant(X)  # Adds intercept term
y = df['perf']

# Fit the model
model = sm.OLS(y, X).fit()

print(model.summary())

# View the summary
# print(model.summary())

families = set(n.split("/")[0] for n in names)
family_map = {f: i for i, f in enumerate(sorted(families))}
cmap = plt.cm.get_cmap('tab10', len(families))

conf_intervals = model.conf_int(alpha=0.05)

plt.figure(figsize=(5,3))
plt.scatter(nlayers, perf, [math.sqrt(a/1e9) * 10 for a in nparams], color=[cmap(family_map[n.split("/")[0]]) for n in names], alpha=0.5)
minlayers = min(nlayers)
maxlayers = max(nlayers)
plt.plot([minlayers, maxlayers], [model.params["const"] + model.params["nlayers"]*minlayers , model.params["const"] + model.params["nlayers"]*maxlayers], color="red")
plt.fill_between(
    [minlayers, maxlayers],
    [conf_intervals.loc["nlayers",0]*minlayers + conf_intervals.loc["const",0], conf_intervals.loc["nlayers",0]*maxlayers + conf_intervals.loc["const",0]],
    [conf_intervals.loc["nlayers",1]*minlayers + conf_intervals.loc["const",1], conf_intervals.loc["nlayers",1]*maxlayers + conf_intervals.loc["const",1]],
    color="red", alpha=0.05)
plt.xlabel("Number of layers")
plt.ylabel("Open LLM Leaderboard Score")
plt.savefig(os.path.join(target_dir, "open_llm_leaderboard_score_vs_layers.pdf"), bbox_inches="tight", pad_inches=0.0)


print("--------- dff, nlayers, log_nparams ---------")


# X = df[['dff', 'dmodel', 'nlayers']]
# X = df[['dff', 'nlayers', 'log_nparams']]
X = df[['nlayers', 'dff', "log_nparams"]]
X = sm.add_constant(X)  # Adds intercept term
y = df['perf']

# Fit the model
model = sm.OLS(y, X).fit()

# View the summary
print(model.summary())

print("--------- dff, nlayers ---------")

X = df[['dff', 'nlayers']]
X = sm.add_constant(X)  # Adds intercept term
y = df['perf']

model = sm.OLS(y, X).fit()

print(model.summary())



