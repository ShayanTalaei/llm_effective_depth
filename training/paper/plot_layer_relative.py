import lib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


runs = lib.get_runs(["map_qwen"])
offs = 1
target_key = "relative"
extra_args ="-test_batch_size 2"

maps = {}

src_name = None
tgt_name = None

for r in runs:
    src_layer = r.config["map_layers.src_layer"]
    if src_layer not in maps:
        maps[src_layer] = {}

    src_name = r.config["map_layers.src"]
    tgt_name = r.config["map_layers.tgt"]

    val_results = lib.validate(r, flags=f"-lmds.n_validation_token 1000000 {extra_args}")

    for k, v in val_results.items():
        if not k.startswith(f"val/{target_key}/layer_"):
            continue

        layer = int(k.split("_")[-1])
        maps[src_layer][layer] = v

map_matrix = np.zeros((len(maps)+offs, len(maps[1])+offs))

for src_layer, layer_map in maps.items():
    for layer, value in layer_map.items():
        map_matrix[src_layer, layer] = value

src_name = src_name.split("/")[-1]
tgt_name = tgt_name.split("/")[-1]


fig, ax = plt.subplots(figsize=(6,4))
im = ax.imshow(map_matrix[1:,1:])
plt.ylabel(f"{src_name} Layer (source)")
plt.xlabel(f"{tgt_name} Layer (target)")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size=0.2, pad=0.1)
cbar = fig.colorbar(im, cax=cax, label='Relative error')
plt.savefig(f"layer_{target_key}_map_matrix_2.pdf", bbox_inches="tight", pad_inches=0.05)



