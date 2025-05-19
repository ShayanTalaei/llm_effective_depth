from .datasets import GSM8K
import torch
from .nnsight_tokenize import tokenize
from .matplotlib_config import replace_bos
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .ndif_cache import ndif_cache_wrapper

def plot_logit_effect(ls, tokens):
    ls = ls[:, :-1]
    tokens = tokens[:-1]
    tokens = replace_bos(tokens)
    fig, ax = plt.subplots(figsize=(5,5 * max(1, ls.shape[0] / 30)))
    im = ax.imshow(ls.float().cpu().numpy())
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right',rotation_mode="anchor", fontsize=8)
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.1, pad=0.1)
    cbar = fig.colorbar(im, cax=cax, label='Probability Difference Norm')
    return fig


@ndif_cache_wrapper
def measure_token_skip(llm, prompts):
    parsed = {}
    for name, (prompt, answer) in prompts:
        _, tokens = tokenize(llm, prompt)
        _, atokens = tokenize(llm, answer, add_special_tokens=False)

        prompt = prompt + answer

        _, tokens = tokenize(llm, prompt)

        parsed[name] = (prompt, tokens, len(atokens))
        
    results = {}
    with llm.session(remote=getattr(llm, "remote", True)) as session:
        with torch.no_grad():
            baseline_residuals = {
                i: 0 for i in range(len(llm.model.layers)+1)
            }
            count = 0

            for i, prompt in enumerate(GSM8K()):
                if i >= 10:
                    break

                with llm.trace(prompt):
                    for l in range(len(llm.model.layers)):
                        if l == 0:
                            baseline_residuals[0] += llm.model.layers[0].inputs[0][0].detach().sum(dim=1).cpu().float()
                        baseline_residuals[l+1] += llm.model.layers[l].output[0].detach().sum(dim=1).cpu().float()
                    count += llm.model.layers[0].output[0].shape[1]

            baseline_residuals = {k: (v / count) for k, v in baseline_residuals.items()}

            for name, (prompt, tokens, alen) in parsed.items():
                with llm.trace(prompt):
                    outs = llm.output.logits[:, -(alen+1):-1].detach().softmax(dim=-1)

                ls = []
                for l in range(len(llm.model.layers) + 1):
                    ts = []
                    for t in range(len(tokens)):
                        with llm.trace(prompt):
                            if l == 0:
                                layer = llm.model.layers[0]
                                layer.inputs[0][0][:, t] = baseline_residuals[l][:, None]
                            else:
                                layer = llm.model.layers[l-1]
                                layer.output[0][:, t] = baseline_residuals[l][:, None]
                            
                            ts.append((outs - llm.output.logits[:, -(alen+1):-1].detach().softmax(dim=-1)).norm(dim=-1).max(dim=1).values.cpu())

                    ls.append(torch.cat(ts, 0))

                results[name] = torch.stack(ls, dim=0).save()    
        
    return results, parsed