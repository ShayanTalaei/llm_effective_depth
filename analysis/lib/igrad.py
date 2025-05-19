from .nnsight_tokenize import tokenize
import torch
import torch.nn.functional as F
from .matplotlib_config import replace_bos
from .ndif_cache import ndif_cache_wrapper

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


@ndif_cache_wrapper
def get_igrads_multiple(llm, prompt_answer_pairs, what="layer", normalize=False):
    N_STEPS = 256
    BLOCK_SIZE = 16


    if what == "layer":
        def getset_grad_object(llm, l, s=None):
            if s is not None:
                llm.model.layers[l].output = s,

            return llm.model.layers[l].output[0]

    elif what == "attention":
        def getset_grad_object(llm, l, s=None):
            if s is not None:
                llm.model.layers[l].self_attn.output = s, llm.model.layers[l].self_attn.output[1]

            return llm.model.layers[l].self_attn.output[0]
    elif what == "mlp":
        def getset_grad_object(llm, l, s=None):
            if s is not None:
                llm.model.layers[l].mlp.output = s

            return llm.model.layers[l].mlp.output
    else:
        raise ValueError(f"Invalid sublayer: {what}")


    results = []
    alltoks = []
    diffs = []
    with llm.session(remote=getattr(llm, "remote", True)) as session:
        for prompt, answer in prompt_answer_pairs:

            _, atok = tokenize(llm, answer, add_special_tokens=False)

            prompt = prompt + answer

            alen = len(atok)

            _, alltok = tokenize(llm, prompt)
            print(alltok)
            print(f"Answer lenght: {len(atok)}")

            

            igrads = []
            for l in range(len(llm.model.layers)):
                l_igrads = 0

                for step_block in range(0, N_STEPS, BLOCK_SIZE):
                    with llm.trace(prompt) as tracer:
                        # Create a batch out of a single sequence at the target layer
                        if l == 0:
                            llm.model.embed_tokens.output = llm.model.embed_tokens.output.repeat(BLOCK_SIZE, 1, 1)
                        else:
                            llm.model.layers[l-1].output = llm.model.layers[l-1].output[0].repeat(BLOCK_SIZE, 1, 1),

                        gobject = getset_grad_object(llm, l)

                        orig_output = gobject.clone().detach()

                        # Set the baseline to the mean activation
                        baseline = orig_output.mean(dim=1, keepdims=True)

                        # Create a bunch of interpolated activations between the original activation and the baseline which is 0 in this case
                        r = torch.arange(start=step_block, end=min(step_block + BLOCK_SIZE, N_STEPS), device=gobject.device, dtype=gobject.dtype) / N_STEPS
                        target = orig_output * r[:, None, None] + baseline * (1-r[:, None, None])

                        # Overwrite the MLP output with the target
                        gobject = getset_grad_object(llm, l, target)
                        gobject.requires_grad_()

                        # Get the target probability
                        oclasses = F.softmax(llm.output.logits[:, :-1], dim=-1)
                        tid = llm.model.embed_tokens.inputs[0][0][:, 1:]

                        tid = tid.repeat(oclasses.shape[0], 1)
                        oprobs = oclasses.gather(-1, tid.unsqueeze(-1))

                        # Sum grad * activation diff for all different steps
                        igrad = (gobject.grad * (orig_output - baseline)).detach().cpu().float().sum(0)
                        l_igrads = l_igrads + igrad

                        if step_block == 0:
                            orig_probs = oprobs[0, -alen:].detach().clone()
                        elif step_block == N_STEPS - BLOCK_SIZE:
                            baseline_probs = oprobs[-1, -alen:].detach().clone()

                        # Call backward. Should be done after the gardient hooks are set up.
                        oprobs[:, -alen:].sum().backward()

                if normalize:
                    l_igrads = l_igrads / (orig_probs - baseline_probs)

                # Save the grads for this layer
                igrads.append((l_igrads.sum(-1) / N_STEPS))
            
            results.append(torch.stack(igrads, dim=0).save())
            alltoks.append(alltok)

    results = [r.float() for r in results]
    return results, alltoks




def plot_igrads(layer_attributions, tokens):
    fig, ax = plt.subplots(figsize=[5, 5 * max(1, layer_attributions.shape[0] / 30)])

    # Remove the BOS token
    r = layer_attributions[:, 1:].abs().max().item()

    print(tokens)
    tokens = replace_bos(tokens)

    im = ax.imshow(layer_attributions[:, :-1].float().cpu().numpy(), cmap="seismic", vmin=-r, vmax=r, interpolation="nearest")
    plt.xticks(range(len(tokens)-1), tokens[:-1], rotation=45, ha='right',rotation_mode="anchor", fontsize=8)
    plt.ylabel("Layer")
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.1, pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    return fig