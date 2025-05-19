from .datasets import GSM8K
import torch
from .nnsight_tokenize import tokenize
from .ndif_cache import ndif_cache_wrapper
from .sample_pos import sample_pos


@ndif_cache_wrapper
def measure_erasure_future(llm, prompts):
    parsed = []
    for prompt in prompts:
        _, tokens = tokenize(llm, prompt)
        parsed.append(tokens)
        
    results = []
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

            for i, (prompt, tokens) in enumerate(zip(prompts, parsed)):
                with llm.trace(prompt):
                    outs = llm.output.logits.detach().softmax(dim=-1)

                ls = []
                for l in range(len(llm.model.layers) - 1):
                    ts = []
                    for t in range(len(tokens)-1):
                        with llm.trace(prompt):
                            if l == 0:
                                layer = llm.model.layers[0]
                                layer.inputs[0][0][:, t] = baseline_residuals[l] #[:, None]
                            else:
                                layer = llm.model.layers[l-1]
                                layer.output[0][:, t] = baseline_residuals[l] #[:, None]
                            
                            ts.append((outs[:, t+1:] - llm.output.logits[:, t+1:].detach().softmax(dim=-1)).norm(dim=-1).max(dim=1).values.cpu())

                    ls.append(torch.cat(ts, 0))

                results.append(torch.stack(ls, dim=0).save())
        
    return results, parsed



@ndif_cache_wrapper
def measure_erasure_future_multipos(llm, prompts, positions = None):
    parsed = []
    for prompt in prompts:
        _, tokens = tokenize(llm, prompt)
        parsed.append(tokens)
        
    results = []
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

            for i, (prompt, tokens) in enumerate(zip(prompts, parsed)):
                with llm.trace(prompt):
                    outs = llm.output.logits.detach().softmax(dim=-1)

                ls = []
                for l in range(len(llm.model.layers) - 1):
                    ts = []
                    tlist = range(len(tokens)-1) if positions is None else positions[i]
                    for t in tlist:
                        with llm.trace(prompt):
                            if l == 0:
                                layer = llm.model.layers[0]
                                layer.inputs[0][0][:, 1:t] = baseline_residuals[l][:, None]
                            else:
                                layer = llm.model.layers[l-1]
                                layer.output[0][:, 1:t] = baseline_residuals[l][:, None]
                            
                            ts.append((outs[:, t+1:] - llm.output.logits[:, t+1:].detach().softmax(dim=-1)).norm(dim=-1).max(dim=1).values.cpu())

                    ls.append(torch.cat(ts, 0))

                results.append(torch.stack(ls, dim=0).save())
        
    return results, parsed
