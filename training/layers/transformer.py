from dataclasses import dataclass
from typing import Optional, Dict, Callable, Tuple, Any
import torch
from .attention import MultilayerKVCache, AttentionMask, KVCache
import torch.nn.functional as F
import math
import framework
from framework.layers import LayerWithVisualization, LoggingLayer
from framework.data_structures import DotDict
from framework.task import args

@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-dbgvis.residual_diff", default=False)
    parser.add_argument("-dbgvis.residual_norm", default=False)


@dataclass
class TransformerOutput:
    outputs: torch.Tensor
    cache: MultilayerKVCache


def generate_causal_attention_mask(sz: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)


class TransformerFFN(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0,
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
                 d_out: Optional[int] = None):
        super().__init__()
        d_out = d_out or d_model
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.linear2 = torch.nn.Linear(d_ff, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))



class PreLNTransformerLayer(LoggingLayer, torch.nn.Module):
    def __init__(self, attention: torch.nn.Module, ffn: torch.nn.Module,
                 d_model: int, dropout: float = 0.0):

        torch.nn.Module.__init__(self)
        LoggingLayer.__init__(self)
        self.iter = 0
        self.attention = attention
        self.ffn = ffn
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[AttentionMask] = None, kv_cache: KVCache = None,
                strength = 1) -> Tuple[torch.Tensor, KVCache]:

        xnorm = self.ln1(x)
        att, kv_cache = self.attention(xnorm, xnorm, xnorm, mask, kv_cache=kv_cache)
        x = x + strength * self.drop(att)
        upd = self.ffn(self.ln2(x))

        if self.training:
            if self.iter % 20 == 0:
                self.log("attention_norm", att.detach().norm(dim=-1).mean())
                self.log("ffn_norm", upd.detach().norm(dim=-1).mean())
            self.iter += 1

        return x + strength * self.drop(upd), kv_cache


def init_parameters(model, scale: float):
    for layer in model.children():
        if layer is model:
            continue

        if hasattr(layer, "init_parameters"):
            layer.init_parameters(scale)
        elif isinstance(layer, torch.nn.LayerNorm):
            if layer.weight is not None:
                torch.nn.init.ones_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
        elif isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, 0, scale / math.sqrt(layer.weight.shape[-1]))
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
        elif hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        else:
            init_parameters(layer, scale)


class TransformerVisualizer(LayerWithVisualization):
    def __init__(self):
        LayerWithVisualization.__init__(self)
        self.activations = []
        self.activations_norm = []

    def visualize_activations(self, activations: torch.Tensor):
        if self.visualization_enabled:
            self.activations.append(activations[0].detach().clone())
            self.activations_norm.append(activations.detach().norm(dim=-1).mean())

    def plot(self, options: DotDict) -> Dict[str, Any]:
        res = {}

        if options.residual_diff:
            marks = options.get("steplabel")
            n_steps = options.n_steps or 9999999

            tlen = self.activations[0].shape[0]
            if marks is not None:
                # Handle padding
                assert len(marks) <= tlen
                tlen = len(marks)

            ns1 = (tlen + n_steps) if n_steps < 0 else 0
            ns1_e = tlen if n_steps < 0 else min(n_steps, tlen)

            if marks is not None:
                marks = marks[ns1:ns1_e]

            activations = torch.stack([l[ns1:ns1_e] for l in self.activations], 0)
            cossim = F.cosine_similarity(activations[1:], activations[:-1], dim=-1)
            cos_first = F.cosine_similarity(activations, activations[:1], dim=-1)
            cos_last = F.cosine_similarity(activations, activations[-1:], dim=-1)

            res["residual_cosdist"] = framework.visualize.plot.Heatmap(
                1-cossim, ylabel="layer", xlabel="token", textval=False, x_marks=marks, flip_y=True)

            # res["residual_cosdist_to_first"] = framework.visualize.plot.Heatmap(
            #     1-cos_first, ylabel="layer", xlabel="token", textval=False, x_marks=marks, flip_y=True)

            # res["residual_cosdist_to_last"] = framework.visualize.plot.Heatmap(
            #     1-cos_last, ylabel="layer", xlabel="token", textval=False, x_marks=marks, flip_y=True)


        if options.residual_norm:
            for i, n in enumerate(self.activations_norm):
                res[f"residual_norm/layer_{i}"] = n

        self.activations = []
        self.activations_norm = []
        return res


class Transformer(TransformerVisualizer, torch.nn.Module):
    def __init__(self, create_layer, n_layers: int, init_scale: Optional[float] = None, init_params: bool = True):
        torch.nn.Module.__init__(self)
        TransformerVisualizer.__init__(self)

        self.layers = torch.nn.ModuleList([
            create_layer() for _ in range(n_layers)
        ])

        self.activations = []
        if init_params:
            self.init_parameters(init_scale)

    def forward(self, x: torch.Tensor, mask: Optional[AttentionMask] = None,
                kv_cache: MultilayerKVCache = None) -> TransformerOutput:
        # Run the model
        new_cache = {}

        self.visualize_activations(x)

        for i, layer in enumerate(self.layers):
            cache = kv_cache.get(i, {}) if kv_cache is not None else None
            x, new_cache[i] = layer(x, mask, kv_cache = cache, strength=1)

            self.visualize_activations(x)

        return TransformerOutput(x, new_cache if kv_cache is not None else None)

    def init_parameters(self, scale: Optional[float] = None):
        scale = math.sqrt(2 / len(self.layers)) if scale is None else scale
        init_parameters(self, scale)
