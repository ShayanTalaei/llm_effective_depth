
import torch

from framework.data_structures.dotdict import DotDict
from .transformer import TransformerOutput, init_parameters, TransformerVisualizer
from .attention import MultilayerKVCache, AttentionMask
from typing import Optional, Dict, Any
from framework.layers import LayerWithVisualization
import framework
from framework.task import args
import math
import torch.nn.functional as F



class UniversalTransformer(TransformerVisualizer, torch.nn.Module):
    def __init__(self, create_layer, d_model: int, n_layers: int, group_size: int = 2, init_scale: Optional[float] = None):
        TransformerVisualizer.__init__(self)
        torch.nn.Module.__init__(self)

        self.d_model = d_model

        self.n_repeats = n_layers // group_size
        self.layers = torch.nn.ModuleList([
            create_layer() for _ in range(group_size)
        ])

        self.activations = []
        self.init_parameters(init_scale)

    def forward(self, x: torch.Tensor, mask: Optional[AttentionMask] = None,
                kv_cache: MultilayerKVCache = None) -> TransformerOutput:
        # Run the model
        new_cache = {}

        self.visualize_activations(x)

        for r in range(self.n_repeats):
            for li, layer in enumerate(self.layers):
                li_abs = r*len(self.layers)+li
                cache = kv_cache.get(li_abs, {}) if kv_cache is not None else None
                x, new_cache[li_abs] = layer(x, mask, kv_cache = cache, strength=1)

                self.visualize_activations(x)

        return TransformerOutput(x, new_cache if kv_cache is not None else None)

    def init_parameters(self, scale: Optional[float] = None):
        scale = math.sqrt(2 / (self.n_repeats * len(self.layers))) if scale is None else scale
        init_parameters(self, scale)
