import torch
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math
from .rope import RotaryPosEncoding
import torch.nn.functional as F
import framework
from framework.data_structures import DotDict
from framework.layers import LayerWithVisualization
from framework.task import args


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-dbgvis.n_steps", default=128)
    parser.add_argument("-dbgvis.mha.plot_head_details", default=False)
    parser.add_argument("-dbgvis.mha.plot_head_logits", default=False)
    parser.add_argument("-dbgvis.mha.limit_head_details_to_first_n", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-dbgvis.mha.plot_head_max", default=False)
    parser.add_argument("-dbgvis.mha.plot_entropy", default=False)


KVCache = Optional[Dict[str, torch.Tensor]]
MultilayerKVCache = Optional[Dict[int, KVCache]]


@dataclass
class AttentionMask:
    src_length_mask: Optional[torch.Tensor] = None
    position_mask: Optional[torch.Tensor] = None
    soft_mask: Optional[torch.Tensor] = None


def get_mask_tensor(src_len: int, mask: Optional[AttentionMask]) -> Optional[torch.Tensor]:
    if mask is None or (mask.position_mask is None and mask.src_length_mask is None):
        return mask.soft_mask

    # mask.position_mask: [..., N_out, N_in]
    # mask.src_length_mask: [B, ...., N_in]
    # True where it has to be masked

    if mask.position_mask is not None:
        n_pad = src_len - mask.position_mask.shape[-1]
        if n_pad > 0:
            pm = F.pad(mask.position_mask, (n_pad, 0), 'constant', value=False)
        else:
            pm = mask.position_mask

    if mask.position_mask is None:
        m = mask.src_length_mask.unsqueeze(-2).unsqueeze(-2)
    elif mask.src_length_mask is None:
        m = pm
    else:
        m = mask.src_length_mask.unsqueeze(-2).unsqueeze(-2) | pm

    if mask.soft_mask is not None:
        m = mask.soft_mask.masked_fill(m, float('-inf'))

    return m


class AttentionCore(torch.nn.Module):
    def __init__(self, state_size: int, n_heads: int, dropout: float = 0.0, projection_size: Optional[int] = None):
        super().__init__()

        self.input_size = state_size
        self.output_size = state_size

        self.n_heads = n_heads
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else lambda x: x
        self.projection_size = projection_size or (state_size // n_heads)

        self.q = torch.nn.Linear(self.input_size, self.projection_size * self.n_heads, bias=False)
        self.k = torch.nn.Linear(self.input_size, self.projection_size * self.n_heads, bias=False)
        self.v = torch.nn.Linear(self.input_size, self.projection_size * self.n_heads, bias=False)
        self.o = torch.nn.Linear(self.projection_size * self.n_heads, self.output_size, bias=False)

        self.register_buffer("scale", torch.full([1], 1.0 / math.sqrt(self.projection_size)), persistent=False)

    def project_to_torch_order(self, x: torch.Tensor):
        return x.view(*x.shape[:-1], self.n_heads, -1).transpose(-2, -3)

    def attend(self, pos_offset: int, v: torch.Tensor, k: torch.Tensor, q: torch.Tensor,
               mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def get_qk_scale(self) -> torch.Tensor:
        return self.scale.sqrt()

    def forward(self, q_src: torch.Tensor, k_src: torch.Tensor, v_src: torch.Tensor, mask: Optional[AttentionMask],
                kv_cache: KVCache = None) -> Tuple[torch.Tensor, KVCache]:
        q = self.q(q_src)
        k = self.k(k_src)
        v = self.v(v_src)

        scale = self.get_qk_scale()

        q = self.project_to_torch_order(q * scale)
        k = self.project_to_torch_order(k * scale)
        v = self.project_to_torch_order(v)

        if kv_cache is not None:
            v = torch.cat([kv_cache["v"], v], dim=-2) if "v" in kv_cache else v
            k = torch.cat([kv_cache["k"], k], dim=-2) if "k" in kv_cache else k
            kv_cache = {
                "v": v,
                "k": k
            }

        pos_offset = k.shape[-2] - q.shape[-2]
        assert pos_offset >= 0

        q = self.dropout(q)
        res = self.attend(pos_offset, v, k, q, get_mask_tensor(v.shape[-2], mask))
        res = res.transpose(-2, -3).flatten(-2)
        res = self.o(res)

        return res, kv_cache


class AttentionWithVisualization(LayerWithVisualization):
    def __init__(self):
        LayerWithVisualization.__init__(self)
        self.attention_to_visualize = []
        self.attention_logits_to_visualize = []
        self.attention_entropy = []

    def attention_visualization_enabled(self) -> bool:
        return self.visualization_enabled and (
            self.visualization_options.mha.plot_entropy or
            self.visualization_options.mha.plot_head_details or
            self.visualization_options.mha.plot_head_max or
            self.visualization_options.mha.plot_head_logits
        )

    def plot(self, options: DotDict) -> Dict[str, Any]:
        r = {}
        if self.attention_visualization_enabled():
            marks = options.get("steplabel")
            n_steps = options.n_steps or 9999999
            # y_marks = options.get("target_labels", marks)
            y_marks = marks

            tlen_y = self.attention_to_visualize[0].shape[-2]
            tlen_x = self.attention_to_visualize[0].shape[-1]

            if marks is not None:
                assert len(marks) <= tlen_x
                tlen_x = len(marks)

            if y_marks is not None:
                assert len(y_marks) <= tlen_y
                tlen_y = len(y_marks)

            ns1 = max((tlen_y + n_steps),0) if n_steps < 0 else 0
            ns1_e = tlen_y if n_steps < 0 else min(n_steps, tlen_y)
            ns2 = max((tlen_x + n_steps),0) if n_steps < 0 else 0
            ns2_e = tlen_x if n_steps < 0 else min(n_steps, tlen_x)

            if marks is not None:
                marks = marks[ns2:ns2_e]

            if y_marks is not None:
                y_marks = y_marks[ns1:ns1_e]

            all_layers = torch.stack([l[..., ns1:ns1_e, ns2:ns2_e] for l in self.attention_to_visualize], 0)
            if self.attention_logits_to_visualize:
                all_logits = torch.stack([l[..., ns1:ns1_e, ns2:ns2_e] for l in self.attention_logits_to_visualize], 0)
                all_logits = all_logits.masked_fill(~all_logits.isfinite(), 0)

            def plot_heatmap(heatmap):
                return framework.visualize.plot.AnimatedHeatmap(
                    heatmap, ylabel="dest", xlabel="src", textval=False, x_marks=marks, y_marks=y_marks,
                    ignore_wrong_marks=True, fps=1, shape_factor=0.15)

            if options.mha.plot_head_details and self.attention_to_visualize[0].shape[0] > 1:
                n_heads = self.attention_to_visualize[0].shape[0]
                if options.mha.limit_head_details_to_first_n is not None:
                    n_heads = min(n_heads, options.mha.limit_head_details_to_first_n)

                for head in range(n_heads):
                    r[f"head_{head}"] = plot_heatmap(all_layers[:, head])
                    if options.mha.plot_head_logits:
                        r[f"head_logits_{head}"] = plot_heatmap(all_logits[:, head].float())

            if options.mha.plot_head_max:
                r["attention_max"] = plot_heatmap(all_layers.max(1).values)
                if options.mha.plot_head_logits:
                    r["attention_max_logits"] = plot_heatmap(all_logits.max(1).values.float())

            if options.mha.plot_entropy:
                for i, e in enumerate(self.attention_entropy):
                    r[f"mean_entropy_{i}"] = e

        self.attention_to_visualize = []
        self.attention_entropy = []
        self.attention_logits_to_visualize = []
        return r


class BasicAttentionCore(AttentionWithVisualization):
    def __init__(self, activation = None, avoid_sdpa = False):
        super().__init__()
        self.activation = activation
        self.avoid_sdpa = avoid_sdpa

    def attend(self, pos_offset: int, v: torch.Tensor, k: torch.Tensor, q: torch.Tensor,
               mask: Optional[torch.Tensor]) -> torch.Tensor:

        vis_now = self.attention_visualization_enabled()

        if vis_now or (mask is not None and mask.requires_grad) or self.activation is not None or self.avoid_sdpa:
            att = q @ k.transpose(-2, -1)
            if mask is not None:
                if mask.dtype == torch.bool:
                    att.masked_fill_(mask, float('-inf'))
                elif mask.dtype in {torch.float16, torch.bfloat16, torch.float32}:
                    att = att + mask
                else:
                    raise ValueError(f"Unsupported mask type: {mask.dtype}")

            if vis_now and self.visualization_options.mha.plot_head_logits:
                self.attention_logits_to_visualize.append(att[0].detach().clone())

            if self.activation is not None:
                att = self.activation(att)
            else:
                att = F.softmax(att, dim=-1)

            if vis_now:
                with torch.no_grad():
                    entropy = -(att * (att.clamp(torch.finfo(att.dtype).tiny)).log()).mean()
                    self.attention_entropy.append(entropy)

                self.attention_to_visualize.append(att[0].detach().clone())

            return att @ v
        else:
            if mask is not None and mask.dtype == torch.bool:
                mask = ~mask
            return F.scaled_dot_product_attention(q, k, v,mask, scale=1.0)


class RopeCore(BasicAttentionCore):
    def __init__(self, projection_size: int, rotate_fraction: float = 0.5, rope_base: float = 10000, activation = None, avoid_sdpa = False):
        super().__init__(activation, avoid_sdpa)
        self.projection_size = projection_size
        self.n_rotate = int(rotate_fraction * self.projection_size)
        self.n_rotate -= self.n_rotate % 2

        if self.n_rotate > 0:
            self.pe = RotaryPosEncoding(self.n_rotate, seq_dim=-2, base=rope_base)

    def rope(self, q: torch.Tensor, k: torch.Tensor, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pe(q, k, offset)

    def rotate(self, q: torch.Tensor, k: torch.Tensor, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n_rotate < self.projection_size:
            r_k = k[..., :self.n_rotate]
            nr_k = k[..., self.n_rotate:]
            r_q = q[..., :self.n_rotate]
            nr_q = q[..., self.n_rotate:]

            r_q, r_k = self.rope(r_q, r_k, offset)
            return torch.cat([r_q, nr_q], dim=-1), torch.cat([r_k, nr_k], dim=-1)
        else:
            return self.rope(q, k, offset)

    def attend(self, pos_offset: int, v: torch.Tensor, k: torch.Tensor, q: torch.Tensor,
               mask: Optional[torch.Tensor]) -> torch.Tensor:

        if self.n_rotate > 0:
            q, k = self.rotate(q, k, pos_offset or 0)

        return super().attend(pos_offset, v, k, q, mask)


class RopeAttention(RopeCore, AttentionCore):
    def __init__(self, state_size: int, n_heads: int, dropout: float = 0.0, projection_size: Optional[int] = None,
                 rotate_fraction: float = 0.5, rope_base: float = 10000, activation = None, avoid_sdpa: bool = False):

        AttentionCore.__init__(self, state_size, n_heads, dropout, projection_size)
        RopeCore.__init__(self, self.projection_size, rotate_fraction, rope_base, activation, avoid_sdpa)


class BasicAttention(BasicAttentionCore, AttentionCore):
    def __init__(self, state_size: int, n_heads: int, dropout: float = 0.0, projection_size: Optional[int] = None,
                 activation = None, avoid_sdpa: bool = False):
        AttentionCore.__init__(self, state_size, n_heads, dropout, projection_size)
        BasicAttentionCore.__init__(self, activation, avoid_sdpa)
