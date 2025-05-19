import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math
from layers.cvmm import cvmm, cvmm_prepare_sel2, CVMMSel
from dataclasses import dataclass
from .helpers import renorm_keep_std
from .transformer import TransformerOutput
from .attention import AttentionMask, get_mask_tensor, MultilayerKVCache, KVCache, RopeCore
from .universal_transformer import UniversalTransformer
from framework.layers import RegularizedLayer, LengthMaskedLayer
from .rope import RotaryPosEncoding


def log_mean(x: torch.Tensor, dim: int = 0, mask: Optional[torch.Tensor] = None):
    if mask is not None:
        cnt = mask.sum(dim, keepdim=True)
        zcnt = cnt == 0
        n = cnt.clamp(1).log()
        x = x.masked_fill(~mask, float("-inf"))
    else:
        n = math.log(x.shape[dim])

    res = x.logsumexp(dim) - n
    if mask is not None:
        return res.masked_fill(zcnt, 0)

    return res


def entropy_l(l: torch.Tensor) -> torch.Tensor:
    return - (l * l.exp()).sum(-1)


def entropy_reg(sel: torch.Tensor, dim: int, mask: Optional[torch.Tensor]) -> torch.Tensor:
    sel = F.log_softmax(sel, dim=-1)
    sel = log_mean(sel, dim, mask)
    return - entropy_l(sel).mean()


class SigmaMoE(LengthMaskedLayer, torch.nn.Module):
    def __init__(self, dmodel: int, n_experts: int, expert_size: int, k: int,
                 activation=F.relu,
                 v_dim: Optional[int] = None,
                 expert_dropout: float = 0.0,
                 init_to_mlp_size: Optional[float] = None):

        torch.nn.Module.__init__(self)
        LengthMaskedLayer.__init__(self)

        self.k_dim = dmodel
        self.v_dim = v_dim if v_dim is not None else dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.size = self.n_experts * self.expert_size
        self.k_vec_dim = self.k_dim
        self.n_heads = k
        self.activation = activation
        self.expert_dropout = expert_dropout
        self.init_to_mlp_size = init_to_mlp_size

        self.sel_hist = []

        self.keys = torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim, self.expert_size))
        self.values = torch.nn.Parameter(torch.empty(self.n_experts, self.expert_size, self.v_dim))
        self.expert_sel = torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim))

    @torch.no_grad
    def init_parameters(self, std_scale: float):
        torch.nn.init.normal_(self.expert_sel, 0, std_scale / math.sqrt(self.k_dim))
        torch.nn.init.normal_(self.keys, 0, std_scale / math.sqrt(self.k_dim))
        vinit_size = self.init_to_mlp_size if self.init_to_mlp_size is not None else self.n_experts * self.expert_size
        torch.nn.init.normal_(self.values, 0, std_scale / math.sqrt(vinit_size))

        renorm_keep_std(self.expert_sel, dim=1)

    def get_reg_loss(self) -> torch.Tensor:
        if not self.sel_hist:
            return 0

        # Average over time and layers.
        if self.length_mask is not None:
            length_mask = self.length_mask.repeat(1, len(self.sel_hist))[..., None]
        else:
            length_mask = None

        loss = entropy_reg(torch.cat(self.sel_hist, dim=-2), -2, length_mask)
        self.sel_hist = []
        return loss

    def get_sel(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.expert_sel, None)

    def forward(self, input: torch.Tensor, sel_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Selection score calculation
        sel = self.get_sel(sel_input if sel_input is not None else input)
        if self.training:
            self.sel_hist.append(sel)

        # Selection activation and topk
        sel = F.sigmoid(sel)

        if self.training and self.expert_dropout > 0:
            mask = torch.rand_like(sel) < self.expert_dropout
            sel = sel.masked_fill(mask, float("-inf"))

        sel_val, sel_index = sel.topk(self.n_heads, dim=-1, sorted=False)

        # Preprocess the selection indices. They will be needed for both layers and save some time
        sel_indices = cvmm_prepare_sel2(sel_index.int())

        # "Up-projection" layer for each head
        scores = cvmm(input, sel_indices, self.keys)
        scores = self.activation(scores)

        # Down projection layer for each head
        sel_indices = sel_indices.clone()
        sel_indices.reduction_weight = sel_val
        sel_indices.sel_index = sel_indices.out_index
        sel_indices.out_index = None

        out = cvmm(scores, sel_indices, self.values)

        res = out.view(*input.shape[:-1], self.v_dim)
        return res


class SwitchHeadCore(LengthMaskedLayer, torch.nn.Module):
    def __init__(self, state_size: int, n_heads: int, n_experts: int, dropout: float = 0.0,
                 projection_size: Optional[int] = None, expert_dropout: float = 0.0, moe_k: int = 2,
                 vin_size: Optional[int] = None, kin_size: Optional[int] = None):

        torch.nn.Module.__init__(self)
        LengthMaskedLayer.__init__(self)

        self.input_size = state_size
        self.output_size = state_size
        self.pe_size = self.input_size
        self.expert_dropout = expert_dropout
        self.moe_k = moe_k
        self.n_experts = n_experts
        self.vin_size = vin_size or self.input_size
        self.kin_size = kin_size or self.input_size

        self.sel_hist = []

        self.n_heads = n_heads
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else lambda x: x
        self.projection_size = projection_size or (state_size // n_heads)

        self.q = torch.nn.Linear(self.input_size, self.projection_size * self.n_heads, bias=False)
        self.k = torch.nn.Linear(self.kin_size, self.projection_size * self.n_heads, bias=False)

        if self.n_experts > 1:
            self.v = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.vin_size, self.projection_size))
            self.o = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.projection_size, self.output_size))
            self.sel_v = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.input_size))
        else:
            self.v = torch.nn.Parameter(torch.empty(self.n_heads * self.projection_size, self.vin_size))
            self.o = torch.nn.Parameter(torch.empty(self.output_size, self.n_heads * self.projection_size))

        self.sel_o = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.input_size))

        self.register_buffer("scale", torch.full([1], 1.0 / math.sqrt(self.projection_size)), persistent=False)

    @torch.no_grad
    def init_sel(self, w: torch.nn.Parameter, std_scale: float):
        torch.nn.init.normal_(w, 0, std_scale / math.sqrt(self.input_size))
        self.renorm_rows(w)

    @torch.no_grad
    def init_parameters(self, std_scale: float):
        if self.n_experts > 1:
            self.init_sel(self.sel_v, std_scale)

        self.init_sel(self.sel_o, std_scale)

        torch.nn.init.normal_(self.k.weight, 0, std_scale / math.sqrt(self.kin_size))
        torch.nn.init.normal_(self.q.weight, 0, std_scale / math.sqrt(self.input_size))
        torch.nn.init.normal_(self.v, 0, std_scale / math.sqrt(self.vin_size))
        torch.nn.init.normal_(self.o, 0, std_scale / math.sqrt(self.n_heads * self.projection_size))

    def renorm_rows(self, x: torch.Tensor):
        with torch.no_grad():
            std_t = x.std(dim=-1, keepdim=True)
            x.div_(x.norm(dim=-1, keepdim=True))
            x.mul_(std_t / x.std())

    def project_to_torch_order(self, x: torch.Tensor):
        return x.view(*x.shape[:-1], self.n_heads, -1).transpose(-2, -3)

    def get_sel_score(self, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        sel = F.linear(t, w)
        return sel.view(*sel.shape[:-1], self.n_heads, -1)

    def get_sel(self, t: torch.Tensor, w: torch.Tensor) -> Tuple[CVMMSel, torch.Tensor]:
        sel_raw = self.get_sel_score(t, w).float()
        sel = sel_raw.sigmoid()

        with torch.no_grad():
            if self.expert_dropout > 0 and self.training:
                mask = torch.rand_like(sel) < self.expert_dropout
                sel2 = sel.masked_fill(mask, float('-inf'))
            else:
                sel2 = sel
            _, sel_index = sel2.topk(self.moe_k, dim=-1, sorted=False)
        sel_val = torch.gather(sel, -1, sel_index)

        sel_index_shifted = (torch.arange(self.n_heads, device=sel_index.device, dtype=sel_index.dtype) * self.n_experts).unsqueeze(-1) + sel_index
        return cvmm_prepare_sel2(sel_index_shifted.flatten(-2,-1), sel_val), sel_raw

    def attend(self, pos_offset: int, v: torch.Tensor, k: torch.Tensor, q: torch.Tensor,
               mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def get_reg_loss(self) -> torch.Tensor:
        loss = 0

        if self.length_mask is not None:
            len_mask = self.length_mask.repeat(1, len(self.sel_hist))[..., None, None]
        else:
            len_mask = None

        if self.sel_hist:
            for i in range(len(self.sel_hist[0])):
                loss = loss + entropy_reg(torch.cat([l[i] for l in self.sel_hist], dim=-3), -3, len_mask)
        self.sel_hist = []
        return loss

    def forward(self, q_src: torch.Tensor, k_src: torch.Tensor, v_src: torch.Tensor, mask: Optional[AttentionMask],
                kv_cache: KVCache = None) -> Tuple[torch.Tensor, KVCache]:
        # *src: [batch_size, out_len, c]

        scale = self.scale.sqrt()

        q = self.q(q_src)
        k = self.k(k_src)
        q = q * scale.type_as(q)
        k = k * scale.type_as(k)

        if self.n_experts > 1:
            v_sel, v_sel_r = self.get_sel(k_src, self.sel_v)
            o_sel, o_sel_r = self.get_sel(q_src, self.sel_o)
            if self.training:
                self.sel_hist.append((o_sel_r, v_sel_r))

            v = cvmm(v_src, v_sel, self.v).transpose(-2, -3)
        else:
            o_gate = F.sigmoid(F.linear(q_src, self.sel_o))
            v = self.project_to_torch_order(F.linear(v_src, self.v))

        q = self.project_to_torch_order(q)
        k = self.project_to_torch_order(k)

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
        res = res.transpose(-2, -3)

        if self.n_experts > 1:
            # The output selection indices are calculated from the current state and are also used for projecting "q".
            # But that projection needs to create multiple copies for the different heads. Here we already have the
            # heads, but we have to create copies for the top-k elements. We can calculate that from the reduction
            # weight. We also want to compute not only the weighted average between the top-k elements, but also
            # of the different heads. So reshape the reduction weight accordingly.
            o_sel.sel_index = o_sel.out_index // o_sel.reduction_weight.shape[-1]
            o_sel.reduction_weight = o_sel.reduction_weight.flatten(-2)
            out = cvmm(res, o_sel, self.o)
        else:
            res = res * o_gate[..., None]
            out = F.linear(res.flatten(-2), self.o)

        return out, kv_cache


class SwitchHeadRope(RopeCore, SwitchHeadCore):
    def __init__(self, state_size: int, n_heads: int, n_experts: int, dropout: float = 0.0,
                 projection_size: Optional[int] = None, expert_dropout: float = 0.0, moe_k: int = 2,
                 vin_size: Optional[int] = None, kin_size: Optional[int] = None,
                 rotate_fraction: float = 0.5, rope_base: float = 10000):

        SwitchHeadCore.__init__(self, state_size, n_heads, n_experts, dropout, projection_size, expert_dropout, moe_k,
                                vin_size, kin_size)
        RopeCore.__init__(self, self.projection_size, rotate_fraction, rope_base)


class MoEUTLayer(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_expert_size: int, ff_n_experts: int,
                 att_n_experts: int, att_proj_size: Optional[int] = None, att_k: int = 2,
                 ff_k: int = 8, ff_expert_dropout: float = 0.0, att_expert_dropout: float = 0.0,
                 dropout: float = 0.0, attention = SwitchHeadRope, mlp = SigmaMoE):

        super().__init__()
        self.attention = attention(
            d_model, n_heads, att_n_experts, projection_size=att_proj_size, moe_k=att_k,
            expert_dropout=att_expert_dropout)
        self.ffn = mlp(d_model, ff_n_experts, ff_expert_size, k=ff_k, expert_dropout=ff_expert_dropout)
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[AttentionMask] = None, kv_cache: KVCache = None,
                strength = 1) -> Tuple[torch.Tensor, KVCache]:
        xnorm = self.ln1(x)
        att, kv_cache = self.attention(xnorm, xnorm, x, mask, kv_cache=kv_cache)
        x = x + strength * self.drop(att)
        upd = self.ffn(x, self.ln2(x))
        return x + strength * self.drop(upd), kv_cache


class MoEUT(UniversalTransformer, RegularizedLayer):
    def __init__(self, create_layer, d_model: int, n_layers: int, group_size: int = 2, init_scale: Optional[float] = None,
                 entropy_reg: float = 0.01, att_entropy_reg: float = 0.001):
        UniversalTransformer.__init__(self, create_layer, d_model, n_layers, group_size, init_scale)
        RegularizedLayer.__init__(self)

        self.entropy_reg = entropy_reg
        self.att_entropy_reg = att_entropy_reg

    def collect_losses(self, device) -> torch.Tensor:
        # Collect regularizaiton losses. Must be at the end because it is across the layers.
        reg_loss = torch.zeros(1, device=device, dtype=torch.float32)
        for layer in self.modules():
            if isinstance(layer, SigmaMoE):
                reg_loss = reg_loss + self.entropy_reg * layer.get_reg_loss()
            elif isinstance(layer, SwitchHeadCore):
                reg_loss = reg_loss + self.att_entropy_reg * layer.get_reg_loss()
        return reg_loss

    def forward(self, x: torch.Tensor, mask: Optional[AttentionMask] = None,
                kv_cache: MultilayerKVCache = None) -> TransformerOutput:
        res = super().forward(x, mask, kv_cache)
        self.add_reg(lambda: self.collect_losses(x.device), "moeut_reg")
        return res


class MoEUTPrelnLayer(MoEUTLayer):
    def forward(self, x: torch.Tensor, mask: Optional[AttentionMask] = None, kv_cache: KVCache = None,
                strength = 1) -> Tuple[torch.Tensor, KVCache]:
        xnorm = self.ln1(x)
        att, kv_cache = self.attention(xnorm, xnorm, xnorm, mask, kv_cache=kv_cache)
        x = x + strength * self.drop(att)
        upd = self.ffn(self.ln2(x))
        return x + strength * self.drop(upd), kv_cache
