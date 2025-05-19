import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Callable
from .helpers import renorm_keep_std
from .transformer import AttentionMask, MultilayerKVCache, TransformerOutput, generate_causal_attention_mask


class LanguageModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, n_tokens: int, d_model: int, n_layers: int,
                 out_norm: Optional[bool] = True, in_norm: Optional[bool] = False,
                 tied: bool = True, preln: Optional[bool] = None):

        torch.nn.Module.__init__(self)

        self.d_model = d_model
        self.transformer = model

        self.n_layers = n_layers
        self.lm_head = torch.nn.Linear(d_model, n_tokens, bias=False)

        self.preln = preln if preln is not None else getattr(self.transformer, "preln", True)
        if in_norm is None:
            in_norm = not self.preln

        if out_norm is None:
            out_norm = self.preln

        self.in_norm = torch.nn.LayerNorm(d_model, elementwise_affine=False) if in_norm else lambda x: x
        self.out_norm = torch.nn.LayerNorm(d_model, elementwise_affine=tied, bias=False) if out_norm else lambda x: x
        self.tied = tied
        if tied:
            self.embedding = lambda x: F.embedding(x, self.lm_head.weight)
        else:
            self.embedding = torch.nn.Embedding(n_tokens, d_model)

        self.reset_parameters()

    @torch.no_grad
    def reset_parameters(self):
        w = self.lm_head.weight if self.tied else self.embedding.weight

        torch.nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="linear")
        if self.tied:
            renorm_keep_std(self.lm_head.weight, 1)
            w.mul_(self.n_layers ** (-1/2))

    def get_output(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out_norm(x)
        x = self.lm_head(x)
        return x

    def forward(self, x: torch.Tensor, mask: Optional[AttentionMask] = None,
                kv_cache: MultilayerKVCache = None, **kwargs) -> Tuple[TransformerOutput, Dict[str, Any]]:
        # Input shape: [B, T]
        # kv_cache: feed an empty dict to start caching
        if mask is None:
            mask = AttentionMask(None, generate_causal_attention_mask(x.shape[-1], x.device))

        x = self.embedding(x)
        x = self.in_norm(x)

        out = self.transformer(x, mask, kv_cache, **kwargs)
        out.outputs = self.get_output(out.outputs)
        return out, {}

    @torch.no_grad()
    def generate_beam(self, prefix: torch.Tensor, ntok: int, beam_size: int, sample: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None):
        mask = AttentionMask(None, generate_causal_attention_mask(len(prefix), prefix.device))
        prefix = prefix[None]

        def repeat_batch(d):
            if torch.is_tensor(d):
                return d.repeat(beam_size, *([1]*(d.ndim-1)))
            elif isinstance(d, dict):
                return {k: repeat_batch(v) for k, v in d.items()}
            else:
                raise ValueError(f"Unknown type: {type(d)}")

        def gather_cache(d, indices):
            if torch.is_tensor(d):
                return d[indices]
            elif isinstance(d, dict):
                return {k: gather_cache(v, indices) for k, v in d.items()}
            else:
                raise ValueError(f"Unknown type: {type(d)}")

        def sample_index_prob(lprobs):
            if sample is None:
                sortprobs = lprobs.sort(dim=-1, descending=True)
                indices = sortprobs.indices[:, :beam_size]
                new_probs = sortprobs.values[:, :beam_size]
            else:
                indices = sample(lprobs, beam_size)
                new_probs = lprobs.gather(dim=-1, index=indices)

            return indices, new_probs

        self.eval()

        out, _ = self(prefix, mask, {})

        toks = F.logsigmoid(out.outputs[0, -1].float())
        gen_tok, probs = sample_index_prob(toks[None])
        gen_tok = gen_tok.T
        probs = probs.squeeze(0)

        cache = repeat_batch(out.cache)

        for _ in range(ntok-1):
            out, _ = self(gen_tok[..., -1:], AttentionMask(None, None), cache)
            lprobs = F.logsigmoid(out.outputs[:, 0].float())
            if sample is None:
                sortprobs = lprobs.sort(dim=-1, descending=True)
                indices = sortprobs.indices[:, :beam_size]
                new_probs = sortprobs.values[:, :beam_size]
            else:
                indices = sample(lprobs, beam_size)
                new_probs = lprobs.gather(dim=-1, index=indices)

            # probs shape: [N_BEAM], new probs shape: [N_BEAM, N_BEAM]
            all_probs = (probs.unsqueeze(-1) + new_probs).flatten().sort(dim=-1, descending=True)
            src_index = all_probs.indices[:beam_size] // beam_size

            probs = all_probs.values[:beam_size]
            src = gen_tok[src_index]

            newstack = indices.flatten()[all_probs.indices[:beam_size]]
            gen_tok = torch.cat([src, newstack.unsqueeze(-1)], dim=-1)

            cache = gather_cache(out.cache, src_index)

        return gen_tok[0]