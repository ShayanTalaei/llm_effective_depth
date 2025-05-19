import torch
import torch.nn
from typing import Dict, Any, Callable, Tuple, Optional, Set, Union, Iterable


class RegularizedLayer:
    def __init__(self) -> None:
        self.reg_accumulated = {}
        self.regularization_present = False
        self.reg_len_mask = None
        self.reg_mask_sum = None

    def set_length_mask(self, mask: Optional[torch.Tensor]):
        if mask is None:
            self.reg_len_mask = None
            self.reg_mask_sum = None
        else:
            self.reg_len_mask = ~mask.squeeze()
            self.reg_mask_sum = mask.sum()

    @property
    def reg_enabled(self) -> bool:
        return self.training and self.regularization_present

    def _reduce_loss(self, loss: torch.Tensor, allow_mask_reshape: bool = False) -> torch.Tensor:
        if torch.is_tensor(loss) and loss.numel() > 1:
            if self.reg_len_mask is not None:
                loss = loss.squeeze()
                if not allow_mask_reshape:
                    assert loss.shape == self.reg_len_mask.shape
                else:
                    loss = loss.view_as(self.reg_len_mask)

                loss = loss.masked_fill(self.reg_len_mask, 0).sum() / self.reg_mask_sum.clamp(1)
            else:
                loss = loss.mean()

        return loss

    def add_reg(self, l: Callable[[], torch.Tensor], name="reg", allow_mask_reshape=False):
        if self.reg_enabled:
            loss = self._reduce_loss(l(), allow_mask_reshape)
            self.reg_accumulated[name] = self.reg_accumulated.get(name, 0) + loss

    def get_reg_loss(self) -> Dict[str, torch.Tensor]:
        rl = self.reg_accumulated
        self.reg_accumulated = {}
        return rl


class LayerRegularizer:
    def __init__(self, module: Union[torch.nn.Module, Iterable[torch.nn.Module]], stop_after: Optional[int] = None, scales: Dict[str, float] = {},
                 lin_decay: Set[str] = set(), options: Dict[str, Any] = {}):

        self.modules = []
        self.scales = scales
        self.stop_after = stop_after
        self.lin_decay = set(lin_decay)

        if self.lin_decay and stop_after is None:
            raise ValueError("Please specify stop_after when using lin_decay.")

        if isinstance(module, torch.nn.Module):
            self.add_module(module)
        else:
            for m in module:
                self.add_module(m)

    def set_length_mask(self, mask: Optional[torch.Tensor]):
        if mask is not None and mask.all():
            mask = None

        for _, m in self.modules:
            if isinstance(m, RegularizedLayer):
                m.set_length_mask(mask)

    def add_module(self, module: torch.nn.Module):
        for n, m in module.named_modules():
            if isinstance(m, RegularizedLayer):
                self.modules.append((n, m))
                m.regularization_present = True

    def get(self, iter: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        res = {}
        for _, m in self.modules:
            for k, v in m.get_reg_loss().items():
                res[k] = res.get(k, 0) + v

        to_log = {k: v.detach() if torch.is_tensor(v) else v for k, v in res.items()}

        for k, v in res.items():
            res[k] = v * self.scales.get(k, 1)

        for k in self.lin_decay:
            res[k] *= 1 - iter / self.stop_after
        return sum(res.values()), to_log
