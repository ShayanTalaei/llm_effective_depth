import torch
from typing import Optional

class LengthMaskedLayer:
    def __init__(self):
        self.length_mask = None

    def set_inv_mask(self, mask: Optional[torch.Tensor]):
        self.length_mask = mask


def set_length_mask(model, mask: Optional[torch.Tensor]):
    if mask is not None and mask.all():
        mask = None

    for module in model.modules():
        if isinstance(module, LengthMaskedLayer):
            module.set_inv_mask(mask)
