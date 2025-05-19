import torch

@torch.no_grad
def renorm_keep_std(weight: torch.Tensor, dim: int):
    std = weight.std()
    weight.div_(weight.norm(dim=dim, keepdim=True))
    weight.mul_(std / weight.std())
