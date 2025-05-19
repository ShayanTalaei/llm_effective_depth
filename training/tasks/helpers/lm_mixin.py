import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Any, List, Optional
from framework.interfaces import RecurrentResult
import framework
import torch.utils.data
from framework.layers import set_length_mask


class LMTaskMixin:
    NO_OUTPUT_TRACKING = True
    ALLOW_VARYING_BATCH_SIZE = True

    def __init__(self, mask_name: str ="mask", ignore_index: int = -1,
                 loss_mask_name: Optional[str] = None) -> None:
        self.batch_dim = 1
        self.mask_name = mask_name
        self.ignore_index = ignore_index
        self.loss_mask_name = mask_name if loss_mask_name is None else loss_mask_name

    def loss(self, logits: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        target = data["data"].long()

        if self.loss_mask_name in data:
            target = target.masked_fill(~data[self.loss_mask_name].bool(), self.ignore_index)

        target = target.narrow(self.time_dim, 1, data["data"].shape[self.time_dim] - 1)
        return F.cross_entropy(logits.flatten(end_dim=-2), target.flatten(), ignore_index=self.ignore_index)

    def get_steplabels(self, data: Dict[str, torch.Tensor]) -> List[str]:
        d = self.train_set.vocabulary(data["data"][:, 0].cpu().numpy().tolist())
        return d[:-1], d[1:]

    def prepare_visualizer(self, data: Dict[str, Any]):
        inp, outp = self.get_steplabels(data)
        params = {"steplabel": inp, "target_labels": outp}
        self.visualizer.prepare(params)

    def _run_model(self, input: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        r, p = self.model(input)
        return r.outputs, p

    def run_model(self, data, ubatch: int) -> Tuple[RecurrentResult, Dict[str, Any]]:
        input = data["data"]

        input = input.narrow(self.time_dim, 0, data["data"].shape[self.time_dim] - 1)

        if self.mask_name in data:
            mask = data[self.mask_name].narrow(self.time_dim, 1, data["data"].shape[self.time_dim] - 1)
            input = input.masked_fill(~mask.bool(), self.ignore_index)

        if self.time_dim == 0:
            input = input.transpose(0, 1)

        length_mask = input != self.ignore_index
        self.regularizer.set_length_mask(length_mask)
        for m in self.models.values():
            set_length_mask(m, length_mask)

        input = input.long().clamp(0)
        res, plots = self._run_model(input)

        if self.time_dim == 0:
            res = res.transpose(0, 1)

        loss = self.loss(res, data)
        return RecurrentResult(res, loss), plots

    def validation_decode_outputs(self, out: Any) -> Any:
        return out.outputs

    def create_sampler(self, loader: torch.utils.data.Dataset, batch_size: int, allow_uneven: bool = False) -> \
                       framework.loader.sampler.MultibatchSequentialSampler:

        return framework.loader.sampler.MultibatchSequentialSampler(loader, batch_size,
                            world_size=self.helper.dist_env.world_size, rank=self.helper.dist_env.rank,
                            allow_uneven=allow_uneven)

    def create_valid_loader(self, vset: torch.utils.data.Dataset,
                            batch_size: Optional[int] = None) -> torch.utils.data.DataLoader:

        if batch_size is None:
            batch_size = self.test_batch_size

        return torch.utils.data.DataLoader(vset,
                                   batch_sampler=self.create_sampler(vset, batch_size,
                                                                     allow_uneven=self.ALLOW_VARYING_BATCH_SIZE),
                                   collate_fn=framework.loader.collate.VarLengthCollate(
                                        batch_dim=self.batch_dim),
                                   num_workers=self.VALID_NUM_WORKERS)

    def create_train_loader(self, loader: torch.utils.data.Dataset, _=None) -> torch.utils.data.DataLoader:
        sampler = self.create_sampler(loader, self.helper.args.batch_size)
        self.helper.saver.register("sampler", sampler, replace=True)

        return torch.utils.data.DataLoader(loader, batch_sampler=sampler, num_workers=self.TRAIN_NUM_WORKERS,
                                           pin_memory=True, collate_fn=framework.loader.collate.VarLengthCollate(
                                           batch_dim=self.batch_dim))
