
import torch
from typing import Optional, Dict, Any, List, Set
import numpy as np
import torch.nn.functional as F
from ...utils.distributed_ops import reduce_any


class GroupLmTestState:
    SUPPORTS_DISTRIBUTED = True

    def __init__(self, groups: List[str], group_key: str, batch_dim: int = 1, ignore_index: Optional[int] = None,
                 target_key: str = "data", mask_key: Optional[str] = None, custom_groups: Dict[str, Set[str]] = {}):
        self.loss_sum = {"all": 0.0}
        self.n_ok = {"all": 0.0}
        self.n_total = {"all": 0.0}
        self.batch_dim = batch_dim
        self.time_dim = 1 - self.batch_dim
        self.ignore_index = ignore_index or -100
        self.target_key = target_key
        self.mask_key = mask_key

        self.custom_group_names = list(custom_groups.keys())
        self.custom_group_map = {}
        for i, gname in enumerate(self.custom_group_names):
            gset = custom_groups[gname]
            for g in gset:
                self.custom_group_map[groups.index(g)] = i + len(groups)

        self.group_key = group_key
        self.custom_group_offset = len(groups)
        self.groups = list(groups) + self.custom_group_names

        for g in self.groups:
            self.loss_sum[g] = 0.0
            self.n_ok[g] = 0.0
            self.n_total[g] = 0.0


    def step(self, net_out: torch.Tensor, data: Dict[str, torch.Tensor]):
        with torch.no_grad():
            target = data[self.target_key].long()
            if self.mask_key is not None:
                target = target.masked_fill(~data[self.mask_key].bool(), self.ignore_index)

            target = target.narrow(self.time_dim, 1, target.shape[self.time_dim] - 1).contiguous()
            loss = F.cross_entropy(
                net_out.flatten(0, -2), target.flatten(), reduction='none',
                ignore_index=self.ignore_index if self.ignore_index is not None else -100).view_as(target).sum(self.time_dim)

            out = net_out.argmax(-1)

            assert out.shape == target.shape

            ok_per_seq = ((out == target) | (target == self.ignore_index)).all(dim=self.time_dim).long()

            self.n_total["all"] += ok_per_seq.nelement()
            self.n_ok["all"] += ok_per_seq.sum().cpu().item()
            self.loss_sum["all"] += loss.sum().cpu().item()

            for i, gid in enumerate(data[self.group_key].cpu().numpy().tolist()):
                glist = [gid]
                if gid in self.custom_group_map:
                    glist.append(self.custom_group_map[gid])

                for gid in glist:
                    gname = self.groups[gid]
                    self.n_total[gname] += 1
                    self.n_ok[gname] += ok_per_seq[i].cpu().item()
                    self.loss_sum[gname] += loss[i].cpu().item()

    @property
    def accuracy(self) -> float:
        return reduce_any(self.n_ok["all"]) / reduce_any(self.n_total["all"])

    def plot(self) -> Dict[str, Any]:
        res = {}

        print("Test results:")
        for k in self.loss_sum.keys():
            tot = reduce_any(self.n_total[k])
            nok = reduce_any(self.n_ok[k])
            allsum = reduce_any(self.loss_sum[k])
            if tot > 0:
                loss = allsum / tot
                ppl = np.exp(loss)
                acc = nok / tot
                res.update({
                    f"{k}/loss": loss,
                    f"{k}/accuracy": acc,
                    f"{k}/perplexity": ppl
                })

                print(f"  {k}: {acc:.4f}")

        return res

