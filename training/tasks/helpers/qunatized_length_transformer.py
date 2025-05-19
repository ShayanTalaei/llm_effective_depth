import torch.nn.functional as F


class QuantizedLengthTransformer:
    PAD_QUANTUM = 32
    PAD_KEYS = {"data", "mask", "eval_mask"}
    IGNORE_INDEX_KEYS = {"data"}

    def run_model(self, data, *args, **kwargs):
        olen = data["data"].shape[0] - 1
        # Quantize length so that we avoid continous recompilation.
        tlen = ((olen + self.PAD_QUANTUM - 1) // self.PAD_QUANTUM) * self.PAD_QUANTUM

        ignore_index = self.ignore_index if hasattr(self, "ignore_index") and self.ignore_index is not None else 0

        if tlen != olen:
            data = {k: v for k, v in data.items()}
            for f in self.PAD_KEYS:
                if f in data:
                    data[f] = F.pad(data[f], [0] * ((data[f].ndim-1)*2) + [0, tlen-olen],
                                    value=ignore_index if f in self.IGNORE_INDEX_KEYS else 0, mode="constant")

        res, d  = super().run_model(data, *args, **kwargs)
        res.outputs = res.outputs[:olen]
        return res, d
