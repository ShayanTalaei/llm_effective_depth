import torch.utils.data
import numpy as np


class PermuteDataset:
    def __init__(self, dataset: torch.utils.data.Dataset) -> None:
        super().__init__()

        self._dataset = dataset
        self._order = np.random.default_rng(123).permutation(len(dataset)).tolist()

    def __len__(self) -> int:
        return len(self._order)

    def __getitem__(self, index):
        return self._dataset[self._order[index]]

    def __getattr__(self, item):
        return getattr(self._dataset, item)
