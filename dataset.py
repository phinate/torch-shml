from __future__ import annotations

__all__ = ["EventDataset"]

import json

import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset


class EventDataset(Dataset):
    def __init__(
        self,
        data: ak.Array | str,
        fields: list[str] | None = None,
        config_path: str | None = None,
        use_truth_masses: bool = True,
    ) -> None:
        if isinstance(data, str):
            akarr = ak.from_parquet(data)
        else:
            akarr = data

        fields = fields or akarr.fields

        # cast array to float32 so numpy doesn't complain about mixed dtypes
        self.awkward_array = ak.values_astype(akarr[fields], np.float64)
        self.length = len(self.awkward_array)

        config_path = config_path or "config.json"
        with open(config_path) as f:
            self.config = json.load(f)

        self.use_truth_masses = use_truth_masses

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Assumes single integer used for indexing."""
        awk = self.awkward_array[idx]
        fields = self.awkward_array.fields
        cat = awk["category"]
        weight = awk["weight"]

        # remove entries we've already processed from fields
        fields.remove("category")
        fields.remove("weight")
        if not self.use_truth_masses:
            fields.remove("truth_masses")

        X = np.asarray(awk[fields])
        y = ak.to_numpy(cat)
        # supply weight separately for use with loss calc
        w = ak.to_numpy(weight)

        return torch.Tensor(X), y, w

    def __len__(self) -> int:
        return self.length
