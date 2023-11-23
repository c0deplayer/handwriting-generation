import torch

from .base_score import BaseDistance, ProcessedDataset


class EuclideanDistance(BaseDistance):
    def __init__(self):
        super().__init__()

    def __call__(self, data1, data2, **kwargs):
        assert isinstance(data1, ProcessedDataset)
        assert isinstance(data2, ProcessedDataset)
        tmp_1 = data1.features.mean(dim=0).unsqueeze(0)
        tmp_2 = data2.features.mean(dim=0).unsqueeze(0)
        return torch.cdist(tmp_1, tmp_2).item()
