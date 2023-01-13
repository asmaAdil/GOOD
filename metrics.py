import torch
from torch import nn


class HitsRateMetric(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    @torch.no_grad()
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pos_mask = target == 1
        pos_preds = preds[pos_mask]
        neg_preds = preds[~pos_mask]
        kth_score_in_negative_edges = torch.topk(input=neg_preds, k=self.k)[0][-1]
        hits = (pos_preds > kth_score_in_negative_edges).sum().cpu() / pos_preds.shape[0]
        return hits
