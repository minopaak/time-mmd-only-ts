import torch
import torch.nn as nn


class ChannelToSingleTargetHead(nn.Module):
    """
    Maps multivariate outputs [B, pred_len, C] -> [B, pred_len]
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, pred_len, C]
        return self.proj(x).squeeze(-1)  # [B, pred_len]