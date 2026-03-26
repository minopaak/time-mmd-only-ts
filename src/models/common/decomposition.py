import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """
    Moving average block for trend extraction.
    Input:  [B, T, C]
    Output: [B, T, C]
    """
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2

        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)      # [B, T+2p, C]

        x = x.permute(0, 2, 1)                     # [B, C, T]
        x = self.avg(x)                            # [B, C, T]
        x = x.permute(0, 2, 1)                     # [B, T, C]
        return x


class SeriesDecomposition(nn.Module):
    """
    x = seasonal + trend
    """
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend