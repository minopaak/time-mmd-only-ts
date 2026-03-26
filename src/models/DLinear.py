import torch
import torch.nn as nn

from src.models.common.base import ForecastModelBase  # 경로 수정
from src.models.common.decomposition import SeriesDecomposition  # 경로 수정
from src.models.common.linear_heads import ChannelToSingleTargetHead  # 경로 수정


class Model(ForecastModelBase):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.individual = getattr(configs, "individual", True)
        self.moving_avg = getattr(configs, "moving_avg", 25)

        self.decomposition = SeriesDecomposition(kernel_size=self.moving_avg)

        if self.individual:
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)
            ])
            self.linear_trend = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)
            ])

            for i in range(self.enc_in):
                self.linear_seasonal[i].weight = nn.Parameter(
                    (1.0 / self.seq_len) * torch.ones(self.pred_len, self.seq_len)
                )
                self.linear_trend[i].weight = nn.Parameter(
                    (1.0 / self.seq_len) * torch.ones(self.pred_len, self.seq_len)
                )
                nn.init.zeros_(self.linear_seasonal[i].bias)
                nn.init.zeros_(self.linear_trend[i].bias)
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

            self.linear_seasonal.weight = nn.Parameter(
                (1.0 / self.seq_len) * torch.ones(self.pred_len, self.seq_len)
            )
            self.linear_trend.weight = nn.Parameter(
                (1.0 / self.seq_len) * torch.ones(self.pred_len, self.seq_len)
            )
            nn.init.zeros_(self.linear_seasonal.bias)
            nn.init.zeros_(self.linear_trend.bias)

        self.target_head = ChannelToSingleTargetHead(input_dim=self.enc_in)

    def encoder(self, x):
        # x: [B, T, C]
        seasonal_init, trend_init = self.decomposition(x)

        seasonal_init = seasonal_init.permute(0, 2, 1)  # [B, C, T]
        trend_init = trend_init.permute(0, 2, 1)        # [B, C, T]

        if self.individual:
            B, C, _ = seasonal_init.shape
            seasonal_output = torch.zeros(
                B, C, self.pred_len,
                dtype=seasonal_init.dtype, device=seasonal_init.device
            )
            trend_output = torch.zeros(
                B, C, self.pred_len,
                dtype=trend_init.dtype, device=trend_init.device
            )

            for i in range(C):
                seasonal_output[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)  # [B, C, pred_len]
            trend_output = self.linear_trend(trend_init)           # [B, C, pred_len]

        out = seasonal_output + trend_output
        out = out.permute(0, 2, 1)  # [B, pred_len, C]
        return out

    def forward(self, x):
        out = self.encoder(x)       # [B, pred_len, C]
        out = self.target_head(out) # [B, pred_len]
        return out