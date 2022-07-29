from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.IC_PN_BEATS.IC_PN_BEATS import IC_PN_BEATS


class IC_PN_BEATS_model(nn.Module):
    def __init__(self, config):
        super(IC_PN_BEATS_model, self).__init__()

        self.config = config
        self.model = IC_PN_BEATS(config)

    def forward(self, inputs, interpretability=False):
        outputs = defaultdict(list)

        backcast, forecast = self.model(inputs, interpretability)

        if interpretability:
            outputs['per_trend_backcast'] = self.model.per_trend_backcast
            outputs['per_trend_forecast'] = self.model.per_trend_forecast

            outputs['per_seasonality_backcast'] = self.model.per_seasonality_backcast
            outputs['per_seasonality_forecast'] = self.model.per_seasonality_forecast

            outputs['singual_backcast'] = self.model.singual_backcast
            outputs['singual_forecast'] = self.model.singual_forecast

            outputs['attention_matrix'] = self.model.attn_matrix

        return backcast, forecast, outputs
