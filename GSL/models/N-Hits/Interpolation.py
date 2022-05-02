import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, interpolation_mode: str):
        super().__init__()
        assert (interpolation_mode in ['linear', 'nearest']) or ('cubic' in interpolation_mode)
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode

    def forward(self, theta: torch.Tensor, insample_x_t: torch.Tensor, outsample_x_t: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:

        backcast = theta[:, :self.backcast_size]
        knots = theta[:, self.backcast_size:]

        if self.interpolation_mode == 'nearest':
            knots = knots[:, None, :]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode)
            forecast = forecast[:, 0, :]

        elif self.interpolation_mode == 'linear':
            knots = knots[:, None, :]
            forecast = F.interpolate(knots, size=self.forecast_size,
                                     mode=self.interpolation_mode)  # , align_corners=True)
            forecast = forecast[:, 0, :]

        elif 'cubic' in self.interpolation_mode:
            batch_size = int(self.interpolation_mode.split('-')[-1])
            knots = knots[:, None, None, :]
            forecast = torch.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots) / batch_size))

            for i in range(n_batches):
                forecast_i = F.interpolate(knots[i * batch_size:(i + 1) * batch_size], size=self.forecast_size,
                                           mode='bicubic')  # , align_corners=True)
                forecast[i * batch_size:(i + 1) * batch_size] += forecast_i[:, 0, 0, :]

        else:
            raise ValueError("Invalid Interpolation Mode!")

        return backcast, forecast
