import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.N_BEATS.Block import SeasonalityBlock, TrendBlock, GenericBlock, NHITSBlock


class N_model(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def create_stack(self):
        pass

    def forward(self):
        pass
