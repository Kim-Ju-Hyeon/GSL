import torch
import torch.nn as nn

from models.GTS.graph_learning import GlobalGraphLearning
from models.GTS.DCRNN import DCRNN

class GTSModel(nn.Module):
    def __init__(self, config):
        super().__init()
        self.coonfig = config

        self.graph_learning = GlobalGraphLearning(config)

        self.encoder = DCRNN(config)
        self.decoder = DCRNN(config)



