import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from models import *


class GTS_Runner(object):
    def __init__(self, config):
        self.config = config
        self.seed = config.seed

        
