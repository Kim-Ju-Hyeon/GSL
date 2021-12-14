import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from models.GTS.GTS_model import GTS_Model
from utils.utils import build_fully_connected_edge_idx




class GTS_Runner(object):
    def __init__(self, config):
        self.config = config
        self.seed = config.seed
        self.device = config.device

        self.dataset_conf = config.dataset

        self.fully_connected_edge_index = build_fully_connected_edge_idx(self.config.num_nodes)

        if self.dataset_conf.name == 'spike_lambda':
            spike = pickle.load(open('./data/LNP_spk_all.pickle', 'rb'))
            lam = pickle.load(open('./data/LNP_lam_all.pickle', 'rb'))

            self.entire_inputs = torch.FloatTensor(spike)

        self.model = GTS_Model(self.config)



    def train(self):
        pass

    def validation(self):
        pass

    def test(self):
        pass

        
