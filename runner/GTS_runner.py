import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from models.GTS.GTS_model import GTS_Model



class GTS_Runner(object):
    def __init__(self, config):
        self.config = config
        self.seed = config.seed
        self.device = config.device

        self.dataset_conf = config.dataset

        if self.dataset_conf.name == 'spike_lambda':
            spike = pickle.load(open('./data/LNP_spk_all.pickle', 'rb'))
            lam = pickle.load(open('./data/LNP_lam_all.pickle', 'rb'))

            self.entire_inputs = spike

        self.model = GTS_Model(self.config)



    def train(self):
        pass

    def validation(self):
        pass

    def test(self):
        pass

        
