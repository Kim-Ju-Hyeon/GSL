import pickle
import os
from os import path
import torch
import numpy

class MakeDataset():
    def __init__(self, config):
        super(MakeDataset, self).__init__()

        self.data_dir = config.dataset.root
        self.total_time_length = config.dataset.total_time_length
        self.window_size = config.dataset.window_size
        self.slide = config.dataset.slide
        self.train_valid_test = config.dataset.train_valid_test

        self.save_dir = path.join(config.dataset.save, f'{config.dataset.name}_{self.window_size}_{self.slide}')

        if path.exists(self.save_dir):
            self.dataset = pickle.load(open(self.save_dir, 'rb'))
        else:
            self.spk_bin = pickle.load(open('./data/spk_bin_n100.pickle', 'rb'))
            self.lam_bin = pickle.load(open('./data/lam_bin_n100.pickle', 'rb'))

    def __save_dataset__(self):
        pass

    def split(self, types: str):
        pass

    def make(self):
        pass

