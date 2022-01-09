import pickle
import os
from os import path
import torch
import numpy

from torch_geometric.data import Data


class MakeDataset:
    def __init__(self, config):
        super(MakeDataset, self).__init__()

        self.data_dir = config.dataset.root
        self.window_size = config.dataset.window_size
        self.slide = config.dataset.slide
        self.pred_step = config.dataset.pred_step
        self.idx_ratio = config.dataset.idx_ratio
        self.train_valid_test = config.dataset.train_valid_test
        self.encoder_step = config.encoder_step
        self.decoder_step = config.decoder_step

        self.load = None

        self.total_input_size = (self.encoder_step + self.decoder_step - 1) * self.slide + self.window_size
        self.batch_idx = int(self.total_input_size * self.idx_ratio)

        self.save_dir = path.join(config.dataset.save, f'{config.dataset.name}_'
                                                       f'{self.window_size}{self.slide}{self.pred_step}.pickle')

        if path.exists(self.save_dir):
            self.dataset = pickle.load(open(self.save_dir, 'rb'))
            self.load = True
        else:
            self.spk_bin = pickle.load(open('./data/spk_bin_n100.pickle', 'rb'))
            self.lam_bin = pickle.load(open('./data/lam_bin_n100.pickle', 'rb'))
            self.load = False

    def _save_dataset(self):
        pass

    def _valid_sampling(self, i):
        if i == 0:
            start = 0
            total_length = self.train_valid_test[i]
        else:
            start = self.train_valid_test[i - 1]
            total_length = self.train_valid_test[i] - self.train_valid_test[i - 1]

        valid_sampling_locations = []
        valid_sampling_locations += [
            i
            for i in range(start, start + total_length + 1 - self.pred_step - self.total_input_size)
            if (i % self.batch_idx) == 0
        ]

        return valid_sampling_locations

    def _split(self, i):
        if i == 0:
            data = self.spk_bin[:self.train_valid_test[i]]
            lam = self.lam_bin[:self.train_valid_test[i]]
        else:
            data = self.spk_bin[self.train_valid_test[i - 1]:self.train_valid_test[i]]
            lam = self.lam_bin[self.train_valid_test[i - 1]:self.train_valid_test[i]]

        return data, lam

    def make(self):
        data_dict = {'train': None,
                     'valid': None,
                     'test': None}

        if not self.load:
            for i, types in enumerate(list(data_dict.keys())):
                data, lam = self._split(i)
                valid_sampling_locations = self._valid_sampling(i)

                data_list = []
                for start_idx in valid_sampling_locations:
                    spike_input = data[:, start_idx:start_idx + self.total_input_size]
                    lam_output = lam[:,
                                    start_idx + self.encoder_step * self.slide + self.window_size:start_idx +
                                    self.total_input_size + self.pred_step]

                    data_item = Data(x=torch.FloatTensor(spike_input), edge_index=None, y=torch.FloatTensor(lam_output))
                    data_list.append(data_item)

                if types == 'train':
                    data_dict['train'] = data_list
                elif types == 'valid':
                    data_dict['valid'] = data_list
                elif types == 'test':
                    data_dict['test'] = data_list

                pickle.dump(data_dict, open(self.save_dir, 'wb'))

            return data_dict

        else:
            return self.dataset
