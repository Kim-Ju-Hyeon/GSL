import numpy as np
import random
import torch
from glob import glob

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from utils.scalers import Scaler
import os
import pandas as pd

from six.moves import urllib
import zipfile
import gdown
import pickle

from utils.dataset_utils import time_features_from_frequency_str


class Temporal_Graph_Signal(object):
    def __init__(self, dataset_name, scaler_type='std', univariate=False):
        super(Temporal_Graph_Signal, self).__init__()
        self.dataset_name = dataset_name
        self.univariate = univariate
        self.num_workers = 4 * torch.cuda.device_count()
        self._set_dataset_parameters()
        self.scaler = Scaler(scaler_type)

    def preprocess_dataset(self):
        y_df = pd.read_csv(self.path, index_col=0).to_numpy().T
        y_df = self.scaler.scale(y_df)

        total_sequence_length = len(y_df)
        train_index = int(total_sequence_length * 0.7)
        valid_index = int(total_sequence_length * 0.1) + train_index

        self.train_X = y_df[:, :train_index]
        self.valid_X = y_df[:, train_index:valid_index]
        self.test_X = y_df[:, valid_index:]

        if not os.path.isfile(os.path.join(self.path, f'inference.pickle')):
            pickle.dump(self.test_X, open(os.path.join(self.path, f'inference.pickle'), 'wb'))

        if not os.path.isfile(os.path.join(self.path, f'scaler.pickle')):
            pickle.dump(self.scaler, open(os.path.join(self.path, f'scaler.pickle'), 'wb'))

    def _get_timestamp(self, y_df):
        dataframe = pd.DataFrame()
        time_cls = time_features_from_frequency_str(self.freq)
        for cls_ in time_cls:
            cls_name = cls_.__class__.__name__
            dataframe[cls_name] = cls_(y_df['date'].dt)
        time_stamp = dataframe.to_numpy().T

        return time_stamp

    def _set_dataset_parameters(self):
        if self.univariate:
            self.nodes_num = 6
            self.path = './data/port_visible_feature.csv'

        else:
            self.nodes_num = 90
            self.path = './data/port_total_feature.csv'

        self.node_features = 1
        self.freq = '10min'
        self.url = None

    def _generate_dataset(self, dataset, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(dataset.shape[1] - (num_timesteps_in + num_timesteps_out) + 1)
            if i % 3 == 0
        ]

        features, target = [], []

        for i, j in indices:
            features.append((dataset[:, i: i + num_timesteps_in]))
            target.append((dataset[:, i + num_timesteps_in: j]))

        features = torch.FloatTensor(np.array(features))
        targets = torch.FloatTensor(np.array(target))


        _data = []
        for batch in range(len(indices)):
            _data.append(Data(x=features[batch], y=targets[batch], time_stamp=None))

        return _data

    def get_dataset(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12, batch_size: int = 32,
                    return_loader=True):

        train_dataset = self._generate_dataset(self.train_X, num_timesteps_in, num_timesteps_out)
        valid_dataset = self._generate_dataset(self.valid_X, num_timesteps_in, num_timesteps_out)
        test_dataset = self._generate_dataset(self.test_X, num_timesteps_in, num_timesteps_out)

        if return_loader:
            train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                               num_workers=self.num_workers, pin_memory=True)
            valid = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                               num_workers=self.num_workers, pin_memory=True)
            test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

            return train, valid, test

        else:
            return train_dataset, valid_dataset, test_dataset

    def get_scaler(self):
        return self.scaler
