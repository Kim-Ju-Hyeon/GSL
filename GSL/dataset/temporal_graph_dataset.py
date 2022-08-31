import numpy as np
import random
import torch

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
        self._set_dataset_parameters(dataset_name)
        self.scaler = Scaler(scaler_type)

    def preprocess_dataset(self):
        if (self.dataset_name == 'METR-LA') or (self.dataset_name == 'PEMS-BAY'):
            X = self._get_torch_geometric_dataset()
            X = X.astype(np.float32)
            X = X.transpose((1, 2, 0))

            self.time_stamp = X[:, 1, :]
            X = X[:, 0, :]
            X = self.scaler.scale(X)
            X = np.expand_dims(X, axis=1)

            total_sequence_length = X.shape[-1]
            train_index = int(total_sequence_length * 0.7) + 1
            valid_index = int(total_sequence_length * 0.2) + 1 + train_index

        else:
            self._read_web_data()

            y_df = pd.read_csv(os.path.join(self.path, f'{self.dataset_name}.csv'), index_col=self.index_col)
            if 'Date Time' in y_df:
                y_df.rename(columns={'Date Time': 'date'}, inplace=True)

            if (not self.univariate) and (not self.dataset_name == 'Exchange'):
                self.time_stamp = self._get_timestamp(y_df)

            if self.dataset_name == 'Exchange':
                X = y_df.to_numpy().T
                X = X.astype(np.float32)
                X = np.expand_dims(X, axis=1)
                X = self.scaler.scale(X)
            else:
                _df = y_df.drop(['date'], axis=1).to_numpy().T
                X = _df.astype(np.float32)
                X = np.expand_dims(X, axis=1)
                X = self.scaler.scale(X)

            total_sequence_length = X.shape[-1]
            train_index = int(total_sequence_length * 0.6) + 1
            valid_index = int(total_sequence_length * 0.2) + 1 + train_index

        self.train_X = X[:, :, :train_index]
        self.valid_X = X[:, :, train_index:valid_index]
        self.test_X = X[:, :, valid_index:]

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

    def _set_dataset_parameters(self, dataset_name):
        ett_dataset_list = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2']

        if dataset_name == 'METR-LA':
            self.nodes_num = 207
            self.node_features = 2
            self.freq = '15min'
            self.url = "https://graphmining.ai/temporal_datasets/METR-LA.zip"

        elif dataset_name == 'PEMS-BAY':
            self.nodes_num = 325
            self.node_features = 2
            self.freq = '15min'
            self.url = "https://graphmining.ai/temporal_datasets/PEMS-BAY.zip"

        elif dataset_name == 'COVID19':
            self.index_col = 0
            self.nodes_num = 25
            self.node_features = 4
            self.freq = '1D'
            self.url = 'https://drive.google.com/file/d/1rPwzpCH8fzNiteXMyO71EbKixievDu3j/view?usp=sharing'

        elif self.dataset_name == 'ECL':
            self.index_col = False
            self.nodes_num = 321
            self.node_features = 5
            self.url = 'https://drive.google.com/file/d/1nzq4Q3bdVHBqpiz4V7hR1Z3w-kssADmt/view?usp=sharing'
            self.freq = '1H'

        elif dataset_name in ett_dataset_list:
            self.index_col = False
            self.nodes_num = 7

            if dataset_name == ett_dataset_list[0]:
                self.url = 'https://drive.google.com/file/d/1KSAK82HFR2rE8NxkZu5OoG6ax_Od2FNo/view?usp=sharing'
            elif dataset_name == ett_dataset_list[1]:
                self.url = 'https://drive.google.com/file/d/1B_roxkwOS0FBteC-iYstFtM0FhO0SoJ-/view?usp=sharing'
            elif dataset_name == ett_dataset_list[2]:
                self.url = 'https://drive.google.com/file/d/16JUqAOvfaI_AQ16Pv8S2iac7QzlD_D4F/view?usp=sharing'
            elif dataset_name == ett_dataset_list[3]:
                self.url = 'https://drive.google.com/file/d/1U23E3-o7uUipUbzQnbTB_-X0EH8wizNf/view?usp=sharing'

            if (dataset_name == ett_dataset_list[0]) or (dataset_name == ett_dataset_list[1]):
                self.freq = '15min'
                self.node_features = 6
            else:
                self.freq = '1H'
                self.node_features = 5

        elif dataset_name == 'WTH':
            self.index_col = False
            self.nodes_num = 21
            self.node_features = 6
            self.url = 'https://drive.google.com/file/d/1LMv2gdpVW7tG6BJgxTa5z001b743RbdH/view?usp=sharing'
            self.freq = '10min'

        elif dataset_name == 'Traffic':
            self.index_col = 0
            self.nodes_num = 862
            self.node_features = 5
            self.freq = '1H'
            self.url = 'https://drive.google.com/file/d/1v6nDK-X_77OaIYMhce4675I692GsEvDI/view?usp=sharing'

        elif dataset_name == 'Exchange':
            self.index_col = 0
            self.nodes_num = 8
            self.node_features = 1
            self.url = 'https://drive.google.com/file/d/1TIRVGj4KkTtvTzeBVG1-xU8gHCaMagAj/view?usp=sharing'
            self.freq = '1D'
        else:
            raise ValueError("Non-supported dataset!")

        if dataset_name in ett_dataset_list:
            if not os.path.exists(f'./data/ETT'):
                os.mkdir(f'./data/ETT')
            self.path = f'./data/ETT/{dataset_name}'
        else:
            self.path = f'./data/{dataset_name}'

    def _download_url(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        gdown.download(self.url, os.path.join(self.path, f'{self.dataset_name}.csv'), fuzzy=True)

    def _read_web_data(self):
        if not os.path.isfile(os.path.join(self.path, f'{self.dataset_name}.csv')):
            self._download_url()

    def _download_pgt_dataset_url(self, url, path):
        with urllib.request.urlopen(url) as dl_file:
            with open(path, "wb") as out_file:
                out_file.write(dl_file.read())

    def _get_torch_geometric_dataset(self):
        if self.dataset_name == 'METR-LA':
            # Check if zip file is in data folder from working directory, otherwise download
            if not os.path.isfile(
                    os.path.join(self.path, "METR-LA.zip")
            ):  # pragma: no cover
                if not os.path.exists(self.path):
                    os.makedirs(self.path)
                self._download_pgt_dataset_url(self.url, os.path.join(self.path, "METR-LA.zip"))

            if not os.path.isfile(
                    os.path.join(self.path, "adj_mat.npy")
            ) or not os.path.isfile(
                os.path.join(self.path, "node_values.npy")
            ):  # pragma: no cover
                with zipfile.ZipFile(
                        os.path.join(self.path, "METR-LA.zip"), "r"
                ) as zip_fh:
                    zip_fh.extractall(self.path)

            X = np.load(os.path.join(self.path, "node_values.npy"))

        elif self.dataset_name == 'PEMS-BAY':
            # Check if zip file is in data folder from working directory, otherwise download
            if not os.path.isfile(
                    os.path.join(self.path, "PEMS-BAY.zip")
            ):  # pragma: no cover
                if not os.path.exists(self.path):
                    os.makedirs(self.path)
                self._download_pgt_dataset_url(self.url, os.path.join(self.path, "PEMS-BAY.zip"))

            if not os.path.isfile(
                    os.path.join(self.path, "pems_adj_mat.npy")
            ) or not os.path.isfile(
                os.path.join(self.path, "pems_node_values.npy")
            ):  # pragma: no cover
                with zipfile.ZipFile(
                        os.path.join(self.path, "PEMS-BAY.zip"), "r"
                ) as zip_fh:
                    zip_fh.extractall(self.path)

            X = np.load(os.path.join(self.path, "pems_node_values.npy"))

        return X

    def _generate_dataset(self, dataset, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(dataset.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        if not self.univariate:
            time_feature = []
        features, target = [], []
        if not self.univariate:
            time_feature = []
        for i, j in indices:
            features.append((dataset[:, :, i: i + num_timesteps_in]))
            target.append((dataset[:, 0, i + num_timesteps_in: j]))

            if not self.univariate:
                time_feature.append(self.time_stamp[:, i:i + num_timesteps_in])

        features = torch.FloatTensor(np.array(features))
        targets = torch.FloatTensor(np.array(target))
        if not self.univariate:
            time_feature = torch.FloatTensor(np.array(time_feature))

        _data = []
        for batch in range(len(indices)):
            if self.univariate:
                _data.append(Data(x=features[batch], y=targets[batch], time_stamp=None))
            elif not self.univariate:
                _data.append(Data(x=features[batch], y=targets[batch], time_stamp=time_feature[batch]))

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
            test = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)

            return train, valid, test

        else:
            return train_dataset, valid_dataset, test_dataset

    def get_scaler(self):
        return self.scaler
