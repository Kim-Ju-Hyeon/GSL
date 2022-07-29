import numpy as np
import random
import torch
from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch
from utils.scalers import Scaler
import os
import pandas as pd

from six.moves import urllib
import zipfile
import gdown

from utils.dataset_utils import time_features_from_frequency_str


class Temporal_Graph_Signal(object):
    def __init__(self, dataset_name, scaler_type='std'):
        super(Temporal_Graph_Signal, self).__init__()
        self.dataset_name = dataset_name
        self._set_dataset_parameters(dataset_name)
        self.scaler = Scaler(scaler_type)

    def preprocess_dataset(self):
            if (self.dataset_name == 'METR-LA') or (self.dataset_name == 'PEMS-BAY'):
                X, A = self._get_torch_geometric_dataset()
                X = X.astype(np.float32)
                X = X.transpose((1, 2, 0))
                X = self.scaler.scale(X)

                total_sequence_length = X.shape[-1]
                train_index = int(total_sequence_length * 0.7) + 1
                valid_index = int(total_sequence_length * 0.2) + 1 + train_index

                self.train_X = X[:, :, :train_index]
                self.valid_X = X[:, :, train_index:valid_index]
                self.test_X = X[:, :, valid_index:]

                self.A = torch.from_numpy(A)
            else:
                self._read_web_data()

                y_df = pd.read_csv(os.path.join(self.path, f'{self.dataset_name}.csv'))

                y_df['date'] = pd.to_datetime(y_df['date'])
                y_df.rename(columns={'date': 'ds'}, inplace=True)
                u_ids = y_df.columns.to_list()
                u_ids.remove('ds')
                time_cls = time_features_from_frequency_str(self.freq)
                for cls_ in time_cls:
                    cls_name = cls_.__class__.__name__
                    y_df[cls_name] = cls_(y_df['ds'].dt)

                time_stamp = y_df.drop(u_ids + ['ds'], axis=1).to_numpy().T
                temp = np.array([time_stamp for _ in range(self.nodes_num)])

                df = y_df[u_ids].to_numpy().T
                df = np.expand_dims(df, axis=1)
                df = self.scaler.scale(df)

                # Total X dimension = [Number of Nodes, Number of Features, Sequence Length]
                X = np.concatenate([df, temp], axis=1)

                total_sequence_length = X.shape[-1]
                train_index = int(total_sequence_length * 0.6) + 1
                valid_index = int(total_sequence_length * 0.2) + 1 + train_index

                self.train_X = X[:, :, :train_index]
                self.valid_X = X[:, :, train_index:valid_index]
                self.test_X = X[:, :, valid_index:]

    def _set_dataset_parameters(self, dataset_name):
        ett_dataset_list = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2']

        if dataset_name == 'METR-LA':
            self.nodes_num = 207
            self.node_features = 2
            self.url = "https://graphmining.ai/temporal_datasets/METR-LA.zip"
        elif dataset_name == 'PEMS-BAY':
            self.nodes_num = 325
            self.node_features = 2
            self.url = "https://graphmining.ai/temporal_datasets/PEMS-BAY.zip"
        elif dataset_name == 'COVID19':
            self.nodes_num = 25
            self.node_features = 5
            self.freq = '1D'
        elif self.dataset_name == 'ECL':
            self.nodes_num = 321
            self.node_features = 5
            self.url = 'https://drive.google.com/uc?id=1rUPdR7R2iWFW-LMoDdHoO2g4KgnkpFzP'
            self.freq = '1H'
        elif dataset_name in ett_dataset_list:
            self.nodes_num = 7
            self.node_features = 5
            self.url = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/'
            if (dataset_name == ett_dataset_list[0]) or (dataset_name == ett_dataset_list[1]):
                self.freq = '15min'
            else:
                self.freq = '1H'
        elif dataset_name == 'WTH':
            self.nodes_num = 21
            self.node_features = 5
            self.url = 'https://drive.google.com/uc?id=1UBRz-aM_57i_KCC-iaSWoKDPTGGv6EaG'
            self.freq = '10min'
        elif dataset_name == 'Traffic':
            self.nodes_num = 862
            self.node_features = 5
            self.freq = '1H'
        elif dataset_name == 'Exchange':
            self.nodes_num = 8
            self.node_features = 1
            self.freq = '1D'
        else:
            raise ValueError("Non-supported dataset!")

        if dataset_name in ett_dataset_list:
            self.path = f'./data/ETT/{dataset_name}'
        else:
            self.path = f'./data/{dataset_name}'

    def _download_url(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        gdown.download(self.url, os.path.join(self.path, f'{self.dataset_name}.csv'))

    def _read_web_data(self):
        if not os.path.isfile(os.path.join(self.path, f'{self.dataset_name}.csv')):
            self._download_url()

    def _download_pgt_dataset_url(self, url, path):
        with urllib.request.urlopen(url) as dl_file:
            with open(path, "wb") as out_file:
                out_file.write(dl_file.read())

    def _get_torch_geometric_dataset(self):
        if self.dataset_name == 'METR-LA':
            url = "https://graphmining.ai/temporal_datasets/METR-LA.zip"

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

            A = np.load(os.path.join(self.path, "adj_mat.npy"))
            X = np.load(os.path.join(self.path, "node_values.npy"))

        elif self.dataset_name == 'PEMS-BAY':
            url = "https://graphmining.ai/temporal_datasets/PEMS-BAY.zip"

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

            A = np.load(os.path.join(self.path, "pems_adj_mat.npy"))
            X = np.load(os.path.join(self.path, "pems_node_values.npy"))

        return X, A

    def _generate_dataset(self, dataset, num_timesteps_in: int = 12, num_timesteps_out: int = 12,
                          batch_size: int = 32,
                          inference: bool = False):
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(dataset.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        if inference:
            pass
        else:
            random.shuffle(indices)

        features, target = [], []
        for i, j in indices:
            features.append((dataset[:, :, i: i + num_timesteps_in]))
            target.append((dataset[:, 0, i + num_timesteps_in: j]))

        features = torch.FloatTensor(np.array(features))
        targets = torch.FloatTensor(np.array(target))

        dataset = self._get_batch_graph_temporal_signal_batch(num_timesteps_in, num_timesteps_out,
                                                              features, targets, batch_size)

        return dataset

    def _get_batch_graph_temporal_signal_batch(self, num_timesteps_in, num_timesteps_out, features, targets,
                                               batch_size):
        data_batch = features.shape[0] // batch_size
        node_num, feature_dim = features.shape[1], features.shape[2]

        _batch = []
        for batch_iter in range(batch_size):
            _batch += [batch_iter] * node_num
        _batch = np.array(_batch)

        feature_Tensor = features[:data_batch * batch_size].reshape(data_batch, node_num * batch_size,
                                                                    feature_dim, num_timesteps_in).numpy()

        target_Tensor = targets[:data_batch * batch_size].reshape(data_batch, node_num * batch_size,
                                                                  num_timesteps_out).numpy()

        dataset = StaticGraphTemporalSignalBatch(edge_index=None, edge_weight=None,
                                                 features=feature_Tensor, targets=target_Tensor, batches=_batch)

        return dataset

    def get_dataset(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12, batch_size: int = 32):
        train_dataset = self._generate_dataset(self.train_X, num_timesteps_in, num_timesteps_out, batch_size)
        valid_dataset = self._generate_dataset(self.valid_X, num_timesteps_in, num_timesteps_out, batch_size)
        test_dataset = self._generate_dataset(self.test_X, num_timesteps_in, num_timesteps_out, 1,
                                              inference=True)

        return train_dataset, valid_dataset, test_dataset

    def get_scaler(self):
        return self.scaler
