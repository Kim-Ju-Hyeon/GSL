import os
import gdown
import numpy as np
import pandas as pd
import random
import torch
from utils.dataset_utils import time_features_from_frequency_str
from utils.utils import build_fully_connected_edge_idx, build_batch_edge_index
from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch
from utils.scalers import Scaler


class ECLDatasetLoader(object):
    def __init__(self, raw_data_dir, scaler_type='std'):
        super(ECLDatasetLoader, self).__init__()
        self.path = f'{raw_data_dir}/ecl/datasets/'
        self.node_num = 321
        self._read_web_data()

        self.scaler = Scaler(scaler_type)

    def _download_url(self):  # pragma: no cover
        url = 'https://drive.google.com/uc?id=1rUPdR7R2iWFW-LMoDdHoO2g4KgnkpFzP'
        os.makedirs(self.path)
        gdown.download(url, os.path.join(self.path, 'ECL.csv'))
    
    def _read_web_data(self):
        if not os.path.isfile(os.path.join(self.path, 'ECL.csv')):
            self._download_url()
        
        y_df = pd.read_csv(os.path.join(self.path, 'ECL.csv'))

        y_df['date'] = pd.to_datetime(y_df['date'])
        y_df.rename(columns={'date': 'ds'}, inplace=True)
        u_ids = y_df.columns.to_list()
        u_ids.remove('ds')
        time_cls = time_features_from_frequency_str('h')
        for cls_ in time_cls:
            cls_name = cls_.__class__.__name__
            y_df[cls_name] = cls_(y_df['ds'].dt)

        time_stamp = y_df.drop(u_ids+['ds'], axis=1).to_numpy().T
        temp = np.array([time_stamp for _ in range(321)])

        df = y_df[u_ids].to_numpy().T
        df = np.expand_dims(df, axis=1)

        # Total X dimension = [Number of Nodes, Number of Features, Sequence Length]
        X = np.concatenate([df, temp], axis=1)
        self.X = X

    def _make_init_edge_index(self):
        self.edges = build_fully_connected_edge_idx(self.node_num)

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]
        random.shuffle(indices)

        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i: i + num_timesteps_in]))
            target.append((self.X[:, 0, i + num_timesteps_in: j]))

        # features dimension = [# iter, # Nodes, # Features, length of input timestep]
        # targets dimension = [# iter, # Nodes, # Features, length of output timestep]
        self.features = torch.FloatTensor(np.array(features))
        self.targets = torch.FloatTensor(np.array(target))

    def get_dataset(
            self, num_timesteps_in: int = 12, num_timesteps_out: int = 12, batch_size: int = 32
    ):
        self._make_init_edge_index()
        self._generate_task(num_timesteps_in, num_timesteps_out)

        data_batch = self.features.shape[0] // batch_size
        node_num, feature_dim = self.features.shape[1], self.features.shape[2]

        _batch = []
        for batch_iter in range(batch_size):
            _batch += [batch_iter] * node_num
        _batch = np.array(_batch)

        feature_Tensor = self.features[:data_batch * batch_size].reshape(data_batch, node_num * batch_size,
                                                                         feature_dim, num_timesteps_in).numpy()

        target_Tensor = self.targets[:data_batch * batch_size].reshape(data_batch, node_num * batch_size,
                                                                       num_timesteps_out).numpy()

        batch_edge = build_batch_edge_index(self.edges, num_graphs=batch_size, num_nodes=node_num)

        dataset = StaticGraphTemporalSignalBatch(edge_index=batch_edge, edge_weight=None,
                                                 features=feature_Tensor, targets=target_Tensor, batches=_batch)

        return dataset, self.X

    def get_scaler(self):
        return self.scaler

