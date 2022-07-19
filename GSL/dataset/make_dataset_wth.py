# Cell
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
from utils.dataset_utils import time_features_from_frequency_str
from utils.utils import build_fully_connected_edge_idx

import gdown
import numpy as np
import pandas as pd

from dataset.make_dataset_base import DatasetLoader


class WTHDatasetLoader(DatasetLoader):
    def __init__(self, raw_data_dir, scaler_type='std'):
        super(WTHDatasetLoader, self).__init__(raw_data_dir, scaler_type)
        self.node_num = 12
        self._read_web_data()

    def _download_url(self):
        url = 'https://drive.google.com/uc?id=1UBRz-aM_57i_KCC-iaSWoKDPTGGv6EaG'
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        gdown.download(url, os.path.join(self.path, 'WTH.csv'))

    def _read_web_data(self):
        if not os.path.isfile(os.path.join(self.path, 'WTH.csv')):
            self._download_url()

        y_df = pd.read_csv(os.path.join(self.path, 'WTH.csv'))

        y_df['date'] = pd.to_datetime(y_df['date'])
        y_df.rename(columns={'date': 'ds'}, inplace=True)
        u_ids = y_df.columns.to_list()
        u_ids.remove('ds')
        time_cls = time_features_from_frequency_str('h')
        for cls_ in time_cls:
            cls_name = cls_.__class__.__name__
            y_df[cls_name] = cls_(y_df['ds'].dt)

        time_stamp = y_df.drop(u_ids + ['ds'], axis=1).to_numpy().T
        temp = np.array([time_stamp for _ in range(self.node_num)])

        df = y_df[u_ids].to_numpy().T
        df = np.expand_dims(df, axis=1)
        df = self.scaler.scale(df)

        # Total X dimension = [Number of Nodes, Number of Features, Sequence Length]
        X = np.concatenate([df, temp], axis=1)
        self.X = X

        total_sequence_length = X.shape[-1]
        train_index = int(total_sequence_length * 0.6) + 1
        valid_index = int(total_sequence_length * 0.2) + 1 + train_index

        self.train_X = X[:, :, :train_index]
        self.valid_X = X[:, :, train_index:valid_index]
        self.test_X = X[:, :, valid_index:]
        self.entire_dataset = torch.FloatTensor(X)

    def _make_init_edge_index(self):
        self.edges = build_fully_connected_edge_idx(self.node_num)