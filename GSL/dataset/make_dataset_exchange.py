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
from dataset.make_dataset_base import DatasetLoader


class ExchangeDatasetLoader(DatasetLoader):
    def __init__(self, raw_data_dir, scaler_type='std'):
        super(ExchangeDatasetLoader, self).__init__(raw_data_dir, scaler_type)
        self.node_num = 8
        self._read_web_data()

    # def _download_url(self):
    #     # url = 'https://drive.google.com/uc?id=1rUPdR7R2iWFW-LMoDdHoO2g4KgnkpFzP'
    #     if os.path.exists(self.path):
    #         os.makedirs(self.path)
    #     gdown.download(url, os.path.join(self.path, 'covid19.csv'))

    def _read_web_data(self):
        # if not os.path.isfile(os.path.join(self.path, 'ECL.csv')):
        #     self._download_url()

        y_df = pd.read_csv(os.path.join(self.path, 'exchange.csv'), index_col=0)

        X = y_df.to_numpy().T
        X = np.expand_dims(X, axis=1)
        X = self.scaler.scale(X)

        # Total X dimension = [Number of Nodes, Number of Features, Sequence Length]
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