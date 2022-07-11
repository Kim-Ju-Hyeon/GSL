import os
import zipfile
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from utils.scalers import Scaler
from six.moves import urllib
from utils.utils import build_batch_edge_index
from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch

import random


class TrafficDatasetLoader(object):
    def __init__(self, raw_data_dir=os.path.join(os.getcwd(), "data"), dataset_name: str = 'METR-LA',
                 scaler_type='std'):
        super(TrafficDatasetLoader, self).__init__()
        self.scaler = Scaler(scaler_type)
        self.raw_data_dir = raw_data_dir
        self._dataset_name = dataset_name
        self._read_web_data()

    def _download_url(self, url, save_path):  # pragma: no cover
        with urllib.request.urlopen(url) as dl_file:
            with open(save_path, "wb") as out_file:
                out_file.write(dl_file.read())

    def _read_web_data(self):
        if self._dataset_name == 'METR-LA':
            url = "https://graphmining.ai/temporal_datasets/METR-LA.zip"

            # Check if zip file is in data folder from working directory, otherwise download
            if not os.path.isfile(
                    os.path.join(self.raw_data_dir, "METR-LA.zip")
            ):  # pragma: no cover
                if not os.path.exists(self.raw_data_dir):
                    os.makedirs(self.raw_data_dir)
                self._download_url(url, os.path.join(self.raw_data_dir, "METR-LA.zip"))

            if not os.path.isfile(
                    os.path.join(self.raw_data_dir, "adj_mat.npy")
            ) or not os.path.isfile(
                os.path.join(self.raw_data_dir, "node_values.npy")
            ):  # pragma: no cover
                with zipfile.ZipFile(
                        os.path.join(self.raw_data_dir, "METR-LA.zip"), "r"
                ) as zip_fh:
                    zip_fh.extractall(self.raw_data_dir)

            A = np.load(os.path.join(self.raw_data_dir, "adj_mat.npy"))
            X = np.load(os.path.join(self.raw_data_dir, "node_values.npy"))

        elif self._dataset_name == 'PEMS-BAY':
            url = "https://graphmining.ai/temporal_datasets/PEMS-BAY.zip"

            # Check if zip file is in data folder from working directory, otherwise download
            if not os.path.isfile(
                    os.path.join(self.raw_data_dir, "PEMS-BAY.zip")
            ):  # pragma: no cover
                if not os.path.exists(self.raw_data_dir):
                    os.makedirs(self.raw_data_dir)
                self._download_url(url, os.path.join(self.raw_data_dir, "PEMS-BAY.zip"))

            if not os.path.isfile(
                    os.path.join(self.raw_data_dir, "pems_adj_mat.npy")
            ) or not os.path.isfile(
                os.path.join(self.raw_data_dir, "pems_node_values.npy")
            ):  # pragma: no cover
                with zipfile.ZipFile(
                        os.path.join(self.raw_data_dir, "PEMS-BAY.zip"), "r"
                ) as zip_fh:
                    zip_fh.extractall(self.raw_data_dir)

            A = np.load(os.path.join(self.raw_data_dir, "pems_adj_mat.npy"))
            X = np.load(os.path.join(self.raw_data_dir, "pems_node_values.npy"))

        else:
            raise ValueError("Invalid Dataset")

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
        self.entire_dataset = torch.from_numpy(X)

    def _make_init_edge_index(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = torch.LongTensor(edge_indices)
        self.edge_weights = torch.FloatTensor(values)
