import os
import zipfile
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from six.moves import urllib
from utils.utils import build_batch_edge_index
from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch


class TrafficDatasetLoader(object):
    def __init__(self, raw_data_dir=os.path.join(os.getcwd(), "data"), dataset_name: str = 'METR-LA'):
        super(TrafficDatasetLoader, self).__init__()
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
            X = np.load(os.path.join(self.raw_data_dir, "node_values.npy")).transpose(
                (1, 2, 0)
            )

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
            X = np.load(os.path.join(self.raw_data_dir, "pems_node_values.npy")).transpose(
                (1, 2, 0)
            )

        else:
            raise ValueError("Invalid Dataset")

        X = X.astype(np.float32)

        # Normalise as in DCRNN paper (via Z-Score Method)
        means = np.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = np.std(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)

        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = torch.LongTensor(edge_indices)
        self.edge_weights = torch.FloatTensor(values)

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i: i + num_timesteps_in]).numpy())
            target.append((self.X[:, 0, i + num_timesteps_in: j]).numpy())

        self.features = torch.FloatTensor(np.array(features))
        self.targets = torch.FloatTensor(np.array(target))

    def get_dataset(
            self, num_timesteps_in: int = 12, num_timesteps_out: int = 12, batch_size: int = 32
    ):
        self._get_edges_and_weights()
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
