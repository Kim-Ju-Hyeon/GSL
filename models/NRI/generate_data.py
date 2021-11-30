import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as DL_PyG
from torch.utils.data import DataLoader as DL_Py
from torch.utils.data.dataset import TensorDataset

import yaml
import os
from tqdm import tqdm
root_dir = os.getcwd()
data_dir = root_dir + '/data'

# [sims, tsteps, features, nodes] / x = [nodes, features], edge_index = []
# edge > adjacent matrix [sims, nodes, nodes]

# Using PyTorch Geometric
class spring5_edge2_train(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spring5_edge2_train, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['spring5_train.pt']

    def download(self):
        pass

    def process(self): #[sims, nodes * nodes-1(n C 2)]
        train_loc = np.load('./data/loc_train_springs5.npy')
        train_vel = np.load('./data/vel_train_springs5.npy')
        train_edge = np.load('./data/edges_train_springs5.npy')

        # reshape to [sims, nodes, tsteps, features]
        train_loc = train_loc.transpose([0, 3, 1, 2])
        train_vel = train_vel.transpose([0, 3, 1, 2])
        train_edge = train_edge.reshape([-1, train_loc.shape[1] ** 2])

        # Normalize to [-1, 1]

        train_feat = np.concatenate([train_loc, train_vel], axis=-1)

        fully_connected = np.ones((train_loc.shape[1], train_loc.shape[1])) - np.eye(train_loc.shape[1])
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        off_diag_idx = np.ravel_multi_index(np.where(fully_connected), [train_loc.shape[1], train_loc.shape[1]])

        train_edge = train_edge[:, off_diag_idx]
        train_feat = torch.FloatTensor(train_feat)
        train_edge = torch.LongTensor(train_edge)
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        sims, nodes = train_feat.shape[0], train_feat.shape[1]

        for i in tqdm(range(sims)):
            train_feat = train_feat.view((sims, nodes, -1))
            node_features = train_feat[i, :, :]
            data = Data(x=node_features, edge_index=encoder_edge, y=train_edge[i, :])
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class spring5_edge2_valid(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spring5_edge2_valid, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['spring5_valid.pt']

    def download(self):
        pass

    def process(self): #[sims, nodes * nodes-1(n C 2)]
        valid_loc = np.load('./data/loc_valid_springs5.npy')
        valid_vel = np.load('./data/vel_valid_springs5.npy')
        valid_edge = np.load('./data/edges_valid_springs5.npy')

        # reshape to [sims, nodes, tsteps, features]
        valid_loc = valid_loc.transpose([0, 3, 1, 2])
        valid_vel = valid_vel.transpose([0, 3, 1, 2])
        valid_edge = valid_edge.reshape([-1, valid_loc.shape[1] ** 2])

        # Normalize to [-1, 1]

        valid_feat = np.concatenate([valid_loc, valid_vel], axis=-1)

        fully_connected = np.ones((valid_loc.shape[1], valid_loc.shape[1])) - np.eye(valid_loc.shape[1])
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        off_diag_idx = np.ravel_multi_index(np.where(fully_connected), [valid_loc.shape[1], valid_loc.shape[1]])

        valid_edge = valid_edge[:, off_diag_idx]
        valid_feat = torch.FloatTensor(valid_feat)
        valid_edge = torch.LongTensor(valid_edge)
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        sims, nodes = valid_feat.shape[0], valid_feat.shape[1]

        for i in tqdm(range(sims)):
            train_feat = train_feat.view((sims, nodes, -1))
            node_features = train_feat[i, :, :]
            data = Data(x=node_features, edge_index=encoder_edge, y=valid_edge[i, :])
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class spring5_edge2_test(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spring5_edge2_test, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['spring5_test.pt']

    def download(self):
        pass

    def process(self): #[sims, nodes * nodes-1(n C 2)]
        train_loc = np.load('./data/loc_test_springs5.npy')
        train_vel = np.load('./data/vel_test_springs5.npy')
        train_edge = np.load('./data/edges_test_springs5.npy')

        # reshape to [sims, nodes, tsteps, features]
        train_loc = train_loc.transpose([0, 3, 1, 2])
        train_vel = train_vel.transpose([0, 3, 1, 2])
        train_edge = train_edge.reshape([-1, train_loc.shape[1] ** 2])

        # Normalize to [-1, 1]

        train_feat = np.concatenate([train_loc, train_vel], axis=-1)

        fully_connected = np.ones((train_loc.shape[1], train_loc.shape[1])) - np.eye(train_loc.shape[1])
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        off_diag_idx = np.ravel_multi_index(np.where(fully_connected), [train_loc.shape[1], train_loc.shape[1]])

        train_edge = train_edge[:, off_diag_idx]
        train_feat = torch.FloatTensor(train_feat)
        train_edge = torch.LongTensor(train_edge)
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        sims, nodes = train_feat.shape[0], train_feat.shape[1]

        for i in tqdm(range(sims)):
            train_feat = train_feat.view((sims, nodes, -1))
            node_features = train_feat[i, :, :]
            data = Data(x=node_features, edge_index=encoder_edge, y=train_edge[i, :])
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class spring5_edge3_train(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spring5_edge3_train, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['spring5_edge3_train.pt']

    def download(self):
        pass

    def process(self): #[sims, nodes * nodes-1(n C 2)]
        train_loc = np.load('./data/edge_3/loc_train_springs5.npy')
        train_vel = np.load('./data/edge_3/vel_train_springs5.npy')
        train_edge = np.load('./data/edge_3/edges_train_springs5.npy')

        # reshape to [sims, nodes, tsteps, features]
        train_loc = train_loc.transpose([0, 3, 1, 2])
        train_vel = train_vel.transpose([0, 3, 1, 2])
        train_edge = train_edge.reshape([-1, train_loc.shape[1] ** 2])

        # Normalize to [-1, 1]

        train_feat = np.concatenate([train_loc, train_vel], axis=-1)

        fully_connected = np.ones((train_loc.shape[1], train_loc.shape[1])) - np.eye(train_loc.shape[1])
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        off_diag_idx = np.ravel_multi_index(np.where(fully_connected), [train_loc.shape[1], train_loc.shape[1]])

        train_edge = train_edge[:, off_diag_idx]
        train_feat = torch.FloatTensor(train_feat)
        train_edge = torch.FloatTensor(train_edge) # [0, 0.5, 1] regression
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        sims, nodes = train_feat.shape[0], train_feat.shape[1]

        for i in tqdm(range(sims)):
            train_feat = train_feat.view((sims, nodes, -1))
            node_features = train_feat[i, :, :]
            data = Data(x=node_features, edge_index=encoder_edge, y=train_edge[i, :]*2)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class spring5_edge3_valid(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spring5_edge3_valid, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['spring5_edge3_valid.pt']

    def download(self):
        pass

    def process(self): #[sims, nodes * nodes-1(n C 2)]
        train_loc = np.load('./data/edge_3/loc_valid_springs5.npy')
        train_vel = np.load('./data/edge_3/vel_valid_springs5.npy')
        train_edge = np.load('./data/edge_3/edges_valid_springs5.npy')

        # reshape to [sims, nodes, tsteps, features]
        train_loc = train_loc.transpose([0, 3, 1, 2])
        train_vel = train_vel.transpose([0, 3, 1, 2])
        train_edge = train_edge.reshape([-1, train_loc.shape[1] ** 2])

        # Normalize to [-1, 1]

        train_feat = np.concatenate([train_loc, train_vel], axis=-1)

        fully_connected = np.ones((train_loc.shape[1], train_loc.shape[1])) - np.eye(train_loc.shape[1])
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        off_diag_idx = np.ravel_multi_index(np.where(fully_connected), [train_loc.shape[1], train_loc.shape[1]])

        train_edge = train_edge[:, off_diag_idx]
        train_feat = torch.FloatTensor(train_feat)
        train_edge = torch.FloatTensor(train_edge) # [0, 0.5, 1] regression
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        sims, nodes = train_feat.shape[0], train_feat.shape[1]

        for i in tqdm(range(sims)):
            train_feat = train_feat.view((sims, nodes, -1))
            node_features = train_feat[i, :, :]
            data = Data(x=node_features, edge_index=encoder_edge, y=train_edge[i, :]*2)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class spring5_edge3_test(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spring5_edge3_test, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['spring5_edge3_test.pt']

    def download(self):
        pass

    def process(self): #[sims, nodes * nodes-1(n C 2)]
        train_loc = np.load('./data/edge_3/loc_test_springs5.npy')
        train_vel = np.load('./data/edge_3/vel_test_springs5.npy')
        train_edge = np.load('./data/edge_3/edges_test_springs5.npy')

        # reshape to [sims, nodes, tsteps, features]
        train_loc = train_loc.transpose([0, 3, 1, 2])
        train_vel = train_vel.transpose([0, 3, 1, 2])
        train_edge = train_edge.reshape([-1, train_loc.shape[1] ** 2])

        # Normalize to [-1, 1]

        train_feat = np.concatenate([train_loc, train_vel], axis=-1)

        fully_connected = np.ones((train_loc.shape[1], train_loc.shape[1])) - np.eye(train_loc.shape[1])
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        off_diag_idx = np.ravel_multi_index(np.where(fully_connected), [train_loc.shape[1], train_loc.shape[1]])

        train_edge = train_edge[:, off_diag_idx]
        train_feat = torch.FloatTensor(train_feat)
        train_edge = torch.FloatTensor(train_edge) # [0, 0.5, 1] regression
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        sims, nodes = train_feat.shape[0], train_feat.shape[1]

        for i in tqdm(range(sims)):
            train_feat = train_feat.view((sims, nodes, -1))
            node_features = train_feat[i, :, :]
            data = Data(x=node_features, edge_index=encoder_edge, y=train_edge[i, :]*2)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class spike_binned(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spike_binned, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['spike_binned.pt']
    
    def download(self):
        pass

    def process(self):
        data = np.load('data/connectivity_train_binary_binned100.npy')
        num_neurons = data.shape[-1]
        time_steps = 100 # recording 1(s) of binned spike data

        data = data.reshape(-1, time_steps, num_neurons) #[batch_size, tsteps, num_neurons]
        data = data.transpose([0, 2, 1])
        target = np.load('data/connectivity_W100.npy').reshape(-1)

        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)
        
        off_diag_idx = np.ravel_multi_index(np.where(fully_connected),
                                            [num_neurons, num_neurons])
        weight_profile = target[off_diag_idx]

        data = torch.FloatTensor(data)
        weight_profile = torch.FloatTensor(weight_profile)
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []

        for i in tqdm(range(data.size(0))):
            node_features = data[i, :, :]
            data_item = Data(x=node_features, edge_index=encoder_edge, y=weight_profile)
            data_list.append(data_item)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class activation_binned(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(activation_binned, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['activation_binned_test.pt'] #depends on
    
    def download(self):
        pass

    def process(self):
        data_all = np.load('data/raw/data_all.npy')

        num_neurons = data_all.shape[-1]        
        time_steps = 200
        np.save('data/raw/data_all.npy', data_all)

        data = np.reshape(data_all, (-1, time_steps, num_neurons))
        data = data.transpose([0, 2, 1]) # batch_size, num_neurons, tsteps
        target = np.load('data/connectivity_W100.npy').reshape(-1)

        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)
        
        off_diag_idx = np.ravel_multi_index(np.where(fully_connected),
                                            [num_neurons, num_neurons])
        weight_profile = target[off_diag_idx]

        data = torch.FloatTensor(data)
        weight_profile = torch.FloatTensor(weight_profile)
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []

        for i in tqdm(range(data.size(0))):
            node_features = data[i, :, :]
            data_item = Data(x=node_features, edge_index=encoder_edge, y=weight_profile)
            data_list.append(data_item)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class charged5_edge2_train(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(charged5_edge2_train, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['charged5_edge2_train.pt']

    def download(self):
        pass

    def process(self): #[sims, nodes * nodes-1(n C 2)]
        train_loc = np.load('./data/charged_5/loc_train_charged5.npy')
        train_vel = np.load('./data/charged_5/vel_train_charged5.npy')
        train_edge = np.load('./data/charged_5/edges_train_charged5.npy')

        # reshape to [sims, nodes, tsteps, features]
        train_loc = train_loc.transpose([0, 3, 1, 2])
        train_vel = train_vel.transpose([0, 3, 1, 2])
        train_edge = train_edge.reshape([-1, train_loc.shape[1] ** 2])

        # Normalize to [-1, 1]

        train_feat = np.concatenate([train_loc, train_vel], axis=-1)

        fully_connected = np.ones((train_loc.shape[1], train_loc.shape[1])) - np.eye(train_loc.shape[1])
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        off_diag_idx = np.ravel_multi_index(np.where(fully_connected), [train_loc.shape[1], train_loc.shape[1]])

        train_edge = train_edge[:, off_diag_idx]
        train_feat = torch.FloatTensor(train_feat)
        train_edge = torch.FloatTensor(train_edge) # [0, 0.5, 1] regression
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        sims, nodes = train_feat.shape[0], train_feat.shape[1]

        for i in tqdm(range(sims)):
            train_feat = train_feat.view((sims, nodes, -1))
            node_features = train_feat[i, :, :]
            data = Data(x=node_features, edge_index=encoder_edge, y=train_edge[i, :])
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class charged5_edge2_valid(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(charged5_edge2_valid, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['charged5_edge2_valid.pt']

    def download(self):
        pass

    def process(self): #[sims, nodes * nodes-1(n C 2)]
        train_loc = np.load('./data/charged_5/loc_valid_charged5.npy')
        train_vel = np.load('./data/charged_5/vel_valid_charged5.npy')
        train_edge = np.load('./data/charged_5/edges_valid_charged5.npy')

        # reshape to [sims, nodes, tsteps, features]
        train_loc = train_loc.transpose([0, 3, 1, 2])
        train_vel = train_vel.transpose([0, 3, 1, 2])
        train_edge = train_edge.reshape([-1, train_loc.shape[1] ** 2])

        # Normalize to [-1, 1]

        train_feat = np.concatenate([train_loc, train_vel], axis=-1)

        fully_connected = np.ones((train_loc.shape[1], train_loc.shape[1])) - np.eye(train_loc.shape[1])
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        off_diag_idx = np.ravel_multi_index(np.where(fully_connected), [train_loc.shape[1], train_loc.shape[1]])

        train_edge = train_edge[:, off_diag_idx]
        train_feat = torch.FloatTensor(train_feat)
        train_edge = torch.FloatTensor(train_edge) # [0, 0.5, 1] regression
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        sims, nodes = train_feat.shape[0], train_feat.shape[1]

        for i in tqdm(range(sims)):
            train_feat = train_feat.view((sims, nodes, -1))
            node_features = train_feat[i, :, :]
            data = Data(x=node_features, edge_index=encoder_edge, y=train_edge[i, :])
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class charged5_edge2_test(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(charged5_edge2_test, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['charged5_edge2_test.pt']

    def download(self):
        pass

    def process(self): #[sims, nodes * nodes-1(n C 2)]
        train_loc = np.load('./data/charged_5/loc_test_charged5.npy')
        train_vel = np.load('./data/charged_5/vel_test_charged5.npy')
        train_edge = np.load('./data/charged_5/edges_test_charged5.npy')

        # reshape to [sims, nodes, tsteps, features]
        train_loc = train_loc.transpose([0, 3, 1, 2])
        train_vel = train_vel.transpose([0, 3, 1, 2])
        train_edge = train_edge.reshape([-1, train_loc.shape[1] ** 2])

        # Normalize to [-1, 1]

        train_feat = np.concatenate([train_loc, train_vel], axis=-1)

        fully_connected = np.ones((train_loc.shape[1], train_loc.shape[1])) - np.eye(train_loc.shape[1])
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        off_diag_idx = np.ravel_multi_index(np.where(fully_connected), [train_loc.shape[1], train_loc.shape[1]])

        train_edge = train_edge[:, off_diag_idx]
        train_feat = torch.FloatTensor(train_feat)
        train_edge = torch.FloatTensor(train_edge) # [0, 0.5, 1] regression
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        sims, nodes = train_feat.shape[0], train_feat.shape[1]

        for i in tqdm(range(sims)):
            train_feat = train_feat.view((sims, nodes, -1))
            node_features = train_feat[i, :, :]
            data = Data(x=node_features, edge_index=encoder_edge, y=train_edge[i, :])
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
