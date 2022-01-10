import torch
import numpy as np
import mat73
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as DL_PyG
from torch.utils.data import DataLoader as DL_Py
from torch.utils.data.dataset import TensorDataset
from scipy.io import loadmat

import os
from tqdm import tqdm
root_dir = os.getcwd()
data_dir = root_dir + '/data'

# [sims, tsteps, features, nodes] / x = [nodes, features], edge_index = []
# edge > adjacent matrix [sims, nodes, nodes]

# Using PyTorch Geometric

class spike_LNP_raw_train_whole(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spike_LNP_raw_train_whole, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        #return ['neuron/spike/LNP/raw/train_whole_20.pt']
        return ['neuron/spike/LNP/raw/train_n50.pt']
        #return ['neuron/spike/LNP/raw/train_whole_40.pt']
        #return ['neuron/spike/LNP/raw/train_whole_200.pt']
    
    def download(self):
        pass

    def process(self):
        if not os.path.exists(data_dir + '/processed/neuron/spike/LNP/raw'):
            os.makedirs(data_dir + '/processed/neuron/spike/LNP/raw')

        #lam = mat73.loadmat('data/binary_spike_raw/LNP_lam_all.mat')
        #spk = mat73.loadmat('data/binary_spike_raw/LNP_spk_all.mat')
        lam = mat73.loadmat('data/binary_spike_raw/LNP_lam_n50.mat')
        spk = mat73.loadmat('data/binary_spike_raw/LNP_spk_n50.mat')
        
        lam = lam['lambda']
        spk = spk['spikes']

        #[tsteps, neurons] > [neurons, tsteps]
        data = spk[:4000000]
        lam = lam[:4000000]
        data = data.transpose((1, 0))
        lam = lam.transpose((1, 0))

        num_neurons = data.shape[0]
        total_time = data.shape[-1]
        time_steps = 200 #previous time steps = 20ms
        pred_steps = 20 #steps to predict
        window_size = time_steps + pred_steps - 1 # for training only
        batch_size = int(np.floor(total_time / (window_size + 1)) - 1)

        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        data = torch.FloatTensor(data)
        lam = torch.FloatTensor(lam)
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        
        for i in tqdm(range(batch_size)):
            step = i * (window_size+1)
            data_sample = data[:, step:step+window_size]
            lam_tar = lam[:, step+time_steps:step+time_steps+pred_steps]
            spk_tar = data[:, step+time_steps:step+time_steps+pred_steps]
            lam_spk_tar = torch.stack([lam_tar, spk_tar], dim=-1)
            data_item = Data(x=data_sample, edge_index=encoder_edge, y=lam_spk_tar)
            data_list.append(data_item)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class spike_LNP_raw_valid_whole(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spike_LNP_raw_valid_whole, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        #return ['neuron/spike/LNP/raw/valid_whole_20.pt']
        return ['neuron/spike/LNP/raw/valid_n50.pt']
        #return ['neuron/spike/LNP/raw/valid_whole_40.pt']
        #return ['neuron/spike/LNP/raw/valid_whole_200.pt']
    
    def download(self):
        pass

    def process(self):
        if not os.path.exists(data_dir + '/processed/neuron/spike/LNP/raw'):
            os.makedirs(data_dir + '/processed/neuron/spike/LNP/raw')

        #lam = mat73.loadmat('data/binary_spike_raw/LNP_lam_all.mat')
        #spk = mat73.loadmat('data/binary_spike_raw/LNP_spk_all.mat')
        lam = mat73.loadmat('data/binary_spike_raw/LNP_lam_n50.mat')
        spk = mat73.loadmat('data/binary_spike_raw/LNP_spk_n50.mat')
        
        lam = lam['lambda']
        spk = spk['spikes']

        #[tsteps, neurons] > [neurons, tsteps]
        data = spk[4000000:4400000]
        lam = lam[4000000:4400000]
        data = data.transpose((1, 0))
        lam = lam.transpose((1, 0))

        num_neurons = data.shape[0]
        total_time = data.shape[-1]
        time_steps = 200 #previous time steps = 20ms
        pred_steps = 20 #steps to predict
        window_size = time_steps + pred_steps - 1 # for training only
        batch_size = int(np.floor(total_time / (window_size + 1)) - 1)

        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        data = torch.FloatTensor(data)
        lam = torch.FloatTensor(lam)
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        
        for i in tqdm(range(batch_size)):
            step = i * (window_size+1)
            data_sample = data[:, step:step+window_size]
            lam_tar = lam[:, step+time_steps:step+time_steps+pred_steps]
            spk_tar = data[:, step+time_steps:step+time_steps+pred_steps]
            lam_spk_tar = torch.stack([lam_tar, spk_tar], dim=-1)
            data_item = Data(x=data_sample, edge_index=encoder_edge, y=lam_spk_tar)
            data_list.append(data_item)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class spike_LNP_raw_test_whole(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spike_LNP_raw_test_whole, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        #return ['neuron/spike/LNP/raw/test_whole_20.pt']
        return ['neuron/spike/LNP/raw/test_n50.pt']
        #return ['neuron/spike/LNP/raw/test_whole_40.pt']
        #return ['neuron/spike/LNP/raw/test_whole_200.pt']
    
    def download(self):
        pass

    def process(self):
        if not os.path.exists(data_dir + '/processed/neuron/spike/LNP/raw'):
            os.makedirs(data_dir + '/processed/neuron/spike/LNP/raw')

        #lam = mat73.loadmat('data/binary_spike_raw/LNP_lam_all.mat')
        #spk = mat73.loadmat('data/binary_spike_raw/LNP_spk_all.mat')
        lam = mat73.loadmat('data/binary_spike_raw/LNP_lam_n50.mat')
        spk = mat73.loadmat('data/binary_spike_raw/LNP_spk_n50.mat')
        
        lam = lam['lambda']
        spk = spk['spikes']

        #[tsteps, neurons] > [neurons, tsteps]
        data = spk[4400000:4800000]
        lam = lam[4400000:4800000]
        data = data.transpose((1, 0))
        lam = lam.transpose((1, 0))

        num_neurons = data.shape[0]
        total_time = data.shape[-1]
        time_steps = 200 #previous time steps = 20ms
        pred_steps = 20 #steps to predict
        #pred_steps = 40 #steps to predict
        #pred_steps = 200 #steps to predict
        window_size = time_steps + pred_steps - 1 # for training only
        batch_size = int(np.floor(total_time / (window_size + 1)) - 1)

        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        data = torch.FloatTensor(data)
        lam = torch.FloatTensor(lam)
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        
        iter_over = int((total_time - time_steps) / pred_steps)
        for i in tqdm(range(iter_over)):
            step = i * pred_steps
            data_sample = data[:, step:step+window_size]
            lam_tar = lam[:, step+time_steps:step+time_steps+pred_steps]
            spk_tar = data[:, step+time_steps:step+time_steps+pred_steps]
            lam_spk_tar = torch.stack([lam_tar, spk_tar], dim=-1)
            data_item = Data(x=data_sample, edge_index=encoder_edge, y=lam_spk_tar)
            data_list.append(data_item)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class spike_LNP_bin_train_whole(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spike_LNP_bin_train_whole, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['neuron/spike/LNP/bin/train_whole_n100.pt']

    def download(self):
        pass

    def process(self):
        if not os.path.exists(data_dir + '/processed/neuron/spike/LNP/bin'):
            os.makedirs(data_dir + '/processed/neuron/spike/LNP/bin')

        lam = mat73.loadmat('data/binary_spike_bin/lam_bin_n100.mat')
        spk = mat73.loadmat('data/binary_spike_bin/spk_bin_n100.mat')
        
        lam = lam['lam_bin']
        spk = spk['spk_bin']

        #[tsteps, neurons] > [neurons, tsteps]
        data = spk[:4000000]
        lam = lam[:4000000]
        data = data.transpose((1, 0))
        lam = lam.transpose((1, 0))

        num_neurons = data.shape[0]
        total_time = data.shape[-1]
        time_steps = 200 #previous time steps = 20ms
        pred_steps = 20 #steps to predict
        window_size = time_steps + pred_steps - 1 # for training only
        batch_size = int(np.floor(total_time / (window_size + 1)) - 1)

        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        data = torch.FloatTensor(data)
        lam = torch.FloatTensor(lam)
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        
        for i in tqdm(range(batch_size)):
            step = i * (window_size+1)
            data_sample = data[:, step:step+window_size]
            lam_tar = lam[:, step+time_steps:step+time_steps+pred_steps]
            spk_tar = data[:, step+time_steps:step+time_steps+pred_steps]
            lam_spk_tar = torch.stack([lam_tar, spk_tar], dim=-1)
            data_item = Data(x=data_sample, edge_index=encoder_edge, y=lam_spk_tar)
            data_list.append(data_item)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class spike_LNP_bin_valid_whole(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spike_LNP_bin_valid_whole, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['neuron/spike/LNP/bin/train_whole_n100.pt']
    
    def download(self):
        pass

    def process(self):
        if not os.path.exists(data_dir + '/processed/neuron/spike/LNP/bin'):
            os.makedirs(data_dir + '/processed/neuron/spike/LNP/bin')

        lam = mat73.loadmat('data/binary_spike_bin/lam_bin_n100.mat')
        spk = mat73.loadmat('data/binary_spike_bin/spk_bin_n100.mat')
        
        lam = lam['lam_bin']
        spk = spk['spk_bin']

        #[tsteps, neurons] > [neurons, tsteps]
        data = spk[4000000:4400000]
        lam = lam[4000000:4400000]
        data = data.transpose((1, 0))
        lam = lam.transpose((1, 0))

        num_neurons = data.shape[0]
        total_time = data.shape[-1]
        time_steps = 200 #previous time steps = 20ms
        pred_steps = 20 #steps to predict
        window_size = time_steps + pred_steps - 1 # for training only
        batch_size = int(np.floor(total_time / (window_size + 1)) - 1)

        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        data = torch.FloatTensor(data)
        lam = torch.FloatTensor(lam)
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        
        for i in tqdm(range(batch_size)):
            step = i * (window_size+1)
            data_sample = data[:, step:step+window_size]
            lam_tar = lam[:, step+time_steps:step+time_steps+pred_steps]
            spk_tar = data[:, step+time_steps:step+time_steps+pred_steps]
            lam_spk_tar = torch.stack([lam_tar, spk_tar], dim=-1)
            data_item = Data(x=data_sample, edge_index=encoder_edge, y=lam_spk_tar)
            data_list.append(data_item)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class spike_LNP_bin_test_whole(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(spike_LNP_bin_test_whole, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['neuron/spike/LNP/bin/test_whole_n100.pt']

    def download(self):
        pass

    def process(self):
        if not os.path.exists(data_dir + '/processed/neuron/spike/LNP/bin'):
            os.makedirs(data_dir + '/processed/neuron/spike/LNP/bin')

        lam = mat73.loadmat('data/binary_spike_bin/lam_bin_n100.mat')
        spk = mat73.loadmat('data/binary_spike_bin/spk_bin_n100.mat')
        
        lam = lam['lam_bin']
        spk = spk['spk_bin']

        #[tsteps, neurons] > [neurons, tsteps]
        data = spk[4400000:4800000]
        lam = lam[4400000:4800000]
        data = data.transpose((1, 0))
        lam = lam.transpose((1, 0))

        num_neurons = data.shape[0]
        total_time = data.shape[-1]
        time_steps = 200 #previous time steps = 20ms
        pred_steps = 20 #steps to predict
        window_size = time_steps + pred_steps - 1 # for training only
        batch_size = int(np.floor(total_time / (window_size + 1)) - 1)

        fully_connected = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        encoder_edge = np.where(fully_connected)
        encoder_edge = np.array([encoder_edge[0], encoder_edge[1]], dtype=np.int64)

        data = torch.FloatTensor(data)
        lam = torch.FloatTensor(lam)
        encoder_edge = torch.LongTensor(encoder_edge)

        data_list = []
        
        iter_over = int((total_time - time_steps) / pred_steps)
        for i in tqdm(range(iter_over)):
            step = i * pred_steps
            data_sample = data[:, step:step+window_size]
            lam_tar = lam[:, step+time_steps:step+time_steps+pred_steps]
            spk_tar = data[:, step+time_steps:step+time_steps+pred_steps]
            lam_spk_tar = torch.stack([lam_tar, spk_tar], dim=-1)
            data_item = Data(x=data_sample, edge_index=encoder_edge, y=lam_spk_tar)
            data_list.append(data_item)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])