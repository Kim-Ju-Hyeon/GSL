import torch
import numpy as np
import pickle

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


class Train_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Train_Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'train_n100.pt']

    def download(self):
        pass

    def process(self):
        lam = pickle.load(open('./data/LNP_lam_all.pickle', 'rb'))
        spike = pickle.load(open('./data/LNP_spk_all.pickle', 'rb'))

        # [tsteps, neurons] > [neurons, tsteps]
        data = spike[:, :4000000]
        lam = lam[:, :4000000]

        num_neurons = data.shape[0]
        total_time = data.shape[-1]

        time_steps = 200  # previous time steps = 20ms
        pred_steps = 20  # steps to predict
        window_size = 100  # for training only

        valid_sampling_locations = []
        valid_sampling_locations += [
            (time_steps + i)
            for i in range(total_time - time_steps + 1)
            if (i % window_size) == 0
        ]

        samples = len(valid_sampling_locations)

        ranges = [
            valid_sampling_locations[i] for i in np.random.choice(
                len(valid_sampling_locations), samples, replace=False)
        ]


        data = torch.FloatTensor(data)
        lam = torch.FloatTensor(lam)

        data_list = []

        for i, start_idx in enumerate(ranges):
            spike_inputs = data[:, start_idx - time_steps:start_idx]

            target_spike = data[:, start_idx - pred_steps:start_idx]
            target_lam = lam[:, start_idx - pred_steps:start_idx]
            target = torch.stack([target_lam, target_spike], dim=-1)

            data_item = Data(x=spike_inputs, edge_index=None, y=target)

            data_list.append(data_item)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Validation_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Validation_Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'validation_n100.pt']

    def download(self):
        pass

    def process(self):
        lam = pickle.load(open('./data/LNP_lam_all.pickle', 'rb'))
        spike = pickle.load(open('./data/LNP_spk_all.pickle', 'rb'))

        # [tsteps, neurons] > [neurons, tsteps]
        data = spike[:, 4000000:4400000]
        lam = lam[:, 4000000:4400000]

        num_neurons = data.shape[0]
        total_time = data.shape[-1]

        time_steps = 200  # previous time steps = 20ms
        pred_steps = 20  # steps to predict
        window_size = 100  # for training only

        valid_sampling_locations = []
        valid_sampling_locations += [
            (time_steps + i)
            for i in range(total_time - time_steps + 1)
            if (i % window_size) == 0
        ]

        samples = len(valid_sampling_locations)

        ranges = [
            valid_sampling_locations[i] for i in np.random.choice(
                len(valid_sampling_locations), samples, replace=False)
        ]


        data = torch.FloatTensor(data)
        lam = torch.FloatTensor(lam)

        data_list = []

        for i, start_idx in enumerate(ranges):
            spike_inputs = data[:, start_idx - time_steps:start_idx]

            target_spike = data[:, start_idx - pred_steps:start_idx]
            target_lam = lam[:, start_idx - pred_steps:start_idx]
            target = torch.stack([target_lam, target_spike], dim=-1)

            data_item = Data(x=spike_inputs, edge_index=None, y=target)

            data_list.append(data_item)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Test_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Test_Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'test_n100.pt']

    def download(self):
        pass

    def process(self):
        lam = pickle.load(open('./data/LNP_lam_all.pickle', 'rb'))
        spike = pickle.load(open('./data/LNP_spk_all.pickle', 'rb'))

        # [tsteps, neurons] > [neurons, tsteps]
        data = spike[:, 4400000:4800000]
        lam = lam[:, 4400000:4800000]

        num_neurons = data.shape[0]
        total_time = data.shape[-1]

        time_steps = 200  # previous time steps = 20ms
        pred_steps = 20  # steps to predict
        window_size = 100  # for training only

        valid_sampling_locations = []
        valid_sampling_locations += [
            (time_steps + i)
            for i in range(total_time - time_steps + 1)
            if (i % window_size) == 0
        ]

        samples = len(valid_sampling_locations)

        ranges = [
            valid_sampling_locations[i] for i in np.random.choice(
                len(valid_sampling_locations), samples, replace=False)
        ]


        data = torch.FloatTensor(data)
        lam = torch.FloatTensor(lam)

        data_list = []

        for i, start_idx in enumerate(ranges):
            spike_inputs = data[:, start_idx - time_steps:start_idx]

            target_spike = data[:, start_idx - pred_steps:start_idx]
            target_lam = lam[:, start_idx - pred_steps:start_idx]
            target = torch.stack([target_lam, target_spike], dim=-1)

            data_item = Data(x=spike_inputs, edge_index=None, y=target)

            data_list.append(data_item)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])