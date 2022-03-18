import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch.nn import functional as F
import torch.nn as nn

from collections import defaultdict
import torch.optim as optim
from torch_geometric.loader import DataLoader

from models.GTS.GTS_model import GTS_Model
from utils.utils import build_fully_connected_edge_idx
from dataset.make_spike_datset import MakeSpikeDataset
from utils.train_helper import model_snapshot, load_model
from utils.logger import get_logger
from torch_geometric_temporal.dataset import METRLADatasetLoader, PemsBayDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

from pytorch_forecasting.metrics import MAPE, RMSE



logger = get_logger('exp_logger')


class GTS_Runner(object):
    def __init__(self, config):
        self.config = config
        self.exp_dir = config.exp_dir
        self.model_save = config.model_save
        self.seed = config.seed
        self.use_gpu = config.use_gpu
        self.device = config.device

        self.best_model_dir = os.path.join(self.model_save, 'best.pth')
        self.ck_dir = os.path.join(self.exp_dir, 'training.ck')

        self.dataset_conf = config.dataset
        self.train_conf = config.train

        if self.train_conf.loss_function == 'MAE':
            self.loss = nn.L1Loss()
        elif self.train_conf.loss_function == 'RMSE':
            self.loss = RMSE()
        elif self.train_conf.loss_function == 'MAPE':
            self.loss = MAPE()
        elif self.train_conf.loss_function == 'Poisson':
            self.loss = nn.PoissonNLLLoss()
        else:
            raise ValueError('Non-supported Loss Function')

        self.node_nums = config.nodes_num

        self.graph_learning_module = config.graph_learning_module
        self.graph_forecasting_module = config.graph_forecasting_module

        self.initial_edge_index = config.initial_edge_index

        if self.initial_edge_index == 'Fully Connected':
            self.init_edge_index = build_fully_connected_edge_idx(self.config.nodes_num)
        else:
            raise ValueError("Non-supported Edge Index!")

        if self.use_gpu and (self.device != 'cpu'):
            self.init_edge_index = self.init_edge_index.to(device=self.device)

        if self.dataset_conf.name == 'spike_lambda_bin100':
            print("Load Spike Dataset")
            spike = pickle.load(open('./data/spk_bin_n100.pickle', 'rb'))

            self.entire_inputs = torch.FloatTensor(spike[:, :self.dataset_conf.graph_learning_length])

            if self.use_gpu and (self.device != 'cpu'):
                self.entire_inputs = self.entire_inputs.to(device=self.device)

            dataset_maker = MakeSpikeDataset(self.config)
            total_dataset = dataset_maker.make()

            self.train_dataset = DataLoader(total_dataset['train'], batch_size=self.train_conf.batch_size)
            self.valid_dataset = DataLoader(total_dataset['valid'], batch_size=self.train_conf.batch_size)
            self.test_dataset = DataLoader(total_dataset['test'], batch_size=self.train_conf.batch_size)

        elif self.dataset_conf.name == 'METR-LA':
            loader = METRLADatasetLoader(raw_data_dir='./data/METR-LA')
            dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
            self.train_dataset, _dataset = temporal_signal_split(dataset, train_ratio=0.8)
            self.validation_dataset, self.test_dataset = temporal_signal_split(_dataset, train_ratio=0.5)

        elif self.dataset_conf.name == 'PEMS-BAY':
            loader = PemsBayDatasetLoader(raw_data_dir='./data/PEMS-BAY')
            dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
            self.train_dataset, _dataset = temporal_signal_split(dataset, train_ratio=0.8)
            self.validation_dataset, self.test_dataset = temporal_signal_split(_dataset, train_ratio=0.5)

        else:
            raise ValueError("Non-supported dataset!")

        self.model = GTS_Model(self.config)

        if self.use_gpu and (self.device != 'cpu'):
            self.model = self.model.to(device=self.device)

    def train(self):
        print('Train Start')
        # create optimizer
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr)
        else:
            raise ValueError("Non-supported optimizer!")

        results = defaultdict(list)
        best_val_loss = np.inf

        # ========================= Training Loop ============================= #
        for epoch in range(self.train_conf.epoch):
            # ====================== training ============================= #
            self.model.train()

            train_loss = []

            iters = 0
            for data_batch in tqdm(self.train_dataset):

                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)

                _, outputs = self.model(data_batch.x, data_batch.y, self.entire_inputs, self.init_edge_index)
                loss = self.loss(outputs, data_batch.y)

                # backward pass (accumulates gradients).
                loss.backward()

                # performs a single update step.
                optimizer.step()
                optimizer.zero_grad()

                train_loss += [float(loss.data.cpu().numpy())]
                iters += 1

                # display loss
                if (iters + 1) % 100 == 0:
                    logger.info(
                        "Train Loss @ epoch {} iteration {} = {}".format(epoch + 1, iters + 1,
                                                                         float(loss.data.cpu().numpy())))

            train_loss = np.stack(train_loss).mean()
            results['train_loss'] += [train_loss]

            # ===================== validation ============================ #
            self.model.eval()

            val_loss = []
            for data_batch in tqdm(self.validation_dataset):

                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)

                with torch.no_grad():
                    adj_matrix, outputs = self.model(data_batch.x, data_batch.y, self.entire_inputs, self.init_edge_index)

                loss = self.loss(outputs, data_batch.y)
                val_loss += [float(loss.data.cpu().numpy())]

            val_loss = np.stack(val_loss).mean()

            results['val_loss'] += [val_loss]

            if type(adj_matrix) == dict:
                results['val_adj_matirix'] += adj_matrix
            else:
                results['val_adj_matirix'] += [adj_matrix.detach().cpu()]

            logger.info("Epoch {} Avg. Validation Loss = {:.6}".format(epoch + 1, val_loss, 0))
            logger.info("Current Best Validation Loss = {:.6}".format(best_val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_dir)

            model_snapshot(epoch, self.model, optimizer, best_val_loss, self.ck_dir)

        pickle.dump(results, open(os.path.join(self.config.exp_dir, 'training_result.pickle'), 'wb'))

    def test(self):
        print('Test Start')

        self.best_model = GTS_Model(self.config)
        best_snapshot = load_model(self.best_model_dir)

        self.best_model.load_state_dict(best_snapshot)

        if self.use_gpu and (self.device != 'cpu'):
            self.best_model = self.best_model.to(device=self.device)

        # ===================== validation ============================ #
        self.best_model.eval()

        test_loss = []
        results = defaultdict(list)
        output = []
        target = []
        for data_batch in tqdm(self.test_dataset):

            if self.use_gpu and (self.device != 'cpu'):
                data_batch = data_batch.to(device=self.device)

            with torch.no_grad():
                adj_matrix, outputs = self.model(data_batch.x, data_batch.y, self.entire_inputs, self.init_edge_index)

            loss = self.loss(outputs, data_batch.y)

            test_loss += [float(loss.data.cpu().numpy())]
            output += [outputs.cpu()]
            target += [data_batch.y.cpu()]

        test_loss = np.stack(test_loss).mean()

        results['test_loss'] += [test_loss]
        results['adj_matrix'] = adj_matrix
        results['prediction'] = output
        results['target'] = target

        logger.info("Avg. Test Loss = {:.6}".format(test_loss, 0))

        pickle.dump(results, open(os.path.join(self.config.exp_dir, 'test_result.pickle'), 'wb'))
