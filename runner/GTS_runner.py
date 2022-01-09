import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch.nn import functional as F
import torch.nn as nn

from models.GTS.gts_graph_learning import GTS_Graph_Learning
from models.GTS.gts_forecasting_module import GTS_Forecasting_Module

from collections import defaultdict
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

from models.GTS.GTS_model import GTS_Model
from utils.utils import build_fully_connected_edge_idx
from utils.utils import build_batch_edge_index
from dataset.make_datset import MakeDataset
from utils.train_helper import model_snapshot, load_model
from utils.logger import get_logger

logger = get_logger('exp_logger')


class GTS_Runner(object):
    def __init__(self, config):
        self.config = config
        self.exp_dir = config.exp_dir
        self.model_save = config.model_save
        self.seed = config.seed
        self.use_gpu = config.use_gpu
        self.device = config.device

        self.best_gl_model_dir = os.path.join(self.model_save, 'best_gl.pth')
        self.best_fe_model_dir = os.path.join(self.model_save, 'best_fe.pth')
        self.ck_dir = os.path.join(self.exp_dir, 'training.ck')

        self.dataset_conf = config.dataset
        self.train_conf = config.train

        self.node_nums = config.nodes_num
        self.undirected_adj = config.graph_learning.to_symmetric
        self.tau = config.tau

        self.graph_learning_module = config.graph_learning_module
        self.graph_forecasting_module = config.graph_forecasting_module

        self.fully_connected_edge_index = build_fully_connected_edge_idx(self.config.nodes_num)
        if self.use_gpu and (self.device != 'cpu'):
            self.fully_connected_edge_index = self.fully_connected_edge_index.to(device=self.device)

        if self.dataset_conf.name == 'spike_lambda_bin100':
            print("Load Spike Dataset")
            spike = pickle.load(open('./data/spk_bin_n100.pickle', 'rb'))

            self.entire_inputs = torch.FloatTensor(spike[:, :40000])
            if self.use_gpu and (self.device != 'cpu'):
                self.entire_inputs = self.entire_inputs.to(device=self.device)

            dataset_maker = MakeDataset(self.config)
            total_dataset = dataset_maker.make()

            self.train_dataset = total_dataset['train']
            self.valid_dataset = total_dataset['valid']
            self.test_dataset = total_dataset['test']

        else:
            raise ValueError("Non-supported dataset!")

        if self.use_gpu and (self.device != 'cpu'):
            self.graph_learning_module = self.graph_learning_module.to(device=self.device)
            self.graph_forecasting_module = self.graph_forecasting_module.to(device=self.device)

    def _gumbel_softmax_structure_sampling(self, adj, batch_size):
        edge_probability = F.gumbel_softmax(adj, tau=self.tau, hard=True)
        connect = torch.where(edge_probability[:, 0])

        adj_matrix = torch.stack([self.fully_connected_edge_index[0, :][connect],
                                  self.fully_connected_edge_index[1, :][connect]])
        if self.undirected_adj:
            adj_matrix = to_undirected(adj_matrix)
        batch_adj_matrix = build_batch_edge_index(adj_matrix, batch_size)

        return batch_adj_matrix, adj_matrix

    def train(self):
        print('Train Start')
        train_loader = DataLoader(self.train_dataset, batch_size=self.train_conf.batch_size)
        valid_loader = DataLoader(self.valid_dataset, batch_size=self.train_conf.batch_size)

        # create optimizer
        # params = filter(lambda p: p.requires_grad, self.graph_learning_module.parameters())
        # if self.train_conf.optimizer == 'SGD':
        #     optimizer = optim.SGD(
        #         params,
        #         lr=self.train_conf.lr)
        # elif self.train_conf.optimizer == 'Adam':
        #     optimizer = optim.Adam(
        #         params,
        #         lr=self.train_conf.lr)
        # else:
        #     raise ValueError("Non-supported optimizer!")

        optimizer = optim.Adam(lr=self.train_conf.lr)

        results = defaultdict(list)
        best_val_loss = np.inf

        # ========================= Training Loop ============================= #
        for epoch in range(self.train_conf.epoch):
            # ====================== training ============================= #
            self.graph_learning_module.train()
            self.graph_forecasting_module.train()

            train_loss = []

            iters = 0
            for data_batch in tqdm(train_loader):
                batch_size = data_batch.x.shape[0] // self.node_nums
                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)

                adj = self.graph_learning(self.entire_inputs, self.fully_connected_edge_index)
                batch_adj_matrix, _ = self._gumbel_softmax_structure_sampling(adj, batch_size)

                outputs = self.graph_forecasting_module(data_batch.x, data_batch.y, batch_adj_matrix)

                loss = F.poisson_nll_loss(outputs, data_batch.y, log_input=True)

                # backward pass (accumulates gradients).
                loss.backward()
                # performs a single update step.
                optimizer.step()
                optimizer.zero_grad()

                train_loss += [float(loss.data.cpu().numpy())]
                iters += 1

                # display loss
                if (iters + 1) % 10 == 0:
                    logger.info(
                        "Train Loss @ epoch {} iteration {} = {}".format(epoch + 1, iters + 1,
                                                                         float(loss.data.cpu().numpy())))

            train_loss = np.stack(train_loss).mean()
            results['train_loss'] += [train_loss]

            # ===================== validation ============================ #
            self.graph_learning_module.eval()
            self.graph_forecasting_module.eval()

            val_loss = []
            for data_batch in tqdm(valid_loader):
                batch_size = data_batch.x.shape[0] // self.node_nums
                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)
                with torch.no_grad():
                    adj = self.graph_learning(self.entire_inputs, self.fully_connected_edge_index)
                    batch_adj_matrix, adj_matrix = self._gumbel_softmax_structure_sampling(adj, batch_size)

                    outputs = self.graph_forecasting_module(data_batch.x, data_batch.y, batch_adj_matrix)

                loss = F.poisson_nll_loss(outputs, data_batch.y, log_input=True)
                val_loss += [float(loss.data.cpu().numpy())]

            val_loss = np.stack(val_loss).mean()

            results['val_loss'] += [val_loss]
            results['val_adj_matirix'] += [adj_matrix]

            logger.info("Avg. Validation Loss = {:.6}".format(val_loss, 0))
            logger.info("Current Best Validation Loss = {:.6}".format(best_val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.graph_learning_module.state_dict(), self.best_gl_model_dir)
                torch.save(self.graph_forecasting_module.state_dict(), self.best_fe_model_dir)

            # model_snapshot(epoch, self.graph_learning_module, optimizer, best_val_loss, self.ck_dir)
            # model_snapshot(epoch, self.graph_forecasting_module, optimizer, best_val_loss, self.ck_dir)

        pickle.dump(results, open(os.path.join(self.config.exp_dir, 'training_result.pickle'), 'wb'))

    def test(self):
        print('Test Start')
        test_loader = DataLoader(self.test_dataset, batch_size=self.train_conf.batch_size)

        best_gl_snapshot = load_model(self.best_gl_model_dir)
        best_fe_snapshot = load_model(self.best_fe_model_dir)

        self.graph_learning_module.load_state_dict(best_gl_snapshot)
        self.graph_forecasting_module.load_state_dict(best_fe_snapshot)

        if self.use_gpu and (self.device != 'cpu'):
            self.graph_learning_module = self.graph_learning_module.to(device=self.device)
            self.graph_forecasting_module = self.graph_forecasting_module.to(device=self.device)

        # ===================== validation ============================ #
        self.graph_learning_module.eval()
        self.graph_forecasting_module.eval()

        test_loss = []
        results = defaultdict(list)
        output = []
        target = []
        for data_batch in tqdm(test_loader):
            batch_size = data_batch.x.shape[0] // self.node_nums
            if self.use_gpu and (self.device != 'cpu'):
                data_batch = data_batch.to(device=self.device)
            with torch.no_grad():
                adj = self.graph_learning(self.entire_inputs, self.fully_connected_edge_index)
                batch_adj_matrix, adj_matrix = self._gumbel_softmax_structure_sampling(adj, batch_size)

                outputs = self.graph_forecasting_module(data_batch.x, data_batch.y, batch_adj_matrix)

            loss = F.poisson_nll_loss(outputs, data_batch.y, log_input=True)

            test_loss += [float(loss.data.cpu().numpy())]
            output += [outputs.cpu().numpy()]
            target += [data_batch.y]

        test_loss = np.stack(test_loss).mean()

        results['test_loss'] += [test_loss]
        results['adj_matrix'] = adj_matrix.cpu().numpy()
        results['prediction'] = output
        results['target'] = target

        logger.info("Avg. Test Loss = {:.6}".format(test_loss, 0))

        pickle.dump(results, open(os.path.join(self.config.exp_dir, 'test_result.pickle'), 'wb'))
