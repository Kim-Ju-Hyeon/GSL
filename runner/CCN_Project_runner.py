import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models.GTS.GTS_model_for_CNN_Project import GTS_Model
from utils.utils import build_fully_connected_edge_idx
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

        self.best_model_dir = os.path.join(self.model_save, 'best.pth')
        self.ck_dir = os.path.join(self.exp_dir, 'training.ck')

        self.dataset_conf = config.dataset
        self.train_conf = config.train

        self.fully_connected_edge_index = build_fully_connected_edge_idx(self.config.nodes_num)
        if self.use_gpu and (self.device != 'cpu'):
            self.fully_connected_edge_index = self.fully_connected_edge_index.to(device=self.device)

        if self.dataset_conf.name == 'monkey_data':
            print("Load Spike Dataset")
            spike = pickle.load(open('./data/CCN/monkey_spike.pickle', 'rb'))

            print("Split Spike Train, Valid, Test Dataset")
            self.train_entire = spike[:, :80, :, :]
            self.valid_entire = spike[:, 80:90, :, :]
            self.test_entire = spike[:, 90:, :, :]

            self.train_dataset = self.prepare_dataset(self.train_entire)
            self.valid_dataset = self.prepare_dataset(self.valid_entire)
            self.test_dataset = self.prepare_dataset(self.test_entire)

            self.train_entire = self.train_entire.reshape(-1, 98, 742)
            self.valid_entire = self.valid_entire.reshape(-1, 98, 742)
            self.test_entire = self.test_entire.reshape(-1, 98, 742)

        else:
            raise ValueError("Non-supported dataset!")

        self.model = GTS_Model(self.config)

        if self.use_gpu and (self.device != 'cpu'):
            self.model = self.model.to(device=self.device)

    def prepare_dataset(self, dataset):
        valid_sampling_locations = []
        valid_sampling_locations += [
            (self.dataset_conf.window_size + i)
            for i in range(self.dataset_conf.total_time_length - self.dataset_conf.window_size + 1)
            if (i % self.dataset_conf.slide) == 0
        ]

        data_list = []
        idx = 0
        for target in range(8):
            for trial in range(dataset.shape[1]):

                spike_data = []
                for start_idx in valid_sampling_locations:
                    one_cell_inputs = dataset[target, trial, :,
                                      (start_idx - self.dataset_conf.window_size):start_idx]
                    spike_data.append(one_cell_inputs)

                spike_inputs = np.stack(spike_data, axis=1)
                data_item = Data(x=torch.FloatTensor(spike_inputs), edge_index=None, y=[target, idx])
                data_list.append(data_item)
                idx += 1

        return data_list

    def train(self):
        print('Train Start')
        train_loader = DataLoader(self.train_dataset, batch_size=self.train_conf.batch_size, shuffle=True)
        valid_loader = DataLoader(self.valid_dataset, batch_size=self.train_conf.batch_size, shuffle=True)

        # create optimizer
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum,
                weight_decay=self.train_conf.wd)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr,
                weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_conf.lr_decay_steps,
            gamma=self.train_conf.lr_decay)

        # reset gradient
        optimizer.zero_grad()

        iter_count = 0
        results = defaultdict(list)
        best_val_loss = np.inf

        # ========================= Training Loop ============================= #
        for epoch in range(self.train_conf.epoch):
            # ====================== training ============================= #
            self.model.train()
            train_loss = []

            iter = 0
            for data_batch in tqdm(train_loader):
                target = np.stack(data_batch.y, axis=0)[:, 0]
                target = torch.Tensor(target).to(torch.int64)
                # target = F.one_hot(torch.Tensor(target).to(torch.int64), num_classes=8)
                # target = torch.FloatTensor(target)

                indexs = np.stack(data_batch.y, axis=0)[:, 1]

                gl = [self.train_entire[i] for i in indexs]
                entire_input = torch.FloatTensor(np.array(gl))

                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)
                    target = target.to(device=self.device)
                    entire_input = entire_input.to(device=self.device)

                # print(f'data_batch.x: {data_batch.x.shape}')
                # print(f'data_batch.x: {target.shape}')
                # print(f'entire_inputs: {entire_input.shape}')
                # print(f'fully_connected_edge_index.x: {self.fully_connected_edge_index.shape}')

                _, _, loss = self.model(inputs=data_batch.x, targets=target,
                                        entire_inputs=entire_input,
                                        edge_index=self.fully_connected_edge_index)

                # backward pass (accumulates gradients).
                loss.backward()
                # performs a single update step.
                optimizer.step()

                train_loss += [float(loss.data.cpu().numpy())]
                iter += 1

                # display loss
                if (iter + 1) % 100 == 0:
                    logger.info(
                        "Train Loss @ epoch {} iteration {} = {}".format(epoch + 1, iter + 1, float(loss.data.cpu().numpy())))

            lr_scheduler.step()

            train_loss = np.stack(train_loss).mean()
            results['train_loss'] += [train_loss]

            # ===================== validation ============================ #
            self.model.eval()

            val_loss = []
            total = 0
            correct = 0
            for data_batch in tqdm(valid_loader):
                target = np.stack(data_batch.y, axis=0)[:, 0]
                target = torch.Tensor(target).to(torch.int64)
                # target = F.one_hot(torch.Tensor(target).to(torch.int64), num_classes=8)

                indexs = np.stack(data_batch.y, axis=0)[:, 1]
                gl = [self.train_entire[i] for i in indexs]
                entire_input = torch.FloatTensor(np.array(gl))

                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)
                    target = target.to(device=self.device)
                    entire_input = entire_input.to(device=self.device)

                with torch.no_grad():
                    adj_matrix, out, loss = self.model(inputs=data_batch.x, targets=target,
                                                       entire_inputs=entire_input,
                                                       edge_index=self.fully_connected_edge_index)
                val_loss += [float(loss.data.cpu().numpy())]

                _, pred = torch.max(out.data, 1)
                total += target.size(0)
                correct += (pred == target).sum().item()

            val_loss = np.stack(val_loss).mean()

            results['val_loss'] += [val_loss]
            results['val_adj_matirix'] += [adj_matrix]
            results['val_acc'] += [correct / total]

            logger.info("Avg. Validation Loss = {:.6}".format(val_loss, 0))
            logger.info("Current Best Validation Loss = {:.6}".format(best_val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_dir)

            model_snapshot(epoch, self.model, optimizer, best_val_loss, self.ck_dir)

        pickle.dump(results, open(os.path.join(self.config.exp_dir, 'training_result.pickle'), 'wb'))

    def test(self):
        print('Test Start')
        test_loader = DataLoader(self.test_dataset, batch_size=self.train_conf.batch_size, shuffle=True)

        self.best_model = GTS_Model(self.config)
        best_snapshot = load_model(self.best_model_dir)

        self.best_model.load_state_dict(best_snapshot)

        if self.use_gpu and (self.device != 'cpu'):
            self.best_model = self.best_model.to(device=self.device)

        # ===================== Test ============================ #
        self.best_model.eval()

        test_loss = []
        results = defaultdict(list)
        total = 0
        correct = 0

        for data_batch in tqdm(test_loader):
            target = np.stack(data_batch.y, axis=0)[:, 0]
            target = torch.Tensor(target).to(torch.int64)
            # target = F.one_hot(torch.Tensor(target).to(torch.int64), num_classes=8)

            indexs = np.stack(data_batch.y, axis=0)[:, 1]
            gl = [self.train_entire[i] for i in indexs]
            entire_input = torch.FloatTensor(np.array(gl))

            if self.use_gpu and (self.device != 'cpu'):
                data_batch = data_batch.to(device=self.device)
                target = target.to(device=self.device)
                entire_input = entire_input.to(device=self.device)

            with torch.no_grad():
                adj_matrix, out, loss = self.best_model(inputs=data_batch.x, targets=target,
                                                   entire_inputs=entire_input,
                                                   edge_index=self.fully_connected_edge_index)

            test_loss += [float(loss.data.cpu().numpy())]
            _, pred = torch.max(out.data, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()


        test_loss = np.stack(test_loss).mean()

        results['test_loss'] += [test_loss]
        results['adj_matrix'] = adj_matrix.cpu().numpy()
        results['test_acc'] += [correct / total]


        logger.info("Avg. Test Loss = {:.6}".format(test_loss, 0))

        pickle.dump(results, open(os.path.join(self.config.exp_dir, 'test_result.pickle'), 'wb'))
