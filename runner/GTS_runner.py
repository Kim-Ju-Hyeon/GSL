import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn

from collections import defaultdict
import torch.optim as optim
from torch_geometric.loader import DataLoader

from models.GTS.GTS_model import GTS_Model
from utils.utils import build_fully_connected_edge_idx
from dataset.torch_geometric_spike_dataset import *
from utils.train_helper import model_snapshot, load_model


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

        self.fully_connected_edge_index = build_fully_connected_edge_idx(self.config.num_nodes)

        if self.dataset_conf.name == 'spike_lambda':
            spike = pickle.load(open('./data/LNP_spk_all.pickle', 'rb'))
            # lam = pickle.load(open('./data/LNP_lam_all.pickle', 'rb'))

            self.entire_inputs = torch.FloatTensor(spike)

            self.train_dataset = Train_Dataset(root=self.dataset_conf.root)
            self.valid_dataset = Validation_Dataset(root=self.dataset_conf.root)
            self.test_dataset = Test_Dataset(root=self.dataset_conf.root)

        else:
            raise ValueError("Non-supported dataset!")

        self.model = GTS_Model(self.config)

        if self.use_gpu and (self.device != 'cpu'):
            self.model = self.model.to(device=self.device)

    def train(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.train_conf.batch_size)
        valid_loader = DataLoader(self.valid_dataset, batch_size=self.train_conf.batch_size)

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
            for data_batch in train_loader:
                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)

                _, _, loss = self.model(inputs=data_batch.x, targets=data_batch.y[:, :, 0],
                                        entire_inputs=self.entire_inputs,
                                        edge_index=self.fully_connected_edge_index)

                # backward pass (accumulates gradients).
                loss.backward()
                # performs a single update step.
                optimizer.step()

                train_loss += [float(loss.data.cpu().numpy())]

                # display loss
                if (iter_count + 1) % 10 == 0:
                    print(
                        "Train Loss @ epoch {} iteration {} = {}".format(epoch + 1, iter_count + 1, train_loss))

            train_loss = np.stack(train_loss).mean()
            results['train_loss'] += [train_loss]

            # ===================== validation ============================ #
            self.model.eval()

            val_loss = []
            for data_batch in valid_loader:
                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)
                with torch.no_grad():
                    adj_matrix, _, loss = self.model(inputs=data_batch.x, targets=data_batch.y[:, :, 0],
                                                     entire_inputs=self.entire_inputs,
                                                     edge_index=self.fully_connected_edge_index)
                val_loss += [float(loss.data.cpu().numpy())]

            val_loss = np.stack(val_loss).mean()
            results['val_loss'] += [val_loss]
            results['val_adj_matirix'] += [adj_matrix]

            print("Avg. Validation Loss = {:.6}".format(val_loss, 0))
            print("Current Best Validation Loss = {:.6}".format(best_val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_dir)

            model_snapshot(epoch, self.model, optimizer, best_val_loss, self.ck_dir)

    def test(self):
        pass
