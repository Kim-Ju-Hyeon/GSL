import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from models.GTS.GTS_model import GTS_Model
from utils.utils import build_fully_connected_edge_idx
from dataset.torch_geometric_spike_dataset import *


class GTS_Runner(object):
    def __init__(self, config):
        self.config = config
        self.seed = config.seed
        self.use_gpu = config.use_gpu
        self.device = config.device

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

        for epoch in range(self.train_conf.epoch):
            self.model.train()
            for data_batch in train_loader:
                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)

                _, _, loss = self.model(inputs=data_batch.x, targets=data_batch.y[:, :, 0], entire_inputs=self.entire_inputs,
                           edge_index=self.fully_connected_edge_index)

                # backward pass (accumulates gradients).
                loss.backward()
                # performs a single update step.
                optimizer.step()

    def test(self):
        pass
