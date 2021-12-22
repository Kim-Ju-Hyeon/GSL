import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

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

        self.model = GTS_Model(self.config)

        self.CE_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.loss_ratio = config.loss_ratio

        if self.use_gpu and (self.device != 'cpu'):
            self.model = self.model.to(device=self.device)

        dataset = pickle.load(open('./data/CCN/monkeydata_with_class.pickle', 'rb'))
        train_data, valid_data, test_data = self.split_train_valid_test(dataset)

        self.train_dataset = self.make_dataset(train_data, edge_index=self.fully_connected_edge_index)
        self.valid_dataset = self.make_dataset(valid_data, edge_index=self.fully_connected_edge_index)
        self.test_dataset = self.make_dataset(test_data, edge_index=self.fully_connected_edge_index)

    def split_train_valid_test(self, dataset):
        train_x = []
        train_y = []

        valid_x = []
        valid_y = []

        test_x = []
        test_y = []

        for angle in range(8):
            data = dataset[angle, :, :, :]
            X_train, X_test, y_train, y_test = train_test_split(data[:, :98, :], data[:, 98:, :], test_size=0.2)
            X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5)

            train_x += X_train
            train_y += y_train

            valid_x += X_valid
            valid_y += y_valid

            test_x += X_test
            test_y += y_test

        train_x = torch.stack(train_x, dim=0)
        train_y = torch.stack(train_y, dim=0)

        valid_x = torch.stack(valid_x, dim=0)
        valid_y = torch.stack(valid_y, dim=0)

        test_x = torch.stack(test_x, dim=0)
        test_y = torch.stack(test_y, dim=0)

        return [train_x, train_y], [valid_x, valid_y], [test_x, test_y]

    def make_dataset(self, dataset, edge_index):
        data_len = dataset[0].shape[0]
        data_list = []
        for trial in range(data_len):
            data_item = Data(x=torch.FloatTensor(dataset[0][trial, :, :]), edge_index=edge_index,
                             y=torch.FloatTensor(dataset[1][trial, :, :]))
            data_list.append(data_item)

        return data_list

    def compute_loss(self, reg_prediction, class_prediction, target, loss_ratio):
        cost_x = self.mse_loss(reg_prediction[0,:], target[0,:])
        cost_y = self.mse_loss(reg_prediction[1, :], target[1, :])
        cost_z = self.mse_loss(reg_prediction[2, :], target[2, :])

        reg_cost = (cost_x+cost_y+cost_z) / 3 / target.shape[1]

        classification_cost = self.CE_loss(class_prediction.reshape(1,8), target[-1,:1].to(torch.long))

        loss = loss_ratio*reg_cost + (1-loss_ratio)*classification_cost
        return loss


    def train(self):
        print('Train Start')
        train_loader = DataLoader(self.train_dataset, batch_size=self.train_conf.batch_size, shuffle=True)
        valid_loader = DataLoader(self.valid_dataset, batch_size=self.train_conf.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        results = defaultdict(list)
        best_val_loss = np.inf

        # ========================= Training Loop ============================= #
        for epoch in range(self.train_conf.epoch):
            # ====================== training ============================= #
            self.model.train()
            train_loss = []

            for data_batch in tqdm(train_loader):
                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)

                _, outputs, angle = self.model(inputs=data_batch.x, edge_index=data_batch.edge_index)

                loss = self.compute_loss(outputs, angle, data_batch.y, self.loss_ratio)
                # backward pass (accumulates gradients).
                loss.backward()
                # performs a single update step.
                optimizer.step()
                optimizer.zero_grad()

                train_loss += [float(loss.data.cpu().numpy())]

            train_loss = np.stack(train_loss).mean()
            results['train_loss'] += [train_loss]

            # ===================== validation ============================ #
            self.model.eval()

            val_loss = []
            total = 0
            correct = 0
            for data_batch in tqdm(valid_loader):
                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)

                with torch.no_grad():
                    _, outputs, angle = self.model(inputs=data_batch.x, edge_index=data_batch.edge_index)

                loss = self.compute_loss(outputs, angle, data_batch.y, self.loss_ratio)

                val_loss += [float(loss.data.cpu().numpy())]

                _, pred = torch.max(angle.data.reshape(1, 8), 1)
                correct += (pred == data_batch.y[-1, 0]).sum().item()
                total += 1

            val_loss = np.stack(val_loss).mean()
            results['val_loss'] += [val_loss]

            results['valid_classification_acc'] += [correct / total]

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
            if self.use_gpu and (self.device != 'cpu'):
                data_batch = data_batch.to(device=self.device)

            with torch.no_grad():
                adj_matrix, out, angle = self.model(inputs=data_batch.x, edge_index=data_batch.edge_index)

            loss = self.compute_loss(out, angle, data_batch.y, self.loss_ratio)

            test_loss += [float(loss.data.cpu().numpy())]

            _, pred = torch.max(angle.data.reshape(1, 8), 1)
            correct += (pred == data_batch.y[-1, 0]).sum().item()
            total += 1


        test_loss = np.stack(test_loss).mean()

        results['test_loss'] += [test_loss]
        results['adj_matrix'] = adj_matrix.cpu().numpy()
        results['test_classification_acc'] += [correct / total]


        logger.info("Avg. Test Loss = {:.6}".format(test_loss, 0))

        pickle.dump(results, open(os.path.join(self.config.exp_dir, 'test_result.pickle'), 'wb'))
