import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn

from collections import defaultdict
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader

from models.model import My_Model
from utils.utils import build_fully_connected_edge_idx
from dataset.make_spike_datset import MakeSpikeDataset
from utils.train_helper import model_snapshot, load_model
from utils.logger import get_logger


from dataset.temporal_graph_dataset import Temporal_Graph_Signal

from utils.score import get_score
import yaml
from utils.train_helper import edict2dict

def get_dataset_length(_dataset):
    length = 0
    for _ in _dataset:
        length += 1
    return length


class Runner(object):
    def __init__(self, config):
        self.get_dataset(config)
        self.config = config
        self.exp_dir = config.exp_dir
        self.model_save = config.model_save
        self.logger = get_logger(logger_name=str(config.seed))
        self.seed = config.seed
        self.use_gpu = config.use_gpu
        self.device = config.device
        self.backcast_loss = config.train.backcast_loss

        self.best_model_dir = os.path.join(self.model_save, 'best.pth')
        self.ck_dir = os.path.join(self.model_save, 'training.ck')

        self.train_conf = config.train
        self.dataset_conf = config.dataset
        self.nodes_num = config.dataset.nodes_num


        if self.train_conf.loss_function == 'MAE':
            self.loss = nn.L1Loss()
        elif self.train_conf.loss_function == 'Poisson':
            self.loss = nn.PoissonNLLLoss()
        else:
            raise ValueError('Non-supported Loss Function')


        self.model = My_Model(self.config)

        if self.use_gpu and (self.device != 'cpu'):
            self.model = self.model.to(device=self.device)

    def get_dataset(self, config):
        loader = Temporal_Graph_Signal(config.dataset.name, config.dataset.scaler_type, config.dataset.univariate)
        config.dataset.nodes_num = loader.nodes_num
        config.dataset.node_features = loader.node_features
        config.dataset.root = loader.path
        config.dataset.freq = loader.freq
        save_name = os.path.join(config.exp_sub_dir, 'config.yaml')
        yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

        loader.preprocess_dataset()
        self.train_dataset, self.valid_dataset, self.test_dataset = loader.get_dataset(
            num_timesteps_in=config.forecasting_module.backcast_length,
            num_timesteps_out=config.forecasting_module.forecast_length,
            batch_size=config.train.batch_size)

        self.scaler = loader.get_scaler()
    def train(self):
        # create optimizer
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr,
                weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.train_conf.T_0, T_mult=self.train_conf.T_mult)

        results = defaultdict(list)
        best_val_loss = np.inf

        if self.config.train_resume:
            checkpoint = load_model(self.ck_dir)
            self.model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            best_val_loss = checkpoint['best_valid_loss']
            self.train_conf.epoch -= checkpoint['epoch']

        # ========================= Training Loop ============================= #
        for epoch in range(self.train_conf.epoch):
            # ====================== training ============================= #
            self.model.train()

            train_loss = []

            for i, data_batch in enumerate(tqdm(self.train_dataset)):

                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)

                _, outputs, _ = self.model(data_batch.x, data_batch.y, self.entire_inputs, self.init_edge_index,
                                           interpretability=False)
                if type(outputs) == defaultdict:
                    forecast = outputs['forecast']
                    forecast_target = data_batch.y
                    loss = self.loss(forecast, forecast_target)

                    if self.backcast_loss:
                        backcast = outputs['backcast']
                        backcast_target = data_batch.x[:, 0, :]
                        backcast_loss = self.loss(backcast, backcast_target)
                        loss = backcast_loss + loss

                    outputs = defaultdict(list)

                else:
                    forecast = outputs
                    target = data_batch.y

                    loss = self.loss(forecast, target)

                # backward pass (accumulates gradients).
                loss.backward()

                # performs a single update step.
                optimizer.step()
                optimizer.zero_grad()
                # temp = int(epoch + i / length)
                # scheduler.step(temp)

                train_loss += [float(loss.data.cpu().numpy())]

                # display loss
                if (i + 1) % 500 == 0:
                    self.logger.info(
                        "Train Loss @ epoch {} iteration {} = {}".format(epoch + 1, i + 1,
                                                                         float(loss.data.cpu().numpy())))

            train_loss = np.stack(train_loss).mean()
            results['train_loss'] += [train_loss]

            # ===================== validation ============================ #
            self.model.eval()

            val_loss = []
            for data_batch in tqdm(self.valid_dataset):

                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)
                with torch.no_grad():
                    adj_matrix, outputs, attention_matrix = self.model(data_batch.x, data_batch.y, self.entire_inputs,
                                                                       self.init_edge_index, interpretability=False)
                if type(outputs) == defaultdict:
                    forecast = outputs['forecast']
                else:
                    forecast = outputs

                loss = self.loss(forecast, data_batch.y)

                val_loss += [float(loss.data.cpu().numpy())]

            val_loss = np.stack(val_loss).mean()

            results['val_loss'] += [val_loss]
            results['attention_matrix'] = attention_matrix

            if type(adj_matrix) == dict:
                results['val_adj_matirix'] += adj_matrix
            else:
                results['val_adj_matirix'] += [adj_matrix.detach().cpu()]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_dir)

            self.logger.info("Epoch {} Avg. Validation Loss = {:.6}".format(epoch + 1, val_loss, 0))
            self.logger.info("Current Best Validation Loss = {:.6}".format(best_val_loss))

            # model_snapshot(epoch, self.model, optimizer, scheduler, best_val_loss, self.ck_dir)
            model_snapshot(epoch=epoch, model=self.model, optimizer=optimizer, scheduler=None,
                           best_valid_loss=best_val_loss, exp_dir=self.ck_dir)

        pickle.dump(results, open(os.path.join(self.config.exp_sub_dir, 'training_result.pickle'), 'wb'))

    def test(self):
        self.best_model = My_Model(self.config)
        best_snapshot = load_model(self.best_model_dir)

        self.best_model.load_state_dict(best_snapshot)
        self.best_model.batch_size = 1

        if self.use_gpu and (self.device != 'cpu'):
            self.best_model = self.best_model.to(device=self.device)

        # ===================== validation ============================ #
        self.best_model.eval()

        test_loss = []
        results = defaultdict()
        output = []
        target = []
        inputs = []
        backcast = []

        for data_batch in tqdm(self.test_dataset):
            if self.use_gpu and (self.device != 'cpu'):
                data_batch = data_batch.to(device=self.device)

            with torch.no_grad():
                adj_matrix, outputs, attention_matrix = self.best_model(data_batch.x, data_batch.y, self.entire_inputs,
                                                                        self.init_edge_index, interpretability=True)

                if type(outputs) == defaultdict:
                    forecast = outputs['forecast']
                else:
                    forecast = outputs

                loss = self.loss(forecast, data_batch.y)

                test_loss += [float(loss.data.cpu().numpy())]
                output += [forecast.cpu().numpy()]
                target += [data_batch.y.cpu().numpy()]
                inputs += [data_batch.x.cpu().numpy()]
                backcast += [outputs['backcast'].cpu()]

                if self.config.forecasting_module.name == 'n_beats':
                    stack_per_backcast = []
                    stack_per_forecast = []
                    block_per_backcast = []
                    block_per_forecast = []

                    stack_per_backcast += [outputs['stack_per_backcast']]
                    stack_per_forecast += [outputs['stack_per_forecast']]
                    block_per_backcast += [outputs['block_per_backcast']]
                    block_per_forecast += [outputs['block_per_forecast']]

                    stack_per_backcast = np.stack(stack_per_backcast)
                    stack_per_forecast = np.stack(stack_per_forecast)
                    block_per_backcast = np.stack(block_per_backcast)
                    block_per_forecast = np.stack(block_per_forecast)

                    results['stack_per_backcast'] = stack_per_backcast
                    results['stack_per_forecast'] = stack_per_forecast
                    results['block_per_backcast'] = block_per_backcast
                    results['block_per_forecast'] = block_per_forecast

                elif self.config.forecasting_module.name == 'pn_beats':
                    per_trend_backcast = np.stack(outputs['per_trend_backcast'], axis=0)
                    per_trend_forecast = np.stack(outputs['per_trend_forecast'], axis=0)
                    per_seasonality_backcast = np.stack(outputs['per_seasonality_backcast'], axis=0)
                    per_seasonality_forecast = np.stack(outputs['per_seasonality_forecast'], axis=0)
                    singual_backcast = np.stack(outputs['singual_backcast'], axis=0)
                    singual_forecast = np.stack(outputs['singual_forecast'], axis=0)

                    results['per_trend_backcast'] = per_trend_backcast
                    results['per_trend_forecast'] = per_trend_forecast
                    results['per_seasonality_backcast'] = per_seasonality_backcast
                    results['per_seasonality_forecast'] = per_seasonality_forecast
                    results['singual_backcast'] = singual_backcast
                    results['singual_forecast'] = singual_forecast

        test_loss = np.stack(test_loss).mean()
        output = np.stack(output)
        target = np.stack(target)
        inputs = np.stack(inputs)
        backcast = np.stack(backcast)

        scaled_score = get_score(target.transpose((1, 0, 2)).reshape(self.nodes_num, -1),
                          output.transpose((1, 0, 2)).reshape(self.nodes_num, -1), scaler=None)

        inv_scaled_score = get_score(target.transpose((1, 0, 2)).reshape(self.nodes_num, -1),
                          output.transpose((1, 0, 2)).reshape(self.nodes_num, -1), scaler=self.scaler)

        results['test_loss'] = test_loss
        results['score'] = {'scaled_score': scaled_score,
                            'inv_scaled_score': inv_scaled_score}
        results['adj_matrix'] = adj_matrix.cpu().numpy()
        results['prediction'] = output
        results['target'] = target
        results['Inputs'] = inputs
        results['attention_matrix'] = attention_matrix.cpu().numpy()
        results['backcast'] = backcast

        self.logger.info("Avg. Test Loss = {:.6}".format(test_loss, 0))
        self.logger.info("Avg. MAPE = {:.6}".format(scaled_score['MAPE'], 0))

        pickle.dump(results, open(os.path.join(self.config.exp_sub_dir, 'test_result.pickle'), 'wb'))
