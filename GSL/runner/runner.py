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
from dataset.make_traffic_dataset import TrafficDatasetLoader
from dataset.ecl import ECLDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

from utils.score import get_score

logger = get_logger('exp_logger')


def get_dataset_length(_dataset):
    length = 0
    for _ in _dataset:
        length += 1
    return length


class Runner(object):
    def __init__(self, config):
        self.config = config
        self.exp_dir = config.exp_dir
        self.model_save = config.model_save
        self.seed = config.seed
        self.use_gpu = config.use_gpu
        self.device = config.device
        self.backcast_loss = config.train.backcast_loss

        self.best_model_dir = os.path.join(self.model_save, 'best.pth')
        self.ck_dir = os.path.join(self.model_save, 'training.ck')

        self.dataset_conf = config.dataset
        self.train_conf = config.train

        if self.train_conf.loss_function == 'MAE':
            self.loss = nn.L1Loss()
        elif self.train_conf.loss_function == 'Poisson':
            self.loss = nn.PoissonNLLLoss()
        else:
            raise ValueError('Non-supported Loss Function')

        self.nodes_num = self.dataset_conf.nodes_num
        self.initial_edge_index = config.graph_learning.initial_edge_index
        if self.initial_edge_index == 'Fully Connected':
            self.init_edge_index = build_fully_connected_edge_idx(self.nodes_num)
        else:
            raise ValueError("Non-supported Edge Index!")

        self.get_dataset()

        self.model = My_Model(self.config)

        if self.use_gpu and (self.device != 'cpu'):
            self.model = self.model.to(device=self.device)
            self.init_edge_index = self.init_edge_index.to(device=self.device)
            self.entire_inputs = self.entire_inputs.to(device=self.device)

    def get_dataset(self):
        num_timesteps_in = self.config.forecasting_module.backcast_length
        num_timesteps_out = self.config.forecasting_module.forecast_length
        batch_size = self.train_conf.batch_size
        dataset_hyperparameter = f'{num_timesteps_in}_{num_timesteps_out}_{batch_size}'

        if os.path.exists(os.path.join(self.dataset_conf.root, f'temporal_signal_{dataset_hyperparameter}.pickle')):
            temporal_signal = pickle.load(
                open(os.path.join(self.dataset_conf.root, f'temporal_signal_{dataset_hyperparameter}.pickle'), 'rb'))
            self.train_dataset = temporal_signal['train']
            self.valid_dataset = temporal_signal['validation']
            self.test_dataset = temporal_signal['test']
            self.entire_inputs = temporal_signal['entire_inputs'][:, :, :self.dataset_conf.graph_learning_length]
            self.scaler = temporal_signal['scaler']

        else:
            if self.dataset_conf.name == 'spike_lambda_bin100':
                spike = pickle.load(open('./data/spk_bin_n100.pickle', 'rb'))

                self.entire_inputs = torch.FloatTensor(spike[:, :self.dataset_conf.graph_learning_length])

                dataset_maker = MakeSpikeDataset(self.config)
                total_dataset = dataset_maker.make()

                self.train_dataset = DataLoader(total_dataset['train'], batch_size=self.train_conf.batch_size)
                self.valid_dataset = DataLoader(total_dataset['valid'], batch_size=self.train_conf.batch_size)
                self.test_dataset = DataLoader(total_dataset['test'], batch_size=self.train_conf.batch_size)

            elif (self.dataset_conf.name == 'METR-LA') or (self.dataset_conf.name == 'PEMS-BAY'):
                loader = TrafficDatasetLoader(raw_data_dir=self.dataset_conf.root, dataset_name=self.dataset_conf.name,
                                              scaler_type=self.config.dataset.scaler_type)

            elif self.dataset_conf.name == 'ECL':
                loader = ECLDatasetLoader(raw_data_dir=self.dataset_conf.root,
                                          scaler_type=self.config.dataset.scaler_type)
            else:
                raise ValueError("Non-supported dataset!")

            self.train_dataset, self.valid_dataset, self.test_dataset, self.entire_inputs = loader.get_dataset(
                num_timesteps_in=self.config.forecasting_module.backcast_length,
                num_timesteps_out=self.config.forecasting_module.forecast_length,
                batch_size=self.train_conf.batch_size)

            self.entire_inputs = self.entire_inputs[:, :, :self.dataset_conf.graph_learning_length]
            self.scaler = loader.get_scaler()

            temporal_signal = {'train': self.train_dataset,
                               'validation': self.valid_dataset,
                               'test': self.test_dataset,
                               'entire_inputs': self.entire_inputs,
                               'scaler': self.scaler}

            pickle.dump(temporal_signal,
                        open(os.path.join(self.dataset_conf.root, f'temporal_signal_{dataset_hyperparameter}.pickle'),
                             'wb'))

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

        # length = get_dataset_length(self.train_dataset)
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
                    logger.info(
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

            logger.info("Epoch {} Avg. Validation Loss = {:.6}".format(epoch + 1, val_loss, 0))
            logger.info("Current Best Validation Loss = {:.6}".format(best_val_loss))

            # model_snapshot(epoch, self.model, optimizer, scheduler, best_val_loss, self.ck_dir)
            model_snapshot(epoch=epoch, model=self.model, optimizer=optimizer, scheduler=None,
                           best_valid_loss=best_val_loss, exp_dir=self.ck_dir)

        pickle.dump(results, open(os.path.join(self.config.exp_sub_dir, 'training_result.pickle'), 'wb'))

    def test(self):
        print(self.config.forecasting_module.thetas_dim)
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

        stack_per_backcast = []
        stack_per_forecast = []
        backcast = []

        block_per_backcast = []
        block_per_forecast = []

        for data_batch in tqdm(self.test_dataset):
            if self.use_gpu and (self.device != 'cpu'):
                data_batch = data_batch.to(device=self.device)

            # x_shape = data_batch.x.shape
            # y_shape = data_batch.y.shape
            #
            # x = data_batch.x.reshape(-1, self.nodes_num, x_shape[-2], x_shape[-1])
            # y = data_batch.y.reshape(-1, self.nodes_num, y_shape[-1])

            # for i in range(x.shape[0]):
            #     with torch.no_grad():
            #         adj_matrix, outputs, attention_matrix = self.best_model(x[i], y[i], self.entire_inputs,
            #                                                                 self.init_edge_index, interpretability=True)

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
                stack_per_backcast += [outputs['stack_per_backcast']]
                stack_per_forecast += [outputs['stack_per_forecast']]
                block_per_backcast += [outputs['block_per_backcast']]
                block_per_forecast += [outputs['block_per_forecast']]
                backcast += [outputs['backcast'].cpu()]

        test_loss = np.stack(test_loss).mean()
        output = np.stack(output)
        target = np.stack(target)
        inputs = np.stack(inputs)
        stack_per_backcast = np.stack(stack_per_backcast)
        stack_per_forecast = np.stack(stack_per_forecast)
        block_per_backcast = np.stack(block_per_backcast)
        block_per_forecast = np.stack(block_per_forecast)
        backcast = np.stack(backcast)

        score = get_score(target.transpose((1, 0, 2)).reshape(self.nodes_num, -1),
                          output.transpose((1, 0, 2)).reshape(self.nodes_num, -1), scaler=self.scaler)

        results['test_loss'] = test_loss
        results['score'] = score
        results['adj_matrix'] = adj_matrix.cpu()
        results['prediction'] = output
        results['target'] = target
        results['Inputs'] = inputs
        results['attention_matrix'] = attention_matrix
        results['stack_per_backcast'] = stack_per_backcast
        results['stack_per_forecast'] = stack_per_forecast
        results['block_per_backcast'] = block_per_backcast
        results['block_per_forecast'] = block_per_forecast
        results['backcast'] = backcast

        logger.info("Avg. Test Loss = {:.6}".format(test_loss, 0))
        logger.info("Avg. MAE = {:.6}".format(score['MAE'], 0))

        pickle.dump(results, open(os.path.join(self.config.exp_sub_dir, 'test_result.pickle'), 'wb'))
