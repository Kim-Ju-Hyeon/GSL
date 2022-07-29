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

from models.ic_pn_beats_model import IC_PN_BEATS_model
from utils.utils import build_fully_connected_edge_idx
from utils.train_helper import model_snapshot, load_model
from utils.logger import get_logger

from dataset.make_spike_datset import MakeSpikeDataset
from dataset.make_dataset_METR_PEMS import METR_PEMS_DatasetLoader
from dataset.make_dataset_ett import ETTDatasetLoader
from dataset.make_dataset_covid19 import COVID19DatasetLoader
from dataset.make_dataset_exchange import ExchangeDatasetLoader
from dataset.make_dataset_ecl import ECLDatasetLoader
from dataset.make_dataset_wth import WTHDatasetLoader
from dataset.make_dataset_traffic import TrafficDatasetLoader

from utils.score import get_score


class runner(object):
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

        self.logger = get_logger(logger_name=str(config.seed))

        if self.train_conf.loss_function == 'MAE':
            self.loss = nn.L1Loss()
        elif self.train_conf.loss_function == 'Poisson':
            self.loss = nn.PoissonNLLLoss()
        else:
            raise ValueError('Non-supported Loss Function')

        self.nodes_num = self.dataset_conf.nodes_num
        self.get_dataset()

        self.model = IC_PN_BEATS_model(self.config)

        if self.use_gpu and (self.device != 'cpu'):
            self.model = self.model.to(device=self.device)

    def get_dataset(self):
        num_timesteps_in = self.config.forecasting_module.backcast_length
        num_timesteps_out = self.config.forecasting_module.forecast_length
        batch_size = self.train_conf.batch_size
        dataset_hyperparameter = f'{num_timesteps_in}_{num_timesteps_out}_{batch_size}'

        ett_dataset_list = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2']

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
                loader = METR_PEMS_DatasetLoader(raw_data_dir=self.dataset_conf.root,
                                                 dataset_name=self.dataset_conf.name,
                                                 scaler_type=self.config.dataset.scaler_type)

            elif self.dataset_conf.name == 'ECL':
                loader = ECLDatasetLoader(raw_data_dir=self.dataset_conf.root,
                                          scaler_type=self.config.dataset.scaler_type)

            elif self.dataset_conf.name in ett_dataset_list:
                loader = ETTDatasetLoader(raw_data_dir=self.dataset_conf.root,
                                          scaler_type=self.config.dataset.scaler_type,
                                          group=self.dataset_conf.name)

            elif self.dataset_conf.name == 'COVID19':
                loader = COVID19DatasetLoader(raw_data_dir=self.dataset_conf.root,
                                              scaler_type=self.config.dataset.scaler_type)

            elif self.dataset_conf.name == 'Exchange':
                loader = ExchangeDatasetLoader(raw_data_dir=self.dataset_conf.root,
                                               scaler_type=self.config.dataset.scaler_type)

            elif self.dataset_conf.name == 'WTH':
                loader = WTHDatasetLoader(raw_data_dir=self.dataset_conf.root,
                                          scaler_type=self.config.dataset.scaler_type)

            elif self.dataset_conf.name == 'Traffic':
                loader = TrafficDatasetLoader(raw_data_dir=self.dataset_conf.root,
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

        results = defaultdict(list)
        best_val_loss = np.inf

        if self.config.train_resume:
            checkpoint = load_model(self.ck_dir)
            self.model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
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

                backcast, forecast, _ = self.model(data_batch.x, interpretability=False)
                forecast_loss = self.loss(forecast, data_batch.y)

                if self.backcast_loss:
                    backcast_loss = self.loss(backcast, data_batch.x[:, 0, :])
                    loss = 0.3 * backcast_loss + 0.7 * forecast_loss
                else:
                    loss = forecast_loss

                # backward pass (accumulates gradients).
                loss.backward()

                # performs a single update step.
                optimizer.step()
                optimizer.zero_grad()

                train_loss += [float(forecast_loss.data.cpu().numpy())]

                # display loss
                if (i + 1) % 500 == 0:
                    self.logger.info(
                        "Train Loss @ epoch {} iteration {} = {}".format(epoch + 1, i + 1,
                                                                         float(forecast_loss.data.cpu().numpy())))

            train_loss = np.stack(train_loss).mean()
            results['train_loss'] += [train_loss]

            # ===================== validation ============================ #
            self.model.eval()

            val_loss = []
            for data_batch in tqdm(self.valid_dataset):

                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)
                with torch.no_grad():
                    _, forecast, _ = self.model(data_batch.x, interpretability=False)

                forecast_loss = self.loss(forecast, data_batch.y)
                val_loss += [float(forecast_loss.data.cpu().numpy())]

            val_loss = np.stack(val_loss).mean()

            results['val_loss'] += [val_loss]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_dir)

            self.logger.info("Epoch {} Avg. Validation Loss = {:.6}".format(epoch + 1, val_loss, 0))
            self.logger.info("Current Best Validation Loss = {:.6}".format(best_val_loss))

            model_snapshot(epoch=epoch, model=self.model, optimizer=optimizer, scheduler=None,
                           best_valid_loss=best_val_loss, exp_dir=self.ck_dir)

        pickle.dump(results, open(os.path.join(self.config.exp_sub_dir, 'training_result.pickle'), 'wb'))

    def test(self):
        self.config.train.batch_size = 1

        self.best_model = IC_PN_BEATS_model(self.config)
        best_snapshot = load_model(self.best_model_dir)

        self.best_model.load_state_dict(best_snapshot)

        if self.use_gpu and (self.device != 'cpu'):
            self.best_model = self.best_model.to(device=self.device)

        # ===================== validation ============================ #
        self.best_model.eval()

        test_loss = []
        results = defaultdict()
        target = []
        inputs = []
        forecast_list = []
        backcast_list = []

        per_trend_backcast = []
        per_trend_forecast = []
        per_seasonality_backcast = []
        per_seasonality_forecast = []
        singual_backcast = []
        singual_forecast = []

        attention_matrix = []

        for data_batch in tqdm(self.test_dataset):
            if self.use_gpu and (self.device != 'cpu'):
                data_batch = data_batch.to(device=self.device)

            with torch.no_grad():
                _backcast_output, _forecast_output, outputs = self.model(data_batch.x, interpretability=True)

            loss = self.loss(_forecast_output, data_batch.y)

            test_loss += [float(loss.data.cpu().detach().numpy())]
            target += [data_batch.y.cpu().detach().numpy()]
            inputs += [data_batch.x.cpu().detach().numpy()]
            forecast_list += [_forecast_output.cpu().detach().numpy()]
            backcast_list += [_backcast_output.cpu().detach().numpy()]

            per_trend_backcast += [outputs['per_trend_backcast']]
            per_trend_forecast += [outputs['per_trend_forecast']]
            per_seasonality_backcast += [outputs['per_seasonality_backcast']]
            per_seasonality_forecast += [outputs['per_seasonality_forecast']]
            singual_backcast += [outputs['singual_backcast']]
            singual_forecast += [outputs['singual_forecast']]
            attention_matrix += [outputs['attention_matrix']]

        results['per_trend_backcast'] = np.stack(per_trend_backcast, axis=0)
        results['per_trend_forecast'] = np.stack(per_trend_forecast, axis=0)
        results['per_seasonality_backcast'] = np.stack(per_seasonality_backcast, axis=0)
        results['per_seasonality_forecast'] = np.stack(per_seasonality_forecast, axis=0)
        results['singual_backcast'] = np.stack(singual_backcast, axis=0)
        results['singual_forecast'] = np.stack(singual_forecast, axis=0)

        results['test_loss'] = np.stack(test_loss).mean()
        results['forecast'] = np.stack(forecast_list, axis=0)
        results['backcast'] = np.stack(backcast_list, axis=0)
        results['target'] = np.stack(target, axis=0)
        results['inputs'] = np.stack(inputs, axis=0)
        results['attention_matrix'] = np.stack(attention_matrix, axis=0)

        scaled_score = get_score(results['target'].transpose((1, 0, 2)).reshape(self.nodes_num, -1),
                                 results['forecast'].transpose((1, 0, 2)).reshape(self.nodes_num, -1), scaler=None)

        inv_scaled_score = get_score(results['target'].transpose((1, 0, 2)).reshape(self.nodes_num, -1),
                                     results['forecast'].transpose((1, 0, 2)).reshape(self.nodes_num, -1),
                                     scaler=self.scaler)

        results['score'] = {'scaled_score': scaled_score,
                            'inv_scaled_score': inv_scaled_score}

        self.logger.info("Avg. Test Loss = {:.6}".format(test_loss, 0))
        self.logger.info(f"Avg. MAE = {scaled_score['MAE']:.6}")
        self.logger.info(f"Avg. MAPE = {scaled_score['MAPE']:.6}")
        self.logger.info(f"Avg. RMSE = {scaled_score['RMSE']:.6}")
        self.logger.info(f"Avg. MSE = {scaled_score['MSE']:.6}")

        pickle.dump(results, open(os.path.join(self.config.exp_sub_dir, 'test_result.pickle'), 'wb'))
