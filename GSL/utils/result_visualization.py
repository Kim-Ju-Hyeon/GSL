from glob import glob, escape
import pickle
import yaml
from easydict import EasyDict as edict
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

from torch_geometric.utils import to_dense_adj, dense_to_sparse


def get_config_file(exp):
    config_file = glob(escape(exp + '/config.yaml'))[0]
    if len(config_file) == 0:
        config_file = glob(exp + '/config.yaml')[0]
    config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))

    return config

def get_test_dataset(config):
    dataset = config.dataset.name
    num_timesteps_in = config.forecasting_module.backcast_length
    num_timesteps_out = config.forecasting_module.forecast_length

    _dir = f'../GSL/data/{dataset}'

    ett_dataset_list = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2']
    if dataset in ett_dataset_list:
        _dir = f'../GSL/data/ETT/{dataset}'

    scaler = pickle.load(
        open(os.path.join(_dir, 'scaler.pickle'), 'rb'))

    test_dataset = pickle.load(
        open(os.path.join(_dir, 'inference.pickle'), 'rb')
    )

    indices = [
        (i, i + (num_timesteps_in + num_timesteps_out))
        for i in range(test_dataset.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
    ]

    features, target = [], []
    for i, j in indices:
        features.append((test_dataset[:, 0, i: i + num_timesteps_in]))
        target.append((test_dataset[:, 0, i + num_timesteps_in: j]))

    features = torch.Tensor(features[2])
    target = torch.Tensor(target[2])
    dataset = Data(x=features, y=target)

    node_list = [1, 5, 119, 20, 125, 48, 55, 61, 69, 267]

    return scaler, dataset, node_list


def get_exp_result_files(exp):
    config_file = glob(escape(exp + '/config.yaml'))[0]
    if len(config_file) == 0:
        config_file = glob(exp + '/config.yaml')[0]
    config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))

    train_result_dirs = glob(escape(exp + '/training_result.pickle'))
    if len(train_result_dirs) == 0:
        train_result_dirs = glob(exp + '/training_result.yaml')[0]
    train_result = pickle.load(open(train_result_dirs[0], 'rb'))

    test_result_dirs = glob(escape(exp + '/test_result.pickle'))
    if len(test_result_dirs) == 0:
        test_result_dirs = glob(exp + '/test_result.yaml')[0]
    test_result = pickle.load(open(test_result_dirs[0], 'rb'))

    return config, train_result, test_result


def get_scaler_and_test_dataset(config):
    dataset = config.dataset.name
    num_timesteps_in = config.forecasting_module.backcast_length
    num_timesteps_out = config.forecasting_module.forecast_length

    _dir = f'../GSL/data/{dataset}'

    ett_dataset_list = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2']
    if dataset in ett_dataset_list:
        _dir = f'../GSL/data/ETT/{dataset}'

    scaler = pickle.load(
        open(os.path.join(_dir, 'scaler.pickle'), 'rb'))

    test_dataset = pickle.load(
        open(os.path.join(_dir, 'inference.pickle'), 'rb')
    )

    if dataset == 'Sea_Fog':
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(test_dataset.shape[1] - (num_timesteps_in + num_timesteps_out) + 1)
            if i % 3 == 0
        ]

        features, target = [], []

        for i, j in indices:
            features.append((test_dataset[:, i: i + num_timesteps_in]))
            target.append((test_dataset[:, i + num_timesteps_in: j]))

    else:
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(test_dataset.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        features, target = [], []
        for i, j in indices:
            features.append((test_dataset[:, 0, i: i + num_timesteps_in]))
            target.append((test_dataset[:, 0, i + num_timesteps_in: j]))

    # features = torch.FloatTensor(np.array(features))
    # targets = torch.FloatTensor(np.array(target))

    return scaler, np.array(features), np.array(target)


# def visualize_inference_result(test_result):
#     target = test_result['target']
#     pred = test_result['prediction']
#
#     inputs = test_result['Inputs']
#     backcast = test_result['backcast']

# class visualize_inference_result:
#     def __init__(self, exp):
#         self.config, self.train_result, self.test_result = get_exp_result_files(exp)
#         self.scaler = get_scaler(self.config)
#
#         self._inference_result(self.test_result)
#
#     def _inference_result(self, test_result):
#         self.target = test_result['target']
#         self.pred = test_result['prediction']
#
#         self.inputs = test_result['Inputs']
#         self.backcast = test_result['backcast']
#
#         if self.config.forecasting_module.name == 'pn_beats':
#             self.per_trend_backcast = test_result['per_trend_backcast']
#             self.per_trend_forecast = test_result['per_trend_forecast']
#
#             self.per_seasonality_backcast = test_result['per_seasonality_backcast']
#             self.per_seasonality_forecast = test_result['per_seasonality_forecast']
#
#             self.singual_backcast = test_result['singual_backcast']
#             self.singual_forecast = test_result['singual_forecast']
#
#         elif self.config.forecasting_module.name == 'n_beats':
#             self.stack_per_forecast = np.stack(test_result['stack_per_forecast'])
#             self.stack_per_backcast = np.stack(test_result['stack_per_backcast'])
#
#             self.block_per_backcast = test_result['block_per_backcast']
#             self.block_per_forecast = test_result['block_per_forecast']
#
#     def visualize_val_adj_per_epoch(self):
#         edge_ = []
#
#         for i in range(self.config.train.epoch):
#             edge_.append(dense_to_sparse(self.train_result['val_adj_matirix'][i])[0].shape[1])
#
#         f, axes = plt.subplots(figsize=(10, 5))
#
#         axes.plot(edge_, label='Learned Edge')
#         axes.set_xlabel('Epoch')
#         axes.set_ylabel('# of Total Edge')
#         axes.legend()
#
#     def visualize_test_adj_matrix(self):
#         learn_adj = self.test_result['adj_matrix']
#
#         f, axes = plt.subplots(figsize=(10, 10))
#
#         axes.imshow(learn_adj, cmap='Greys')
#
#     def visualize_learning_curve(self):
#         f, axes = plt.subplots(figsize=(10, 5))
#
#         axes.plot(self.train_result['train_loss'], label='Train Loss')
#         axes.plot(self.train_result['val_loss'], label='Validation Loss')
#         axes.set_xlabel('Epoch')
#         axes.set_ylabel('Loss')
#         axes.legend()
#
#     def visualize_node(self, node: int = -2, figure_num: int = 5, save: bool = False, backcast: bool = False):
#         nrow = figure_num
#         ncol = 1
#
#         input_length = self.config.forecasting_module.backcast_length
#         output_length = self.config.forecasting_module.forecast_length
#
#         x_axis = np.arange(input_length + output_length)
#
#         f, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(4 * nrow, 50 * ncol), dpi=70)
#
#         for ii in range(nrow):
#             axes[ii].plot(x_axis[:input_length], self.inputs[ii, node, 0], label='Input')
#
#             if backcast:
#                 axes[ii].plot(x_axis[:input_length], self.backcast[ii, node], label='Backcast')
#
#             axes[ii].plot(x_axis[input_length:], self.target[ii, node], label='Target')
#             axes[ii].plot(x_axis[input_length:], self.pred[ii, node], label='Prediction')
#             axes[ii].legend()
#
#         if save:
#             f.savefig('./total_result.png')
#
#     def visualize_stack_output(self, node: int = -2, save: bool = False):
#         nrow = self.per_trend_backcast.shape[1] + self.singual_backcast.shape[1]
#         ncol = 2
#
#         input_length = self.config.forecasting_module.backcast_length
#         output_length = self.config.forecasting_module.forecast_length
#
#         x_axis = np.arange(input_length + output_length)
#
#         if self.config.forecasting_module.name == 'pn_beats':
#             f, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(4 * nrow, 50 * ncol), dpi=70)
