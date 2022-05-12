# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from typing import Tuple
# from utils.utils import squeeze_last_dim
# from models.N_BEATS.B_Block import Block
#
#
# ACTIVATIONS = ['ReLU',
#                'Softplus',
#                'Tanh',
#                'SELU',
#                'LeakyReLU',
#                'PReLU',
#                'Sigmoid']
#
#
# class NHITSBlock(Block):
#     def __init__(self, n_theta_hidden, thetas_dim, pooling_mode, n_pool_kernel_size, backcast_length=10, forecast_length=5, activation='ReLU'):
#         self.backcast_length = backcast_length
#         self.forecast_length = forecast_length
#         self.l_out = int(((backcast_length - n_pool_kernel_size) / n_pool_kernel_size) + 1)
#
#         super(NHITSBlock, self).__init__(n_theta_hidden, thetas_dim, self.l_out, forecast_length, activation)
#         assert (pooling_mode in ['max', 'average'])
#
#         self.n_pool_kernel_size = n_pool_kernel_size
#
#         if pooling_mode == 'max':
#             self.pooling_layer = nn.MaxPool1d(kernel_size=self.n_pool_kernel_size,
#                                               stride=self.n_pool_kernel_size, ceil_mode=True)
#         elif pooling_mode == 'average':
#             self.pooling_layer = nn.AvgPool1d(kernel_size=self.n_pool_kernel_size,
#                                               stride=self.n_pool_kernel_size, ceil_mode=True)
#
#
#     def forward(self, x):
#         x = squeeze_last_dim(x)
#         x = self.pooling_layer(x)
#         x = super(NHITSBlock, self).forward(x)
#
#         theta_b = self.theta_b_fc(x)
#         theta_f = self.theta_f_fc(x)
#
#         backcast = F.interpolate(theta_b[:, None, :], size=self.backcast_length,
#                                  mode='linear', align_corners=False).squeeze()
#         forecast = F.interpolate(theta_f[:, None, :], size=self.forecast_length,
#                                  mode='linear', align_corners=False).squeeze()
#
#         return backcast, forecast
