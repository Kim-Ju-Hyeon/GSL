import torch
import torch.nn as nn

from models.N_BEATS.Parallel_N_model import Inter_Correlation_Block, TrendGenerator, SeasonalityGenerator


class MultiHead_Inter_Correlation_Block(Inter_Correlation_Block):
    def __init__(self, inter_correlation_block_type, n_theta_hidden, thetas_dim, backcast_length=10, forecast_length=5,
                 activation='ReLU', inter_correlation_stack_length=1, n_head=4):

        super(MultiHead_Inter_Correlation_Block, self).__init__(inter_correlation_block_type, n_theta_hidden,
                                                                thetas_dim, backcast_length, forecast_length,
                                                                activation, inter_correlation_stack_length)

        self.weight_sum = nn.Linear(n_head, 1)

    def forward(self, x, edge_index, edge_weight):
        for mlp in self.MLP_stack:
            x = mlp(x)
            x = self.drop_out(x)

        _multi_head = []
        for ii, layer in enumerate(self.Inter_Correlation_Block):
            for head in range(len(edge_index)):
                x = layer(x, edge_index[head], edge_weight[head])
                x = self.activ(x)
                x = self.batch_norm_layer_list[ii](x)
                x = self.drop_out(x)
                _multi_head.append(x)

            _multi_head = torch.stack(_multi_head, axis=0)
            x = self.weight_sum(_multi_head.permute(1, 2, 0)).squeeze()

        return x


class Parallel_TrendBlock(MultiHead_Inter_Correlation_Block):
    def __init__(self, inter_correlation_block_type, n_theta_hidden,
                 thetas_dim, backcast_length=10, forecast_length=5,
                 activation='ReLU', inter_correlation_stack_length=1, n_head=4):
        super(Parallel_TrendBlock, self).__init__(inter_correlation_block_type, n_theta_hidden,
                                                  thetas_dim, backcast_length, forecast_length,
                                                  activation, inter_correlation_stack_length, n_head)

        self.norm1 = nn.LayerNorm(self.n_theta_hidden[-1])

        self.backcast_trend_model = TrendGenerator(thetas_dim[0], backcast_length)
        self.forecast_trend_model = TrendGenerator(thetas_dim[1], forecast_length)

    def forward(self, x, edge_index, edge_weight):
        x = super(Parallel_TrendBlock, self).forward(x, edge_index, edge_weight)
        x = self.norm1(x)

        backcast = self.backcast_trend_model(self.theta_b_fc(x))
        forecast = self.forecast_trend_model(self.theta_f_fc(x))

        return backcast, forecast


class Parallel_SeasonalityBlock(MultiHead_Inter_Correlation_Block):
    def __init__(self, inter_correlation_block_type, n_theta_hidden,
                 thetas_dim, backcast_length=10, forecast_length=5,
                 activation='ReLU', inter_correlation_stack_length=1, n_head=4):
        super(Parallel_SeasonalityBlock, self).__init__(inter_correlation_block_type, n_theta_hidden,
                                                        thetas_dim, backcast_length, forecast_length,
                                                        activation, inter_correlation_stack_length, n_head)

        self.norm1 = nn.LayerNorm(self.n_theta_hidden[-1])

        self.backcast_seasonality_model = SeasonalityGenerator(backcast_length)
        self.forecast_seasonality_model = SeasonalityGenerator(forecast_length)

    def forward(self, x, edge_index, edge_weight):
        x = super(Parallel_SeasonalityBlock, self).forward(x, edge_index, edge_weight)
        x = self.norm1(x)

        backcast = self.backcast_seasonality_model(self.theta_b_fc(x))
        forecast = self.forecast_seasonality_model(self.theta_f_fc(x))

        return backcast, forecast


class Parallel_GenericBlock(MultiHead_Inter_Correlation_Block):
    def __init__(self, inter_correlation_block_type, n_theta_hidden,
                 thetas_dim, backcast_length=10, forecast_length=5,
                 activation='ReLU', inter_correlation_stack_length=1, n_head=4):
        super(Parallel_GenericBlock, self).__init__(inter_correlation_block_type, n_theta_hidden,
                                                    thetas_dim, backcast_length, forecast_length,
                                                    activation, inter_correlation_stack_length, n_head)

        self.norm1 = nn.LayerNorm(self.n_theta_hidden[-1])

        self.backcast_fc = nn.Linear(thetas_dim[0], backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim[1], forecast_length)

    def forward(self, x, edge_index, edge_weight=None):
        x = super(Parallel_GenericBlock, self).forward(x, edge_index, edge_weight)
        x = self.norm1(x)

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)
        forecast = self.forecast_fc(theta_f)

        return backcast, forecast
