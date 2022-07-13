from models.N_BEATS.Block import *


class PN_model(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self, config):
        super(PN_model, self).__init__()
        assert len(config.n_pool_kernel_size) == len(
            config.n_stride_size), f'pooling kernel: {len(config.n_pool_kernel_size)} and stride: {len(config.n_stride_size)} is not match'
        assert len(config.n_pool_kernel_size) == config.stack_num, 'Pooling num and Stack num is not match'

        self.activation = config.activ

        self.stack_num = config.stack_num
        self.singular_stack_num = config.singular_stack_num

        self.inter_correlation_block_type = config.inter_correlation_block_type
        self.forecast_length = config.forecast_length
        self.backcast_length = config.backcast_length
        self.n_theta_hidden = config.n_theta_hidden
        self.thetas_dim = config.thetas_dim
        self.n_layers = config.inter_correlation_stack_length
        self.share_weights_in_stack = config.share_weights_in_stack

        self.pooling_mode = config.pooling_mode
        self.n_pool_kernel_size = config.n_pool_kernel_size
        self.n_stride_size = config.n_stride_size

        self.parameters = []

        # Pooling
        self.pooling_stack = []
        for i in range(len(self.n_pool_kernel_size)):
            if self.pooling_mode == 'max':
                pooling_layer = nn.MaxPool1d(kernel_size=self.n_pool_kernel_size[i],
                                             stride=self.n_stride_size[i], ceil_mode=False)
            elif self.pooling_mode == 'average':
                pooling_layer = nn.AvgPool1d(kernel_size=self.n_pool_kernel_size[i],
                                             stride=self.n_stride_size[i], ceil_mode=False)
            else:
                raise ValueError('Invalid Pooling Mode Only "max", "average" is available')

            self.pooling_stack.append(pooling_layer)

        # For Inference Interpret
        self.per_trend_backcast = []
        self.per_trend_forecast = []

        self.per_seasonality_backcast = []
        self.per_seasonality_forecast = []

        self.singual_backcast = []
        self.singual_forecast = []

        # Make Stack For Trend, Seasonality, Singular
        self.trend_stacks = []
        self.seasonality_stacks = []
        self.sigular_stacks = []

        for stack_id in range(self.stack_num):
            self.trend_stacks.append(self.create_stack('trend', stack_id))

        for stack_id in range(self.stack_num):
            self.seasonality_stacks.append(self.create_stack('seasonality', stack_id))

        for stack_id in range(self.singular_stack_num):
            self.sigular_stacks.append(self.create_stack('generic', stack_id))

        self.parameters = nn.ParameterList(self.parameters)

    def create_stack(self, stack_type, stack_num):
        # stack_num -> make theta dim bigger for high stack that modeling high order polynomial
        block_init = PN_model.select_block(stack_type)

        if stack_type == PN_model.TREND_BLOCK:
            thetas_dim = [0, 0]

            thetas_dim[0] = 3
            thetas_dim[1] = 3

            block = block_init(inter_correlation_block_type=self.inter_correlation_block_type,
                               n_theta_hidden=self.n_theta_hidden, thetas_dim=thetas_dim,
                               backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                               activation=self.activation,
                               inter_correlation_stack_length=self.n_layers)

        elif stack_type == PN_model.SEASONALITY_BLOCK:
            thetas_dim = [0, 0]

            thetas_dim[0] = 2 * int(self.backcast_length / 2 - 1) + 1
            thetas_dim[1] = 2 * int(self.forecast_length / 2 - 1) + 1

            block = block_init(inter_correlation_block_type=self.inter_correlation_block_type,
                               n_theta_hidden=self.n_theta_hidden, thetas_dim=thetas_dim,
                               backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                               activation=self.activation,
                               inter_correlation_stack_length=self.n_layers)

        elif stack_type == PN_model.GENERIC_BLOCK:
            block = block_init(inter_correlation_block_type=self.inter_correlation_block_type,
                               n_theta_hidden=self.n_theta_hidden, thetas_dim=self.thetas_dim,
                               backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                               activation=self.activation,
                               inter_correlation_stack_length=self.n_layers)

        self.parameters.extend(block.parameters())

        return block

    @staticmethod
    def select_block(block_type):
        if block_type == PN_model.SEASONALITY_BLOCK:
            return GNN_SeasonalityBlock
        elif block_type == PN_model.TREND_BLOCK:
            return GNN_TrendBlock
        elif block_type == PN_model.GENERIC_BLOCK:
            return GNN_GenericBlock
        else:
            raise ValueError("Invalid block type")

    def forward(self, inputs, edge_index, edge_weight=None, interpretability=False):
        # inputs shape = [Batch_size x Nodes Num, Sequence Length]
        device = inputs.device

        forecast = torch.zeros(size=(inputs.size()[0], self.forecast_length)).to(device=device)
        backcast = torch.zeros(size=(inputs.size()[0], self.backcast_length)).to(device=device)

        for stack_index in range(self.stack_num):
            pooled_inputs = self.pooling_stack[stack_index](inputs)
            interpolate_inputs = F.interpolate(pooled_inputs.unsqueeze(dim=1), size=inputs.size()[1],
                                               mode='linear', align_corners=False).squeeze(dim=1)

            trend_b, trend_f = self.trend_stacks[stack_index](interpolate_inputs, edge_index, edge_weight)
            seasonality_b, seasonality_f = self.seasonality_stacks[stack_index](interpolate_inputs, edge_index,
                                                                                edge_weight)

            if interpretability:
                self.per_trend_backcast.append(trend_b.cpu().numpy())
                self.per_trend_forecast.append(trend_f.cpu().numpy())

                self.per_seasonality_backcast.append(seasonality_b.cpu().numpy())
                self.per_seasonality_forecast.append(seasonality_f.cpu().numpy())

            inputs = inputs - trend_b + seasonality_b

            forecast = forecast + trend_f + seasonality_f
            backcast = backcast + trend_b + seasonality_b

        for singular_stack_index in range(self.singular_stack_num):
            singular_b, singular_f = self.sigular_stacks[singular_stack_index](inputs, edge_index, edge_weight)

            if interpretability:
                self.singual_backcast.append(singular_b.cpu().numpy())
                self.singual_forecast.append(singular_f.cpu().numpy())

            inputs = inputs - singular_b

            forecast = forecast + singular_f
            backcast = backcast + singular_b

        return backcast, forecast
