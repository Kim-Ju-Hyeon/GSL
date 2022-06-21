import click
from runner.runner import Runner

from utils.train_helper import get_config, save_yaml
import traceback
from utils.logger import setup_logging
import os

from utils.slack import slack_message
from utils.train_helper import set_seed, mkdir, edict2dict
import datetime

import pytz
from easydict import EasyDict as edict
import yaml
import logging


@click.command()
@click.option('--conf_file_path', type=click.STRING, default='./config/nbeats_general.yaml')
@click.option('--n_stack', type=int, default=1)
@click.option('--n_block', type=int, default=1)
@click.option('--kernel_size', type=int, default=1)
@click.option('--inter_correlation_stack_length', type=int, default=1)
def main(conf_file_path, n_stack, n_block, kernel_size, inter_correlation_stack_length):
    try:
        mlp_stack_list = [[128, 64, 32], [64, 64, 64], [64]]
        thetas_dim_list = [[64, 16], [32, 8]]
        pooling_mode_list = ['max', 'average']

        for pooling_mode in pooling_mode_list:
            for mlp_stack in mlp_stack_list:
                for thetas_dim in thetas_dim_list:

                    hyperparameter = f'pooling_mode_{pooling_mode}__kernel_size_{kernel_size}_' \
                                     f'_correlationStackLen_{inter_correlation_stack_length}__stacks_' \
                                     f'{n_stack}__num_blocks_per_stack_{n_block}__n_theta_hidden_{mlp_stack}' \
                                     f'__thetas_dim_{thetas_dim}'

                    temp = f'{pooling_mode}_{kernel_size}_{inter_correlation_stack_length}' \
                           f'_{n_stack}_{n_block}_{mlp_stack}_{thetas_dim}'

                    now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
                    sub_dir = '_'.join([temp, now.strftime('%m%d_%H%M%S')])
                    config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))
                    config.seed = set_seed(config.seed)

                    config.exp_dir = os.path.join(config.exp_dir, str(config.exp_name))
                    config.exp_sub_dir = os.path.join(config.exp_dir, config.model_name, sub_dir)
                    config.model_save = os.path.join(config.exp_sub_dir, "model_save")

                    mkdir(config.model_save)

                    config.forecasting_module.num_blocks_per_stack = n_block
                    config.forecasting_module.n_theta_hidden = mlp_stack
                    config.forecasting_module.thetas_dim = thetas_dim
                    config.forecasting_module.stack_types = ['n_hits'] * n_stack

                    config.forecasting_module.pooling_mode = pooling_mode
                    config.forecasting_module.kernel_size = kernel_size
                    config.forecasting_module.inter_correlation_stack_length = inter_correlation_stack_length

                    save_name = os.path.join(config.exp_sub_dir, 'config.yaml')
                    yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

                    log_file = os.path.join(config.exp_sub_dir, "log_exp_{}.txt".format(config.seed))
                    logger = setup_logging('INFO', log_file)
                    logger.info("Writing log file to {}".format(log_file))
                    logger.info("Exp instance id = {}".format(config.exp_name))
                    logger.info(f"HyperParameter Setting {hyperparameter}")

                    runner = Runner(config=config)
                    runner.train()
                    runner.test()

    except:
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    main()
