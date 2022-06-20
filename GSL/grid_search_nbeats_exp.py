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
@click.option('--conf_file_path', type=click.STRING, default=None)
@click.option('--type', type=click.STRING, default=None)
def main(conf_file_path, type):
    num_stack_list = [3, 9]
    num_blocks_per_stack_list = [3, 9]
    mlp_stack_list = [[128, 64, 32], [64, 64, 64]]
    # thetas_dim_list = [[64, 16], [128, 32], [60, 12]]

    for n_stack in num_stack_list:
        for n_block in num_blocks_per_stack_list:
            for mlp_stack in mlp_stack_list:
                # for thetas_dim in thetas_dim_list:
                hyperparameter = f'stacks_{n_stack}__num_blocks_per_stack_{n_block}__n_theta_hidden_{mlp_stack}'

                start = datetime.datetime.now()
                start = start + datetime.timedelta(hours=9)

                config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))
                config.model_name = type
                config.seed = set_seed(config.seed)

                config.exp_dir = os.path.join(config.exp_dir, str(config.exp_name))
                config.exp_sub_dir = os.path.join(config.exp_dir, config.model_name, hyperparameter)
                config.model_save = os.path.join(config.exp_sub_dir, "model_save")

                mkdir(config.model_save)

                config.forecasting_module.inter_correlation_block_type = type
                config.forecasting_module.num_blocks_per_stack = n_block
                config.forecasting_module.n_theta_hidden = mlp_stack
                # config.forecasting_module.thetas_dim = thetas_dim
                config.forecasting_module.stack_types = ['trend'] * n_stack + ['seasonality'] * n_stack

                save_name = os.path.join(config.exp_sub_dir, 'config.yaml')
                yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

                log_file = os.path.join(config.exp_sub_dir, "log_exp_{}.txt".format(config.seed))
                logger = setup_logging('INFO', log_file)
                logger.info("Writing log file to {}".format(log_file))
                logger.info("Exp instance id = {}".format(config.exp_name))

                runner = Runner(config=config)
                runner.train()
                runner.test()


if __name__ == '__main__':
    main()
