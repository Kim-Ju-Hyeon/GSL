import click
from runner.ic_pn_beats_runner import Runner
import traceback
from utils.logger import setup_logging
import os
from utils.train_helper import set_seed, mkdir, edict2dict
import datetime

import pytz
from easydict import EasyDict as edict
import yaml


@click.command()
@click.option('--conf_file_path', type=click.STRING, default=None)
@click.option('--stack_num', type=int, default=1)
@click.option('--singular_stack', type=int, default=1)
def main(conf_file_path, stack_num, singular_stack):
    mlp_stack_list = [32, 256, 512]
    mlp_num = [1, 2, 4]
    thetas_dim_list = [[16, 16], [64, 64], [256, 256], [512, 512]]

    for thetas_dim in thetas_dim_list:
        for _mlp_num in mlp_num:
            for mlp_stack in mlp_stack_list:
                config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))

                hyperparameter = f'stacks_{stack_num}' \
                                 f'__mlp_stack_{mlp_stack}*{_mlp_num}' \
                                 f'__singular_stack_{singular_stack}' \
                                 f'__thetas_dim_{thetas_dim}'

                now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
                sub_dir = '_'.join([hyperparameter, now.strftime('%m%d_%H%M%S')])
                config.seed = set_seed(config.seed)

                config.exp_name = config.exp_name

                config.exp_dir = os.path.join(config.exp_dir, str(config.exp_name))
                config.exp_sub_dir = os.path.join(config.exp_dir, sub_dir)
                config.model_save = os.path.join(config.exp_sub_dir, "model_save")

                mkdir(config.model_save)

                if config.dataset.name == 'ECL':
                    if stack_num == 3:
                        config.train.batch_size = 8
                    elif stack_num == 6:
                        config.train.batch_size = 6
                    elif stack_num == 9:
                        config.train.batch_size = 5

                if stack_num == 1:
                    n_pool_kernel_size = [8]
                    n_stride_size = [4]
                elif stack_num == 2:
                    n_pool_kernel_size = [8, 4]
                    n_stride_size = [4, 2]
                elif stack_num == 3:
                    n_pool_kernel_size = [8, 4, 2]
                    n_stride_size = [4, 2, 1]

                config.forecasting_module.stack_num = stack_num
                config.forecasting_module.singular_stack_num = singular_stack
                config.forecasting_module.n_theta_hidden = [mlp_stack] * mlp_num
                config.forecasting_module.thetas_dim = thetas_dim

                config.forecasting_module.n_pool_kernel_size = n_pool_kernel_size
                config.forecasting_module.n_stride_size = n_stride_size

                save_name = os.path.join(config.exp_sub_dir, 'config.yaml')
                yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

                log_file = os.path.join(config.exp_sub_dir, "log_exp_{}.txt".format(config.seed))
                logger = setup_logging('INFO', log_file, logger_name=str(config.seed))
                logger.info("Writing log file to {}".format(log_file))
                logger.info("Exp instance id = {}".format(config.exp_name))

                try:
                    runner = Runner(config=config)
                    runner.train()
                    runner.test()

                except:
                    logger.error(traceback.format_exc())


if __name__ == '__main__':
    main()
