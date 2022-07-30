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
@click.option('--n_pool_kernel_size', type=click.STRING, default='4')
@click.option('--n_stride_size', type=click.STRING, default='2')
@click.option('--factor', type=int, default=2)
def main(conf_file_path, stack_num, n_pool_kernel_size, n_stride_size, factor):
    temp = stack_num // 3
    n_pool_kernel_size = n_pool_kernel_size.split(',')
    n_pool_kernel_size = [int(j.strip()) for j in n_pool_kernel_size]
    n_pool_kernel_size = n_pool_kernel_size * temp
    n_pool_kernel_size.sort(reverse=True)

    n_stride_size = n_stride_size.split(',')
    n_stride_size = [int(jj.strip()) for jj in n_stride_size]
    n_stride_size = n_stride_size * temp
    n_stride_size.sort(reverse=True)

    mlp_stack_list = [512]
    n_head_list = [4, 16]

    for n_head in n_head_list:
        for mlp_stack in mlp_stack_list:
            config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))

            hyperparameter = f'stacks_{stack_num}' \
                             f'__n_head_{n_head}' \
                             f'__mlp_stack_{mlp_stack}'

            now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
            sub_dir = '_'.join([hyperparameter, now.strftime('%m%d_%H%M%S')])
            config.seed = set_seed(config.seed)

            config.exp_name = config.exp_name

            config.exp_dir = os.path.join(config.exp_dir, str(config.exp_name))
            config.exp_sub_dir = os.path.join(config.exp_dir, sub_dir)
            config.model_save = os.path.join(config.exp_sub_dir, "model_save")

            mkdir(config.model_save)

            config.forecasting_module.stack_num = stack_num
            config.forecasting_module.n_pool_kernel_size = n_pool_kernel_size
            config.forecasting_module.n_theta_hidden = [mlp_stack]
            config.forecasting_module.n_stride_size = n_stride_size

            config.graph_learning.factor = factor
            config.graph_learning.n_head = n_head

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
