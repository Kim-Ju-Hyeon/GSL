import click
from runner.runner import Runner
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
# @click.option('--singular_stack_num', type=int, default=1)
@click.option('--n_pool_kernel_size', type=click.STRING, default='4')
@click.option('--n_stride_size', type=click.STRING, default='2')
# @click.option('--mlp_stack', type=click.STRING, default='64,64,64')
@click.option('--gl', type=bool, default=True)
@click.option('--edge_prob', type=float, default=0.02)
def main(conf_file_path, stack_num, n_pool_kernel_size, n_stride_size, edge_prob, gl):
    config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))

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
    singular_stack_num_list = [0]

    for singular_stack_num in singular_stack_num_list:
        for mlp_stack in mlp_stack_list:

            if not gl:
                # if edge_prob < 0.05:
                #     config.train.batch_size = 32
                # elif 0.05 < edge_prob < 0.1:
                #     config.train.batch_size = 8
                # elif 0.1 < edge_prob < 0.3:
                #     config.train.batch_size = 4
                # else:
                #     config.train.batch_size = 1

                config.graph_learning.edge_prob = edge_prob

                hyperparameter = f'edge_prob_{edge_prob}__stacks_{stack_num}__singular_stack_num_{singular_stack_num}' \
                                 f'__mlp_stack_{mlp_stack}'
            else:
                hyperparameter = f'graph_learning_{config.graph_learning.mode}__stacks_{stack_num}' \
                                 f'__singular_stack_num_{singular_stack_num}' \
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
            config.forecasting_module.singular_stack_num = singular_stack_num
            config.forecasting_module.n_pool_kernel_size = n_pool_kernel_size
            config.forecasting_module.n_theta_hidden = [mlp_stack]
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
