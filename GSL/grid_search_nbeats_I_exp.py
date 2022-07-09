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
@click.option('--n_stack', type=int, default=1)
@click.option('--n_block', type=int, default=1)
@click.option('--mlp_stack', type=click.STRING, default='64,64,64')
def main(conf_file_path, n_stack, n_block, mlp_stack):
    mlp_stack = mlp_stack.split(',')
    mlp_stack = [int(j.strip())for j in mlp_stack]

    hyperparameter = f'stacks_{n_stack}__num_blocks_per_stack_{n_block}__n_theta_hidden_{mlp_stack}'

    now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    sub_dir = '_'.join([hyperparameter, now.strftime('%m%d_%H%M%S')])
    config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))
    config.seed = set_seed(config.seed)

    config.exp_dir = os.path.join(config.exp_dir, str(config.exp_name))
    config.exp_sub_dir = os.path.join(config.exp_dir, sub_dir)
    config.model_save = os.path.join(config.exp_sub_dir, "model_save")

    mkdir(config.model_save)

    config.forecasting_module.num_blocks_per_stack = n_block
    config.forecasting_module.n_theta_hidden = mlp_stack
    config.forecasting_module.stack_types = ['smoothing_trend'] * n_stack + ['seasonality'] * (n_stack // 2) + ['generic']

    save_name = os.path.join(config.exp_sub_dir, 'config.yaml')
    yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

    log_file = os.path.join(config.exp_sub_dir, "log_exp_{}.txt".format(config.seed))
    logger = setup_logging('INFO', log_file)
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
