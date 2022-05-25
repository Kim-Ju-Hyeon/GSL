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


@click.command()
@click.option('--conf_file_path', type=click.STRING, default=None)
@click.option('--stack_type', type=click.STRING, default='generic,generic,generic')
@click.option('--num_blocks_per_stack', type=int, default=1)
@click.option('--n_theta_hidden', type=click.STRING, default='16,16')
@click.option('--thetas_dim', type=click.STRING, default='4,4')
def main(conf_file_path, stack_type, num_blocks_per_stack, n_theta_hidden, thetas_dim):
    n_theta_hidden = n_theta_hidden.split(',')
    n_theta_hidden = [int(i.strip()) for i in n_theta_hidden]
    thetas_dim = thetas_dim.split(',')
    thetas_dim = [int(j.strip()) for j in thetas_dim]
    stack_type = stack_type.split(',')
    stack_type = [j.strip() for j in stack_type]

    hyperparameter = f'stacks_{len(stack_type)}__num_blocks_per_stack_{num_blocks_per_stack}__n_theta_hidden_{n_theta_hidden}__thetas_dim_{thetas_dim}'

    start = datetime.datetime.now()
    start = start + datetime.timedelta(hours=9)

    config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))
    now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    config.seed = set_seed(config.seed)

    config.exp_dir = os.path.join(config.exp_dir, str(config.exp_name))
    config.exp_sub_dir = os.path.join(config.exp_dir, config.model_name, hyperparameter)
    config.model_save = os.path.join(config.exp_sub_dir, "model_save")

    mkdir(config.model_save)

    config.forecasting_module.num_blocks_per_stack = num_blocks_per_stack
    config.forecasting_module.n_theta_hidden = n_theta_hidden
    config.forecasting_module.thetas_dim = thetas_dim
    config.forecasting_module.stack_type = stack_type

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

        # slack_message(start, f"{config.exp_name}: Training Success")

    except:
        logger.error(traceback.format_exc())
        # slack_message(start, f"{config.exp_name}: Error \n {traceback.format_exc()}")


if __name__ == '__main__':
    main()
