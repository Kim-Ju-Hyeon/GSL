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
def main(conf_file_path):
    config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))

    now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    sub_dir = now.strftime('%m%d_%H%M%S')
    config.seed = set_seed(config.seed)

    config.exp_name = config.exp_name

    config.exp_dir = os.path.join(config.exp_dir, str(config.exp_name))
    config.exp_sub_dir = os.path.join(config.exp_dir, sub_dir)
    config.model_save = os.path.join(config.exp_sub_dir, "model_save")

    mkdir(config.model_save)

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
