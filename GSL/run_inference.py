import click
from runner.runner import Runner
import traceback
from utils.logger import setup_logging
import os
from easydict import EasyDict as edict
import yaml


@click.command()
@click.option('--conf_file_path', type=click.STRING, default=None)
def main(conf_file_path):
    config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))
    log_file = os.path.join(config.exp_sub_dir, "Inference_log_exp_{}.txt".format(config.seed))
    logger = setup_logging('INFO', log_file, logger_name=str(config.seed))
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(config.exp_name))

    try:
        runner = Runner(config=config)
        runner.test()
    except:
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    main()
