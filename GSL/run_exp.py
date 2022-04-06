import click
from runner.GTS_runner import GTS_Runner
from runner.runner import Runner

from utils.train_helper import get_config
import traceback
from utils.logger import setup_logging
import os



@click.command()
@click.option('--conf_file_path', type=str, default=None)
def main(conf_file_path):
    config = get_config(conf_file_path)

    log_file = os.path.join(config.exp_dir, "log_exp_{}.txt".format(config.seed))
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
