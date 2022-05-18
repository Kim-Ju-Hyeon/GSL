import click
from runner.runner import Runner

from utils.train_helper import get_config, save_yaml
import traceback
from utils.logger import setup_logging
import os

from utils.slack import slack_message
import datetime


@click.command()
@click.option('--conf_file_path', type=str, default=None)
@click.option('--n_theta_hidden', type=list, default=None)
@click.option('--thetas_dim', type=list, default=None)
def main(conf_file_path, n_theta_hidden=None, thetas_dim=None):
    start = datetime.datetime.now()
    start = start + datetime.timedelta(hours=9)

    config = get_config(conf_file_path)
    if n_theta_hidden is not None:
        config.forecasting_module.n_theta_hidden = n_theta_hidden
    if thetas_dim is not None:
        config.forecasting_module.thetas_dim = thetas_dim
    save_yaml(config)

    log_file = os.path.join(config.exp_sub_dir, "log_exp_{}.txt".format(config.seed))
    logger = setup_logging('INFO', log_file)
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(config.exp_name))

    try:
        runner = Runner(config=config)
        runner.train()
        runner.test()

        slack_message(start, f"{config.exp_name}: Training Success")

    except:
        logger.error(traceback.format_exc())
        slack_message(start, f"{config.exp_name}: Error \n {traceback.format_exc()}")


if __name__ == '__main__':
    main()
