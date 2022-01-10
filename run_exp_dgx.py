import click
from runner.GTS_runner import GTS_Runner
from utils.train_helper import get_config
import traceback
from utils.logger import setup_logging
import datetime
import os

from utils.slack import slack_message, send_slack_message

@click.command()
@click.option('--conf_file_path', type=str, default=None)
def main(conf_file_path):
    start = datetime.datetime.now() + datetime.timedelta(hours=9)
    start_string = start.strftime('%Y-%m-%d %I:%M:%S %p')

    config = get_config(conf_file_path)

    log_file = os.path.join(config.exp_dir, "log_exp_{}.txt".format(config.seed))
    logger = setup_logging('INFO', log_file)
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(config.exp_name))

    try:
        runner = GTS_Runner(config=config)
        runner.train()
        runner.test()
        end_string = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d %I:%M:%S %p')

        slack_message(start,
                      f"EXP Name: {config.exp_name} \n Training Success \n Start at {start_string} \n End at {end_string}")



    except:
        logger.error(traceback.format_exc())
        end_string = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d %I:%M:%S %p')
        slack_message(start,
                      f"EXP Name: {config.exp_name} \n Start at {start_string} \n End at {end_string} \n Error!!!!!")
        send_slack_message(traceback.format_exc())


if __name__ == '__main__':
    main()
