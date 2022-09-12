import click
from runner.ic_pn_beats_runner import Runner
import traceback
from utils.logger import setup_logging
import os
from easydict import EasyDict as edict
import yaml


@click.command()
@click.option('--conf_file_path', type=click.STRING, default=None)
@click.option('--inference', type=bool, default=True)
def main(conf_file_path, inference):
    if inference:
        log_save_name = 'Inference_log_exp'
    else:
        log_save_name = 'Train_Resume'
    config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))
    log_file = os.path.join(config.exp_sub_dir, f"{log_save_name}_log_exp_{config.seed}.txt")
    logger = setup_logging('INFO', log_file, logger_name=str(config.seed))
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(config.exp_name))

    try:
        if not inference:
            config.train_resume = True

        runner = Runner(config=config)

        if not inference:
            runner.train()

        runner.test()
    except:
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    main()
