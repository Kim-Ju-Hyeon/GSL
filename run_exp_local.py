import click
from runner.GTS_runner import GTS_Runner
from utils.train_helper import get_config
import traceback

@click.command()
@click.option('--conf_file_path', type=str, default=None)
def main(conf_file_path):
    config = get_config(conf_file_path)

    try:
        runner = GTS_Runner(config=config)
        runner.train()

    except:
        print(traceback.format_exc())


if __name__ == '__main__':
    main()
