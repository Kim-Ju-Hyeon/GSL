import pickle
import click
from easydict import EasyDict as edict
import yaml
import traceback
import os
import torch
from dataset.temporal_graph_dataset import Temporal_Graph_Signal


def download_save_dataset(config):
    loader = Temporal_Graph_Signal(config.dataset.name, config.dataset.scaler_type, config.dataset.univariate)
    loader.preprocess_dataset()


@click.command()
@click.option('--conf_file_path', type=click.STRING, default=None)
def main(conf_file_path):
    try:
        config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))
        download_save_dataset(config)
        print('Dataset Ready')
    except:
        print(traceback.format_exc())


if __name__ == '__main__':
    main()
