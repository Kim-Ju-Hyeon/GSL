import pickle
import click
from easydict import EasyDict as edict
import yaml
import traceback
import os
import torch
from dataset.temporal_graph_dataset import Temporal_Graph_Signal


def download_save_dataset(config):
    num_timesteps_in = config.forecasting_module.backcast_length
    num_timesteps_out = config.forecasting_module.forecast_length
    batch_size = config.train.batch_size
    dataset_hyperparameter = f'{num_timesteps_in}_{num_timesteps_out}_{batch_size}'

    ett_dataset_list = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2']
    if config.dataset.name in ett_dataset_list:
        path = f'./data/ETT/{config.dataset.name}'
    else:
        path = f'./data/{config.dataset.name}'

    path = os.path.join(path, f'temporal_signal_{dataset_hyperparameter}.pickle')

    loader = Temporal_Graph_Signal(config.dataset.name, config.dataset.scaler_type)

    loader.preprocess_dataset()
    train_dataset, valid_dataset, test_dataset = loader.get_dataset(
        num_timesteps_in=config.forecasting_module.backcast_length,
        num_timesteps_out=config.forecasting_module.forecast_length,
        batch_size=config.train.batch_size)

    scaler = loader.get_scaler()

    temporal_signal = {'train': train_dataset,
                       'validation': valid_dataset,
                       'test': test_dataset,
                       'scaler': scaler}

    pickle.dump(temporal_signal, open(path, 'wb'))


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
    