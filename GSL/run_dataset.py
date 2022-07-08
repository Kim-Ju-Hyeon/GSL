import pickle
import click
from easydict import EasyDict as edict
import yaml
import os
import torch
from torch_geometric.loader import DataLoader

from dataset.make_spike_datset import MakeSpikeDataset
from dataset.make_traffic_dataset import TrafficDatasetLoader
from dataset.ecl import ECLDatasetLoader


def download_save_dataset(config):
    dataset_conf = config.dataset
    train_conf = config.train

    num_timesteps_in = config.forecasting_module.backcast_length
    num_timesteps_out = config.forecasting_module.forecast_length
    batch_size = train_conf.batch_size
    dataset_hyperparameter = f'{num_timesteps_in}_{num_timesteps_out}_{batch_size}'

    if os.path.exists(os.path.join(dataset_conf.root, f'temporal_signal_{dataset_hyperparameter}.pickle')):
        temporal_signal = pickle.load(
            open(os.path.join(dataset_conf.root, f'temporal_signal_{dataset_hyperparameter}.pickle'), 'rb'))
        train_dataset = temporal_signal['train']
        valid_dataset = temporal_signal['validation']
        test_dataset = temporal_signal['test']
        entire_inputs = temporal_signal['entire_inputs'][:, :, :dataset_conf.graph_learning_length]
        scaler = temporal_signal['scaler']

    else:
        if dataset_conf.name == 'spike_lambda_bin100':
            spike = pickle.load(open('./data/spk_bin_n100.pickle', 'rb'))

            entire_inputs = torch.FloatTensor(spike[:, :dataset_conf.graph_learning_length])

            dataset_maker = MakeSpikeDataset(config)
            total_dataset = dataset_maker.make()

            train_dataset = DataLoader(total_dataset['train'], batch_size=train_conf.batch_size)
            valid_dataset = DataLoader(total_dataset['valid'], batch_size=train_conf.batch_size)
            test_dataset = DataLoader(total_dataset['test'], batch_size=train_conf.batch_size)

        elif (dataset_conf.name == 'METR-LA') or (dataset_conf.name == 'PEMS-BAY'):
            loader = TrafficDatasetLoader(raw_data_dir=dataset_conf.root, dataset_name=dataset_conf.name,
                                          scaler_type=config.dataset.scaler_type)

        elif dataset_conf.name == 'ECL':
            loader = ECLDatasetLoader(raw_data_dir=dataset_conf.root,
                                      scaler_type=config.dataset.scaler_type)
        else:
            raise ValueError("Non-supported dataset!")

        train_dataset, valid_dataset, test_dataset, entire_inputs = loader.get_dataset(
            num_timesteps_in=config.forecasting_module.backcast_length,
            num_timesteps_out=config.forecasting_module.forecast_length,
            batch_size=train_conf.batch_size)

        entire_inputs = entire_inputs[:, :, :dataset_conf.graph_learning_length]
        scaler = loader.get_scaler()

        temporal_signal = {'train': train_dataset,
                           'validation': valid_dataset,
                           'test': test_dataset,
                           'entire_inputs': entire_inputs,
                           'scaler': scaler}

        pickle.dump(temporal_signal,
                    open(os.path.join(dataset_conf.root, f'temporal_signal_{dataset_hyperparameter}.pickle'),
                         'wb'))


@click.command()
@click.option('--conf_file_path', type=click.STRING, default=None)
def main(conf_file_path):
    config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))
    