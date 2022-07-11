import os
import gdown
import numpy as np
import pandas as pd
import random
import torch
from utils.dataset_utils import time_features_from_frequency_str
from utils.utils import build_fully_connected_edge_idx, build_batch_edge_index
from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch
from utils.scalers import Scaler
from dataset.make_dataset_base import DatasetLoader

class M5DatasetLoader(DatasetLoader):
    def __init__(self, raw_data_dir, scaler_type='std'):
        super(M5DatasetLoader, self).__init__(raw_data_dir, scaler_type)


    def _download_url(self):
        url = 'https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip'
        os.makedirs(self.path)
        gdown.download(url, os.path.join(self.path, 'M5.csv'))

    def _read_web_data(self):
        if not os.path.isfile(os.path.join(self.path, 'M5.csv')):
            self._download_url()