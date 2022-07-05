__all__ = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ETTInfo', 'ETT']

# Cell
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from utils.dataset_utils import Info, time_features_from_frequency_str, process_multiple_ts

# Cell
@dataclass
class ETTh1:
    freq: str = 'H'
    name: str = 'ETTh1'
    n_ts: int = 7

@dataclass
class ETTh2:
    freq: str = 'H'
    name: str = 'ETTh2'
    n_ts: int = 7

@dataclass
class ETTm1:
    freq: str = '15T'
    name: str = 'ETTm1'
    n_ts: int = 7

@dataclass
class ETTm2:
    freq: str = '15T'
    name: str = 'ETTm2'
    n_ts: int = 7

# Cell
ETTInfo = Info(groups=('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'),
               class_groups=(ETTh1, ETTh2, ETTm1, ETTm2))

# Cell
@dataclass
class ETT:

    source_url: str = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/'

    @staticmethod
    def load(directory: str,
             group: str,
             cache: bool = True) -> Tuple[pd.DataFrame,
                                          Optional[pd.DataFrame],
                                          Optional[pd.DataFrame]]:
        """Downloads and loads ETT data.
        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.
        group: str
            Group name.
            Allowed groups: 'ETTh1', 'ETTh2',
                            'ETTm1', 'ETTm2'.
        cache: bool
            If `True` saves and loads
        Notes
        -----
        [1] Returns train+val+test sets.
        """
        path = f'{directory}/ett/datasets'
        file_cache = f'{path}/{group}.p'

        if os.path.exists(file_cache) and cache:
            df, X_df, S_df = pd.read_pickle(file_cache)

            return df, X_df, S_df


        ETT.download(directory)
        path = f'{directory}/ett/datasets'
        class_group = ETTInfo[group]

        y_df = pd.read_csv(f'{path}/{group}.csv')

        y_df, X_df = process_multiple_ts(y_df)

        S_df = None
        if cache:
            pd.to_pickle((y_df, X_df, S_df), file_cache)

        return y_df, X_df, S_df

    # @staticmethod
    # def download(directory: str) -> None:
    #     """Download ETT Dataset."""
    #     path = f'{directory}/ett/datasets/'
    #     if not os.path.exists(path):
    #         for group in ETTInfo.groups:
    #             download_file(path, f'{ETT.source_url}/{group}.csv')
