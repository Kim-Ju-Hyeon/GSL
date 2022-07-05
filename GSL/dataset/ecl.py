__all__ = ['ECL', 'ECLInfo', 'ECL']

# Cell
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import gdown
import numpy as np
import pandas as pd

from utils.dataset_utils import Info, time_features_from_frequency_str
from ett import process_multiple_ts

# Cell
@dataclass
class ECL:
    freq: str = 'H'
    name: str = 'ECL'
    n_ts: int = 321

# Cell
ECLInfo = Info(groups=('ECL',),
              class_groups=(ECL,))

# Cell
@dataclass
class ECL:

    source_url: str = 'https://drive.google.com/uc?id=1rUPdR7R2iWFW-LMoDdHoO2g4KgnkpFzP'

    @staticmethod
    def load(directory: str,
             cache: bool = True) -> Tuple[pd.DataFrame,
                                          Optional[pd.DataFrame],
                                          Optional[pd.DataFrame]]:
        """Downloads and loads ETT data.
        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.
        cache: bool
            If `True` saves and loads
        Notes
        -----
        [1] Returns train+val+test sets.
        """
        path = f'{directory}/ecl/datasets'
        file_cache = f'{path}/ECL.p'

        if os.path.exists(file_cache) and cache:
            df, X_df, S_df = pd.read_pickle(file_cache)

            return df, X_df, S_df


        ECL.download(directory)
        path = f'{directory}/ecl/datasets'

        y_df = pd.read_csv(f'{path}/ECL.csv')
        y_df, X_df = process_multiple_ts(y_df)

        S_df = None
        if cache:
            pd.to_pickle((y_df, X_df, S_df), file_cache)

        return y_df, X_df, S_df

    @staticmethod
    def download(directory: str) -> None:
        """Download ECL Dataset."""
        path = f'{directory}/ecl/datasets/'
        if not os.path.exists(path):
            os.makedirs(path)
            gdown.download(ECL.source_url, f'{path}/ECL.csv')
