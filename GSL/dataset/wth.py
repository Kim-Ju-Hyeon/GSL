__all__ = ['WTH', 'WTHInfo', 'WTH']

# Cell
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import gdown
import numpy as np
import pandas as pd

from .utils import Info, time_features_from_frequency_str
from .ett import process_multiple_ts

# Cell
@dataclass
class WTH:
    freq: str = 'H'
    name: str = 'WTH'
    n_ts: int = 12

# Cell
WTHInfo = Info(groups=('WTH',),
              class_groups=(WTH,))

# Cell
@dataclass
class WTH:

    source_url: str = 'https://drive.google.com/uc?id=1UBRz-aM_57i_KCC-iaSWoKDPTGGv6EaG'

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
        path = f'{directory}/wth/datasets'
        file_cache = f'{path}/WTH.p'

        if os.path.exists(file_cache) and cache:
            df, X_df, S_df = pd.read_pickle(file_cache)

            return df, X_df, S_df


        WTH.download(directory)
        path = f'{directory}/wth/datasets'

        y_df = pd.read_csv(f'{path}/WTH.csv')
        y_df, X_df = process_multiple_ts(y_df)

        S_df = None
        if cache:
            pd.to_pickle((y_df, X_df, S_df), file_cache)

        return y_df, X_df, S_df

    @staticmethod
    def download(directory: str) -> None:
        """Download WTH Dataset."""
        path = f'{directory}/wth/datasets/'
        if not os.path.exists(path):
            os.makedirs(path)
            gdown.download(WTH.source_url, f'{path}/WTH.csv')