__all__ = ['Scaler', 'norm_scaler', 'inv_norm_scaler', 'norm1_scaler', 'inv_norm1_scaler', 'std_scaler',
           'inv_std_scaler', 'median_scaler', 'inv_median_scaler', 'invariant_scaler', 'inv_invariant_scaler']

# Cell
import numpy as np
import statsmodels.api as sm


class Scaler(object):
    def __init__(self, normalizer):
        assert (normalizer in ['std', 'invariant', 'norm', 'norm1', 'median']), 'Normalizer not defined'
        self.normalizer = normalizer
        self.x_shift = None
        self.x_scale = None

    def scale(self, x):
        if self.normalizer == 'invariant':
            x_scaled, x_shift, x_scale = invariant_scaler(x)
        elif self.normalizer == 'median':
            x_scaled, x_shift, x_scale = median_scaler(x)
        elif self.normalizer == 'std':
            x_scaled, x_shift, x_scale = std_scaler(x)
        elif self.normalizer == 'norm':
            x_scaled, x_shift, x_scale = norm_scaler(x)
        elif self.normalizer == 'norm1':
            x_scaled, x_shift, x_scale = norm1_scaler(x)

        nan_before_scale = np.sum(np.isnan(x))
        nan_after_scale = np.sum(np.isnan(x_scaled))
        assert nan_before_scale == nan_after_scale, 'Scaler induced nans'

        self.x_shift = x_shift
        self.x_scale = x_scale

        return np.array(x_scaled)

    def inv_scale(self, x):
        assert self.x_shift is not None
        assert self.x_scale is not None

        if self.normalizer == 'invariant':
            x_inv_scaled = inv_invariant_scaler(x, self.x_shift, self.x_scale)
        elif self.normalizer == 'median':
            x_inv_scaled = inv_median_scaler(x, self.x_shift, self.x_scale)
        elif self.normalizer == 'std':
            x_inv_scaled = inv_std_scaler(x, self.x_shift, self.x_scale)
        elif self.normalizer == 'norm':
            x_inv_scaled = inv_norm_scaler(x, self.x_shift, self.x_scale)
        elif self.normalizer == 'norm1':
            x_inv_scaled = inv_norm1_scaler(x, self.x_shift, self.x_scale)

        return np.array(x_inv_scaled)

# Norm
def norm_scaler(x):
    x_max = np.max(x)
    x_min = np.min(x)

    x = (x - x_min) / ((x_max - x_min) + 1e-9)
    return x, x_min, x_max

def inv_norm_scaler(x, x_min, x_max):
    return x * (x_max - x_min) + x_min

# Norm1
def norm1_scaler(x):
    x_max = np.max(x)
    x_min = np.min(x)

    x = (x - x_min) / ((x_max - x_min) + 1e-9)
    x = x * (2) - 1
    return x, x_min, x_max

def inv_norm1_scaler(x, x_min, x_max):
    x = (x + 1) / 2
    return x * (x_max - x_min) + x_min

# Std
def std_scaler(x):
    x_mean = np.mean(x)
    x_std = np.std(x)

    x = (x - x_mean) / x_std + 1e-9
    return x, x_mean, x_std

def inv_std_scaler(x, x_mean, x_std):
    return (x * x_std) + x_mean

# Median
def median_scaler(x):
    x_median = np.median(x)
    x_mad = sm.robust.scale.mad(x)
    if x_mad == 0:
        x_mad = np.std(x, ddof = 1) / 0.6744897501960817
    x = (x - x_median) / x_mad
    return x, x_median, x_mad

def inv_median_scaler(x, x_median, x_mad):
    return x * x_mad + x_median

# Invariant
def invariant_scaler(x):
    x_median = np.median(x)
    x_mad = sm.robust.scale.mad(x)
    if x_mad == 0:
        x_mad = np.std(x, ddof = 1) / 0.6744897501960817
    x = np.arcsinh((x - x_median) / x_mad)
    return x, x_median, x_mad

def inv_invariant_scaler(x, x_median, x_mad):
    return np.sinh(x) * x_mad + x_median
