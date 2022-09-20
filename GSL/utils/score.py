import numpy as np
from typing import Optional, Union


def divide_no_nan(a, b):
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    div[div != div] = 0.0
    div[div == float('inf')] = 0.0
    return div

def metric_protections(y: np.ndarray, y_hat: np.ndarray, weights: np.ndarray):
    assert (weights is None) or (np.sum(weights) > 0), 'Sum of weights cannot be 0'
    assert (weights is None) or (weights.shape == y_hat.shape), 'Wrong weight dimension'

def get_score(target, prediction, scaler=None):
    score = dict()

    if scaler is None:
        mae = MAE(target, prediction)
        mape = MAPE(target, prediction)
        rmse = RMSE(target, prediction)
        mse = MSE(target, prediction)
    else:
        target = scaler.inv_scale(target)
        prediction = scaler.inv_scale(prediction)

        mae = MAE(target, prediction)
        mape = MAPE(target, prediction)
        rmse = RMSE(target, prediction)
        mse = MSE(target, prediction)

    score['MAE'] = mae
    score['MAPE'] = mape
    score['RMSE'] = rmse
    score['MSE'] = mse

    return score

def MAPE(v, v_, axis=None):
    mape = (np.abs(v_ - v) / (np.abs(v)+1e-5))
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)*100

def RMSE(v, v_, axis=None):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2, axis)+1e-5)


def MAE(v, v_, axis=None):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v), axis)

# Cell
def MSE(y: np.ndarray, y_hat: np.ndarray,
        weights: Optional[np.ndarray] = None,
        axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculates Mean Squared Error.
    MSE measures the prediction accuracy of a
    forecasting method by calculating the squared deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.
    Parameters
    ----------
    y: numpy array
        Actual test values.
    y_hat: numpy array
        Predicted values.
    weights: numpy array
        Weights for weighted average.
    axis: None or int, optional
        Axis or axes along which to average a.
        The default, axis=None, will average over all of the
        elements of the input array.
        If axis is negative it counts
        from the last to the first axis.
    Returns
    -------
    mse: numpy array or double
        Return the mse along the specified axis.
    """
    metric_protections(y, y_hat, weights)

    delta_y = np.square(y - y_hat)
    if weights is not None:
        mse = np.average(delta_y[~np.isnan(delta_y)],
                         weights=weights[~np.isnan(delta_y)],
                         axis=axis)
    else:
        mse = np.nanmean(delta_y, axis=axis)

    return mse

def smape(y: np.ndarray, y_hat: np.ndarray,
          weights: Optional[np.ndarray] = None,
          axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculates Symmetric Mean Absolute Percentage Error.
    SMAPE measures the relative prediction accuracy of a
    forecasting method by calculating the relative deviation
    of the prediction and the true value scaled by the sum of the
    absolute values for the prediction and true value at a
    given time, then averages these devations over the length
    of the series. This allows the SMAPE to have bounds between
    0% and 200% which is desireble compared to normal MAPE that
    may be undetermined.
    Parameters
    ----------
    y: numpy array
        Actual test values.
    y_hat: numpy array
        Predicted values.
    weights: numpy array
        Weights for weighted average.
    axis: None or int, optional
        Axis or axes along which to average a.
        The default, axis=None, will average over all of the
        elements of the input array.
        If axis is negative it counts
        from the last to the first axis.
    Returns
    -------
    smape: numpy array or double
        Return the smape along the specified axis.
    """
    metric_protections(y, y_hat, weights)

    delta_y = np.abs(y - y_hat)
    scale = np.abs(y) + np.abs(y_hat)
    smape = divide_no_nan(delta_y, scale)
    smape = 200 * np.average(smape, weights=weights, axis=axis)

    if isinstance(smape, float):
        assert smape <= 200, 'SMAPE should be lower than 200'
    else:
        assert all(smape <= 200), 'SMAPE should be lower than 200'

    return smape

# Cell
def mase(y: np.ndarray, y_hat: np.ndarray,
         y_train: np.ndarray,
         seasonality: int,
         weights: Optional[np.ndarray] = None,
         axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculates the Mean Absolute Scaled Error.
    MASE measures the relative prediction accuracy of a
    forecasting method by comparinng the mean absolute errors
    of the prediction and the true value against the mean
    absolute errors of the seasonal naive model.
    Parameters
    ----------
    y: numpy array
        Actual test values.
    y_hat: numpy array
        Predicted values.
    y_train: numpy array
        Actual insample values for Seasonal Naive predictions.
    seasonality: int
        Main frequency of the time series
        Hourly 24,  Daily 7, Weekly 52,
        Monthly 12, Quarterly 4, Yearly 1.
    weights: numpy array
        Weights for weighted average.
    axis: None or int, optional
        Axis or axes along which to average a.
        The default, axis=None, will average over all of the
        elements of the input array.
        If axis is negative it counts
        from the last to the first axis.
    Returns
    -------
    mase: numpy array or double
        Return the mase along the specified axis.
    References
    ----------
    [1] https://robjhyndman.com/papers/mase.pdf
    """
    delta_y = np.abs(y - y_hat)
    delta_y = np.average(delta_y, weights=weights, axis=axis)

    scale = np.abs(y_train[:-seasonality] - y_train[seasonality:])
    scale = np.average(scale, axis=axis)

    mase = delta_y / scale

    return mase

