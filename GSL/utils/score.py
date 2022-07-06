import numpy as np


def get_score(target, prediction, scaler=None):
    score = dict()

    if scaler is None:
        pass
    else:
        target = scaler.inv_scale(target)
        prediction = scaler.inv_scale(prediction)

    mae = MAE(target, prediction)
    mape = MAPE(target, prediction)
    rmse = RMSE(target, prediction)

    score['MAE'] = mae
    score['MAPE'] = mape
    score['RMSE'] = rmse

    return score

def MAPE(v, v_, axis=None):
    mape = (np.abs(v_ - v) / np.abs(v)+1e-5).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)

def RMSE(v, v_, axis=None):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2, axis)+1e-5).astype(np.float64)


def MAE(v, v_, axis=None):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v), axis).astype(np.float64)
