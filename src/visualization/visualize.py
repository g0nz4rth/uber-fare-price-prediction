# libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def model_performance_indicators(y_true: np.ndarray, y_pred: np.ndarray, on_return: bool = False):
    """
    This function calculates three indicators of performance for a regressor,
    they are: MAE, RMSE and MAPE.
    """
    MAE = round(mean_absolute_error(y_true, y_pred), 3)
    RMSE = round(np.sqrt(mean_squared_error(y_true, y_pred)), 3)
    MAPE = round(np.mean(np.abs(((y_true - y_pred) / y_true))), 3) * 100
    SMAPE = round(np.mean( np.abs(y_pred - y_true) / ((y_true + y_pred)/2) ), 3) * 100
    
    if on_return:
        return MAE, RMSE, MAPE, SMAPE
    
    print(f"MAE: {MAE} | RMSE: {RMSE} | MAPE: {MAPE} | SMAPE: {SMAPE}")