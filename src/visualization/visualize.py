import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance

def calc_feature_importance(df: pd.DataFrame, target: str):
    """
    This function uses XGBRegressor to calculate feature importances
    on a given dataset.
    """
    X = df.drop(columns=[target])
    Y = df[target]
    
    model = XGBRegressor()
    model.fit(X, Y)
    
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_importance(model, ax=ax)
    plt.show()