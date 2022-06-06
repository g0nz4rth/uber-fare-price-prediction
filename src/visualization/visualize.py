import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def na_impact_analysis(data: pd.DataFrame, column: str, figsize: tuple = (20, 6)):
    """
    This function implements a plot to analyze the impact of missing data
    from a given column on the target variable.
    """
    # copying the original dataset
    df = data.copy()
    
    # filtering missing values
    df[column] = np.where(df[column].isnull(), 1, 0)
    
    # comparing the mean fuel consumption 
    # for both missing and not missing groups
    tmp = df.groupby(column)['FUEL_CONSUMPTION'].agg(['mean', 'std'])
    
    # making the plot
    tmp.plot(
        kind='barh', 
        y='mean', 
        legend=False,
        xerr='std',
        title='Fuel Consumption',
        color='blue',
        figsize=figsize
    )
    
    plt.show()