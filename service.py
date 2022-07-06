# libs
import os
import bentoml
import numpy as np
import pandas as pd
from bentoml.io import NumpyNdarray, JSON
from pycaret.regression import load_model, predict_model


class UberFareRegressorRunnable(bentoml.Runnable):
    """https://docs.bentoml.org/en/latest/concepts/runner.html"""
    SUPPORTS_CPU_MULTI_THREADING = True
    
    def __init__(self):
        self.pipeline = load_model(
            os.path.join('models', 'xgb_regressor'), 
            verbose=False
        )
        
    
    @bentoml.Runnable.method(batchable=True)
    def predict(self, sample: np.ndarray):
        """
        This method will take a real world sample and then
        predict a fare amount to it.
        """
        # formatting the input data
        data = pd.DataFrame([sample])
        data.columns = [
            'pickup_year', 'pickup_month', 'pickup_dayofyear', 
            'pickup_dayofweek', 'pickup_hour', 'is_holiday', 
            'passenger_count', 'pickup_latitude', 'pickup_longitude', 
            'dropoff_latitude', 'dropoff_longitude', 'trip_distance_km'
        ]
        
        # making predictions
        predictions = predict_model(self.pipeline, data=data)
        predictions.Label = np.exp(predictions.Label)
        
        return {"predictions": list(predictions.Label)}
    

# Initiating Runner
uber_fare_regression_runner = bentoml.Runner(UberFareRegressorRunnable)


# Setting Up the Service
svc = bentoml.Service('uber_fare_price_regressor', runners=[uber_fare_regression_runner])

@svc.api(input=NumpyNdarray(), output=JSON())
def regressor(sample: np.ndarray):
    """
    This function takes the input values and then
    run the regressor.
    """
    predictions = uber_fare_regression_runner.predict.run(sample)
    return predictions