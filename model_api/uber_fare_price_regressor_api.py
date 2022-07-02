
import numpy as np
import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model('../model_api/uber_fare_price_regressor_api')

# Define predict function
@app.post('/predict')
def predict(
    pickup_year, 
    pickup_month, 
    pickup_dayofyear, 
    pickup_dayofweek, 
    pickup_hour, 
    is_holiday, 
    passenger_count, 
    pickup_latitude, 
    pickup_longitude, 
    dropoff_latitude, 
    dropoff_longitude, 
    trip_distance_km
):
    data = pd.DataFrame(
        [
            [
                pickup_year, pickup_month, pickup_dayofyear, pickup_dayofweek, 
                pickup_hour, is_holiday, passenger_count, pickup_latitude, 
                pickup_longitude, dropoff_latitude, dropoff_longitude, trip_distance_km
            ]
        ]
    )
    
    data.columns = [
        'pickup_year', 'pickup_month', 'pickup_dayofyear', 'pickup_dayofweek', 
        'pickup_hour', 'is_holiday', 'passenger_count', 'pickup_latitude', 
        'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'trip_distance_km'
    ]
    
    predictions = predict_model(model, data=data) 
    predictions.Label = np.exp(predictions.Label)
    
    return {'prediction': list(predictions['Label'])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
