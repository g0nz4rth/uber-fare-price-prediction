import os
import numpy as np
import pandas as pd
from pycaret.regression import *


# loading data
df = pd.read_csv(os.path.join('..', '..', 'data', 'interim', 'processed_uber_trip_tax_data.csv'))

# target column log transformation
df.fare_amount = np.log(df.fare_amount)

# training setup
reg_exp = setup(
    data=df,
    target='fare_amount',
    train_size=0.8,
    categorical_features=['pickup_year', 'pickup_dayofweek', 'is_holiday'],
    numeric_features=['pickup_month', 'pickup_dayofyear', 'pickup_hour', 
                      'passenger_count', 'pickup_latitude', 'pickup_longitude',
                      'dropoff_latitude', 'dropoff_longitude', 'trip_distance_km'],
    normalize=True,
    ignore_low_variance=True,
    silent=True,
    html=False
)

# model creating
model = create_model('xgboost', verbose=False)
model = finalize_model(model)

# model saving
save_model(finalized_regressor, os.path.join('..', '..', 'models', 'xgb_regressor'))