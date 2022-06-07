# -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv

# custom dependencies
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

seed = np.random.seed(42)


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


def make_data_for_knn(data: pd.DataFrame, ignore: list, target: str):
    """
    This function takes the original dataset and then makes it ready
    for modeling with the k-NN algorithm.
    """
    # dropping useless columns
    tmp_df = data.copy()
    tmp_df = tmp_df.drop(ignore, axis=1)
    
    # dropping nans
    tmp_df = tmp_df.dropna()
    
    # handling categorical data
    tmp_df = pd.get_dummies(tmp_df)
    
    # splitting predictors-targets
    predictors = tmp_df.drop([target], axis=1)
    targets = tmp_df[target]
    
    # splitting the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        predictors, 
        targets, 
        test_size = 0.3, 
        shuffle=True,
        random_state=seed
    )
    
    # rescaling features
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=predictors.columns)
    x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=predictors.columns)
    
    return (x_train, x_test, y_train, y_test)


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()
