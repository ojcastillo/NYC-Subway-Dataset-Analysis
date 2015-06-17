# -*- coding: utf-8 -*-

import numpy as np
import pandas
import scipy
import statsmodels.api as sm

def normalize_features(dataframe):
    """
    Normalize the features in the data set.
    """
    mu = dataframe.mean()
    sigma = dataframe.std()
    
    if (sigma == 0).any():
        raise Exception("One or more features had the same value for all samples, and thus could " + \
                         "not be normalized. Please do not include features with only a single value " + \
                         "in your model.")
    dataframe_normalized = (dataframe - dataframe.mean()) / dataframe.std()

    return dataframe_normalized, mu, sigma

def predictions(dataframe):
    """
    Generate predictions ENTRIESn_hourly using an Ordinary Least Squares (OLS) method
    """

    # Using rain, fog and meantempi as input features without modifications
    features = dataframe.loc[:,['fog', 'rain', 'meantempi']]

    # Generating a polynomial of degree 25 as input feature from the Hour column
    hours = dataframe.loc[:, 'Hour']
    for x in range(1,25):
        features.loc[:, 'Hour{}'.format(x)] = hours ** x

    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    
    # Values to predict: ENTRIESn_hourly
    values = dataframe['ENTRIESn_hourly']
    m = len(values)
    
    features, mu, sigma = normalize_features(features)
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Applying the Ordinary Least Squares model
    model = sm.OLS(values,features)
    modelResult = model.fit()
    predictions = np.dot(np.array(features), modelResult.params)
    return predictions, modelResult