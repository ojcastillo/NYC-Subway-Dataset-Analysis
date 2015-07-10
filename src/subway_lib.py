# -*- coding: utf-8 -*-

import numpy as np
import pandas
import scipy
import statsmodels.api as sm

from ggplot import *

""" Plot functions """

def plot_ridership_histogram(turnstile_weather):
    '''
    Histograms of ridership for rainy and non-rainy days (limited at 2000 hourly entries)
    '''
    plot = ggplot(aes(x='ENTRIESn_hourly', fill='rain', color='rain'), data=turnstile_weather) + \
            geom_histogram(binwidth=50, alpha=0.6) + xlim(0, 2000) + xlab("Entries per hour") + \
            ylab("Frequency") + ggtitle("Histograms of ridership (limited at 2000 hourly entries)")
    return plot

def plot_mean_hourly_ridership(turnstile_weather):
    '''
    Average amount of riders for each hour of the day
    '''
    df = turnstile_weather
    data = {'Hour': [], 'ENTRIESn_hourly_mean': []}
    for hh in range(0,25):
        data['Hour'].append(hh)
        data['ENTRIESn_hourly_mean'].append(df.loc[df.Hour == hh, 'ENTRIESn_hourly'].mean())
    df_means = pandas.DataFrame(data)
    plot = ggplot(df_means, aes(x='Hour', y='ENTRIESn_hourly_mean')) + geom_point() + \
            geom_line() + xlim(-1, 25) + xlab("Hour") + ylab("Average amount of riders") + \
            ggtitle("Average amount of riders per hour")
    return plot

""" Prediction functions """

def normalize_features(turnstile_weather):
    """
    Normalize the features in the data set.
    """
    mu = turnstile_weather.mean()
    sigma = turnstile_weather.std()
    
    if (sigma == 0).any():
        raise Exception("One or more features had the same value for all samples, and thus could " + \
                         "not be normalized. Please do not include features with only a single value " + \
                         "in your model.")
    turnstile_weather_normalized = (turnstile_weather - turnstile_weather.mean()) / turnstile_weather.std()

    return turnstile_weather_normalized, mu, sigma

def ols_predictions(turnstile_weather):
    """
    Generate predictions for ENTRIESn_hourly using an Ordinary Least Squares (OLS) method
    """

    # Using rain, fog and meantempi as input features without modifications
    features = turnstile_weather.loc[:,['fog', 'rain', 'meantempi']]

    # Generating a polynomial of degree 25 as input feature from the Hour column
    hours = turnstile_weather.loc[:, 'Hour']
    for x in range(1,25):
        features.loc[:, 'Hour{}'.format(x)] = hours ** x

    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(turnstile_weather['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    
    # Values to predict: ENTRIESn_hourly
    values = turnstile_weather['ENTRIESn_hourly']
    m = len(values)
    
    features, mu, sigma = normalize_features(features)
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Applying the Ordinary Least Squares model
    model = sm.OLS(values,features)
    modelResult = model.fit()
    predictions = np.dot(np.array(features), modelResult.params)
    return predictions, modelResult