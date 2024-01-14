'''
Functions needed to decompose a time series
By: Zachary Pulliam
'''

import torch
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import date, timedelta

from scipy import stats
from sklearn.linear_model import LinearRegression


def decompose(data):
    '''
    Decomposes time series for training and also forecasts effects if desired 

    Inputs:
        data (pd.DataFrame) - timeseries data for decomposition with n columns

    Outputs:
        decomp (pd.DataFrame) - decomposed time series data

    '''

    # create decomposition dictionary
    decomp = {'data': data}
    trend = pd.DataFrame()
    residuals = pd.DataFrame()

    # add day of week column
    data['Date'] = pd.to_datetime(data['Date'])
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    
    # copy date columns
    series = pd.DataFrame()
    series[['Date', 'DayOfWeek']] = data[['Date', 'DayOfWeek']] 

    for col in data.columns:
        if col not in ['Date', 'DayOfWeek']:
            # detrend and deason column
            trend_decomp = sm.tsa.seasonal_decompose(data[col], model='additive', period=252, extrapolate_trend=25, two_sided=False)
            trend[col] = trend_decomp.trend
            residuals[col] = trend_decomp.resid + trend_decomp.seasonal

            resids = pd.DataFrame({col: trend_decomp.resid, 'DayOfWeek': data['DayOfWeek']})

            # assign column to decomposed time series for forecasting
            series[col] = resids[col]

    # add effects to dict
    decomp.update({'trend': trend, 'residuals': residuals})

    # drop date columns
    series.drop(columns=['Date', 'DayOfWeek'], inplace=True)

    # remove outliers
    for col in series.columns:
        z_scores = np.abs(stats.zscore(series[col]))
        outliers = z_scores > 3
        series[col][outliers] = series[col].mean()

    # normalize input data
    mins, maxs = series.min(), series.max()
    decomp.update({'minimums': mins, 'maximums': maxs})

    series = (series-mins)/(maxs-mins)
    decomp.update({'input': series})

    return decomp
