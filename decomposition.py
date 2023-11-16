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
        forecast (bool) - if true will return forecasts for effects 

    Outputs:
        normalized (pd.DataFrame) - decomposed, differenced and normalized time series data
        forecasts (dict)
            - trend (pd.DataFrame) - trend forecast
            - seasonal (pd.DataFrame) - seasonal forecast
            - dow (pd.DataFrame) - effect for each day of week

    '''

    # create decomposition dictionary
    decomp = {'data': data}
    trend = pd.DataFrame()
    seasonality = pd.DataFrame()
    dow_effect = pd.DataFrame()


    # add day of week column
    data['Date'] = pd.to_datetime(data['Date'])
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    
    # copy date columns
    series = pd.DataFrame()
    series[['Date', 'DayOfWeek']] = data[['Date', 'DayOfWeek']] 

    for col in data.columns:
        if col not in ['Date', 'DayOfWeek']:
            # detrend and deason column
            trend_seasonal_decomp = sm.tsa.seasonal_decompose(data[col], model='additive', period=252, extrapolate_trend=25, two_sided=False)
            trend[col] = trend_seasonal_decomp.trend
            seasonality[col] = trend_seasonal_decomp.seasonal

            residuals = pd.DataFrame({col: trend_seasonal_decomp.resid, 'DayOfWeek': data['DayOfWeek']})

            # remove day of week effects
            grouped = residuals.groupby('DayOfWeek')
            effect = residuals[col].mean() / grouped[col].mean()
            dow_effect[col] = effect

            effect_df = pd.DataFrame((effect).rename("Effect")).reset_index()
            residuals = pd.merge(residuals, effect_df, on='DayOfWeek', how='inner')

            # assign column to decomposed time series for forecasting
            series[col] = residuals[col] * residuals['Effect']

    # add effects to dict
    decomp.update({'trend': trend, 'seasonality': seasonality, 'residuals': series, 'dow_effect': dow_effect})

    # drop date columns
    series.drop(columns=['Date', 'DayOfWeek'], inplace=True)

    # difference data
    decomp.update({'final_row': series.iloc[-1]})
    series = series.diff().iloc[1: , :]

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
