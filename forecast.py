'''
Functions needed to create a forecast
By: Zachary Pulliam
'''

import torch
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy import stats
from sklearn.linear_model import LinearRegression


def decompose(data, forecast=False):
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
    # add day of week column
    data['Date'] = pd.to_datetime(data['Date'])
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    
    # copy date columns
    decomp = pd.DataFrame()
    decomp[['Date', 'DayOfWeek']] = data[['Date', 'DayOfWeek']] 

    # detrend, deseason, and remove day of week effect from all columns
    if forecast:
        trends = pd.DataFrame()
        seasonals = pd.DataFrame()
        dow_effects = pd.DataFrame({'DayOfWeek': [(i + datetime.date.today().weekday()) % 6 for i in range(252)]})

    for col in data.columns:
        if col not in ['Date', 'DayOfWeek']:
            # detrend and deason column
            result = sm.tsa.seasonal_decompose(data[col], model='additive', period=252, extrapolate_trend=25, two_sided=False)
            decomp[col] = result.resid

            # get day of week effects
            effect = pd.DataFrame(decomp[col] .mean() / decomp.groupby('DayOfWeek')[col].mean().rename("Effect")).reset_index()
            decomp = pd.merge(decomp, effect, on='DayOfWeek', how='inner')
            decomp[col] = decomp[col] * decomp['Effect']
            decomp.drop(columns=['Effect'], inplace=True)

            # create forecasts for next 252 days (1 trading year)
            if forecast:
                # fit OLS for trend component
                model = LinearRegression()
                model.fit(np.array(data.index[-50:]).reshape(-1, 1), result.trend[-50:])
                trends[col] = model.intercept_ + model.coef_ * range(data.index[-1], data.index[-1]+252)

                # use moving avergae for seasonal component
                seasonals[col] = result.seasonal[-252:].reset_index(drop=True).rolling(14, min_periods=1).mean()

                # create effects for day of week
                mapping_dict = effect.set_index('DayOfWeek')['Effect'].to_dict()
                dow_effects[col] = effect['DayOfWeek'].map(mapping_dict)

    # drop date columns
    decomp.drop(columns=['Date', 'DayOfWeek'], inplace=True)

    # difference normalized data
    differenced = decomp.diff().iloc[1: , :]

    # normalize input data
    mins, maxs = differenced.min(), differenced.max() 
    normalized = (differenced-mins)/(maxs-mins)

    # remove outliers
    for col in normalized.columns:
        z_scores = np.abs(stats.zscore(normalized[col]))
        outliers = z_scores > 3
        normalized[col][outliers] = normalized[col].mean()

    # return decomposed data and forecasts
    if forecast:
        forecasts = {'trend': trends, 'seasonal': seasonals, 'dow_effects': dow_effects}
        return [normalized, forecasts]

    return [normalized]


def create_forecast(data, model, horizon):
    '''
    Creates forecast sequentially

    Inputs:
        data (pd.DataFrame) - timeseries data to forecast
        model (LSTM) - forecasting model 

    Outputs:
        pred (pd.DataFrame) - predictions for all features

    '''
    # copy data
    x = data
    
    for _ in range(horizon):
        # predict one time step
        pred = torch.unsqueeze(model(data), dim=1)
        
        # append prediction
        predictions = torch.cat((predictions, pred), dim=1)

        # append predicition to input data for next forecast
        x = torch.cat((x, pred), dim=1)
        
    return pred


def compose(residual, effects):
    '''
    Composes residual forecast and effects

    Inputs:
        

    Outputs:
        

    '''
    pass


def forecast_pipeline(data, model, horizon):
    '''
    Run all steps to create time series forecast 

    Inputs:
        

    Outputs:
        

    '''
    # decompose data
    input, effects_forecasts = decompose(data, forecast=True)
    
    # create residual forecast from model
    residual_forecast = create_forecast(input, model, horizon)
    
    # compose residual forecasts and effects
    composition = compose(residual_forecast, effects_forecasts)
    
    # get close data only
    close = composition.iloc[:, 4]
    
    return close
