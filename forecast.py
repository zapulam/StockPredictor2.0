'''
Functions needed to create a forecast for a time series
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

from decomposition import decompose


def create_rnn_forecast(input, model, horizon, device):
    '''
    Creates forecast sequentially

    Inputs:
        input (pd.DataFrame) - decomposed timeseries data to forecast
        model (LSTM) - forecasting model
        horizon (int) - forecast horizon; max 252
        device (str) - hardware device (cpu, cuda:0, etc.)

    Outputs:
        rnn_forecast (pd.DataFrame) - residual forecasts for all features

    '''
    # copy data
    x = input.unsqueeze(0)

    # predictions tensor
    rnn_forecast = torch.ones(1,0,5)

    if 'cuda' in device:
        x = x.cuda()
        rnn_forecast = rnn_forecast.cuda()
    
    for _ in range(horizon):
    
        # predict one time step
        pred = torch.unsqueeze(model(x), dim=1)
        
        # append prediction
        rnn_forecast = torch.cat((rnn_forecast, pred), dim=1)

        # append predicition to input data for next forecast
        x = torch.cat((x, pred), dim=1)

    rnn_forecast = pd.DataFrame(rnn_forecast.cpu().squeeze().detach().numpy(), columns=['Open', 'High', 'Low', 'Volume', 'Close'])
        
    return rnn_forecast


def create_effects_forecast(decomposition, horizon):
    '''
    Creates forecast sequentially

    Inputs:
        decomposition (pd.DataFrame) - decomposed timeseries data to forecast
        horizon (int) - forecast horizon; max 252

    Outputs:
        effects_forecast (dict) - dictionaryu containing pd.DataFrames trend_forecast and seasonality_forecast

    '''
    # create dataframes to store effects forecasts
    trend_forecast = pd.DataFrame()
    seasonality_forecast = pd.DataFrame()

    # create forecast for each column
    for col in decomposition['trend'].columns:
        # fit OLS for trend component
        model = LinearRegression()
        model.fit(np.array(decomposition['data'].index[-20:]).reshape(-1, 1), decomposition['trend'][col][-20:])
        trend_forecast[col] = model.intercept_ + model.coef_ * range(decomposition['data'].index[-1]+1, decomposition['data'].index[-1]+horizon+1)
        trend_forecast[col] = trend_forecast[col] + (decomposition['trend'][col].iloc[-1] - trend_forecast[col].iloc[0] + model.coef_)

        # add seasonal component
        #seasonality_forecast[col] = decomposition['seasonality'][col][-252:-(252-horizon)]
        seasonality_forecast[col] = decomposition['seasonality'][col].rolling(window=10, min_periods=1).mean()[-252:-(252-horizon)]

    # store effects forecasts
    effects_forecast = {'trend_forecast': trend_forecast, 'seasonality_forecast': seasonality_forecast.reset_index()}

    return effects_forecast


def compose(decomposition, effects_forecast, rnn_forecast, horizon):
    '''
    Composes residual forecast and effects

    Inputs:
        decomposition (pd.DataFrame) - decomposed timeseries data to forecast
        effects_forecast (dict) - dictionaryu containing pd.DataFrames trend_forecast and seasonality_forecast
        rnn_forecast (pd.DataFrame) - residual forecasts for all features
        horizon (int) - forecast horizon; max 252
        
    Outputs:
        forecast (pd.DataFrame) - composed forecasts for all features

    '''
    # holidays to drop from prediciton
    holidays = pd.read_csv(r'tools/NASDAQ_Holidays.csv')['Date']

    # get dates up to 1 year out
    start_date = decomposition['data']['Date'].iloc[-1].date() + timedelta(days=1)
    next_year = start_date + timedelta(days=365)
    date_range = pd.date_range(start=start_date, end=next_year, freq='D')

    # create a DataFrame with a date and dow column
    df = pd.DataFrame({'Date': date_range})
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    df = df[df['DayOfWeek'].isin([0,1,2,3,4])]
    df = df[~df['Date'].isin(holidays)]
    df = df.reset_index(drop=True).iloc[:horizon+1]
    df = df.iloc[:horizon]

    # undo normalization
    forecast = (rnn_forecast * (decomposition['maximums'] - decomposition['minimums'])) + decomposition['minimums']

    # undo differencing, reconstructing the values at each time step for multiple columns
    reconstructed_values = {}
    for col in decomposition['final_row'].columns:
        initial_value = decomposition['final_row'][col][0]  # Initial value at time step 0
        
        # Calculate cumulative sum of differences and add it to the initial value
        reconstructed_values[col] = [initial_value] + (forecast[col].cumsum() + initial_value).tolist()

    # creating a new DataFrame with reconstructed values for multiple columns
    forecast = pd.DataFrame(reconstructed_values).iloc[1:].reset_index(drop=True)
    forecast = pd.concat([df, forecast], axis=1)

    # multiply dow effect
    for col in forecast.columns:
        if col not in ['Date', 'DayOfWeek']:
            forecast['Effect'] = forecast['DayOfWeek'].map(decomposition['dow_effect'][col])
            forecast[col] = forecast[col] / forecast['Effect']
            forecast.drop(columns=['Effect'], inplace=True)

    # add seasonal effect
    comp = forecast + effects_forecast['seasonality_forecast']

    # add trend effect
    comp = forecast + effects_forecast['trend_forecast']

    # reorder columns
    comp['Date'] = forecast['Date']
    forecast = comp.drop(columns=['DayOfWeek'])
    forecast = forecast[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']]

    return forecast


def forecast_pipeline(data, model, horizon, device):
    '''
    Run all steps to create time series forecast 

    Inputs:
        data (pd.DataFrame) - timeseries data to forecast
        model (LSTM) - forecasting model
        horizon (int) - forecast horizon; max 252
        device (str) - hardware device (cpu, cuda:0, etc.)
        

    Outputs:
        forecast (pd.DataFrame) - composed forecasts for all features

    '''
    # decompose data
    decomposition = decompose(data)

    # forecast trend and seasonality
    effects_forecast = create_effects_forecast(decomposition, horizon)
    
    # create residual forecast from model
    rnn_forecast = create_rnn_forecast(torch.tensor(decomposition['input'].values).float(), model, horizon, device)

    # compose residual forecasts and effects
    forecast = compose(decomposition, effects_forecast, rnn_forecast, horizon)
    
    return forecast
