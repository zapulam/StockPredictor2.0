'''
Functions needed to create a forecast
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


def old(data, horizon, forecast=False):
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
    # set max horizon
    if horizon and horizon > 252:
        horizon = 252

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
        dow_effects = pd.DataFrame()

    for col in data.columns:
        if col not in ['Date', 'DayOfWeek']:
            # detrend and deason column
            decomp_result = sm.tsa.seasonal_decompose(data[col], model='additive', period=252, extrapolate_trend=25, two_sided=False)

            residuals = pd.DataFrame({col: decomp_result.resid, 'DayOfWeek': data['DayOfWeek']})

            # remove day of week effects
            grouped = residuals.groupby('DayOfWeek')
            effect = residuals[col].mean() / grouped[col].mean()
            effect_df = pd.DataFrame((effect).rename("Effect")).reset_index()
            residuals = pd.merge(residuals, effect_df, on='DayOfWeek', how='inner')

            decomp[col] = residuals[col] * residuals['Effect']

            # creates trend and seasonal forecasts for next 252 days (1 trading year) for all features
            if forecast:
                # fit OLS for trend component
                model = LinearRegression()
                model.fit(np.array(data.index[-50:]).reshape(-1, 1), decomp_result.trend[-50:])
                trends[col] = model.intercept_ + model.coef_ * range(data.index[-1], data.index[-1]+horizon)

                # use moving average for seasonal component
                seasonals[col] = decomp_result.seasonal[-252:].reset_index(drop=True).rolling(14, min_periods=1).mean()[:horizon]

                # store effects for day of week
                dow_effects[col] = effect

    # drop date columns
    decomp.drop(columns=['Date', 'DayOfWeek'], inplace=True)

    # difference normalized data
    differenced = decomp.diff().iloc[1: , :]

    # remove outliers
    for col in differenced.columns:
        z_scores = np.abs(stats.zscore(differenced[col]))
        outliers = z_scores > 3
        differenced[col][outliers] = differenced[col].mean()

    # normalize input data
    mins, maxs = differenced.min(), differenced.max() 
    normalized = (differenced-mins)/(maxs-mins)

    # return decomposed data and forecasts
    if forecast:
        forecasts = {'trend': trends, 'seasonal': seasonals, 'dow_effects': dow_effects}
        return {'data': normalized, 'forecast': forecasts}

    return {'data': normalized}


def create_rnn_forecast(input, model, horizon, device):
    '''
    Creates forecast sequentially

    Inputs:
        data (pd.DataFrame) - timeseries data to forecast
        model (LSTM) - forecasting model 

    Outputs:
        pred (pd.DataFrame) - predictions for all features

    '''
    # copy data
    x = input.unsqueeze(0)

    # predictions tensor
    predictions = torch.ones(1,0,5)

    if 'cuda' in device:
        x = x.cuda()
        predictions = predictions.cuda()
    
    for _ in range(horizon):
    
        # predict one time step
        pred = torch.unsqueeze(model(x), dim=1)
        
        # append prediction
        predictions = torch.cat((predictions, pred), dim=1)

        # append predicition to input data for next forecast
        x = torch.cat((x, pred), dim=1)
        
    return pd.DataFrame(predictions.cpu().squeeze().detach().numpy(), columns=['Open', 'High', 'Low', 'Volume', 'Close'])


def create_effects_forecast(decomposition, horizon):
    '''
    Creates forecast sequentially

    Inputs:
        data (pd.DataFrame) - timeseries data to forecast
        model (LSTM) - forecasting model 

    Outputs:
        pred (pd.DataFrame) - predictions for all features

    '''
    # create dataframes to store effects forecasts
    trend_forecast = pd.DataFrame()
    seasonality_forecast = pd.DataFrame()

    for col in decomposition['trend'].columns:
        # fit OLS for trend component
        model = LinearRegression()
        model.fit(np.array(decomposition['data'].index[-50:]).reshape(-1, 1), decomposition['trend'][col][-50:])
        trend_forecast[col] = model.intercept_ + model.coef_ * range(decomposition['data'].index[-1], decomposition['data'].index[-1]+horizon)

        # use moving average for seasonal component
        seasonality_forecast[col] = decomposition['seasonality'][col][-252:].reset_index(drop=True).rolling(14, min_periods=1).mean()[:horizon]

    return {'trend_forecast': trend_forecast, 'seasonality_forecast': seasonality_forecast}


def compose(decomposition, effects_forecast, rnn_forecast, horizon):
    '''
    Composes residual forecast and effects

    Inputs:
        

    Outputs:
        

    '''
    # holidays to drop from prediciton
    holidays = pd.read_csv(r'tools/NASDAQ_Holidays.csv')['Date']

    # get dates up to 1 year out
    start_date = decomposition['data']['Date'].iloc[-1].date()
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

    # undo differencing
    forecast = decomposition['data'][['Open', 'High', 'Low', 'Volume', 'Close']].iloc[-1] + forecast[['Open', 'High', 'Low', 'Volume', 'Close']].cumsum()
    forecast = pd.concat([df, forecast], axis=1)
    print(forecast['Close'])

    # multiply dow effect
    for col in forecast.columns:
        if col not in ['Date', 'DayOfWeek']:
            forecast['Effect'] = forecast['DayOfWeek'].map(decomposition['dow_effect'][col])
            forecast[col] = forecast[col] / forecast['Effect']
            forecast.drop(columns=['Effect'], inplace=True)
    print(forecast['Close'])

    # add seasonal effect
    forecast = forecast + effects_forecast['trend_forecast']

    # add trend effect
    forecast = forecast + effects_forecast['seasonality_forecast']

    return forecast


def forecast_pipeline(data, model, horizon, device):
    '''
    Run all steps to create time series forecast 

    Inputs:
        

    Outputs:
        

    '''
    # decompose data
    decomposition = decompose(data)

    # forecast trend and seasonality
    effects_forecast = create_effects_forecast(decomposition, horizon)
    
    # create residual forecast from model
    rnn_forecast = create_rnn_forecast(torch.tensor(decomposition['input'].values).float(), model, horizon, device)

    # compose residual forecasts and effects
    composition = compose(decomposition, effects_forecast, rnn_forecast, horizon)
    
    # get close data only
    close_forecast = composition.iloc[:, 4]
    
    return close_forecast
