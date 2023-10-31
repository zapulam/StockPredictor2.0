'''
Creates forecast for a single stock time series
By: Zachary Pulliam
'''

import os
import torch
import numpy as np
import pandas as pd

from decomposition import decompose
from composition import compose


def create_forecast(data, model, horizon):
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


def forecast_pipeline(data, model, horizon):
    # decompose data
    input, effects_forecasts = decompose(data, forecast=True)
    
    # create residual forecast from model
    residual_forecast = create_forecast(input, model, horizon)
    
    # compose
    composition = compose(residual_forecast, effects_forecasts)
    
    # get close data only
    close = composition.iloc[:, 4]
    
    return close
