'''
Purpose: predicts stock prices n days in the future for all S&P 500 stocks
'''

import os
import sys
import torch
import requests
import argparse
import pandas as pd
import datetime as dt

from datetime import date
from dateutil.relativedelta import relativedelta

from rnn import LSTM
from tools.get_data import download_data


def predict(args):
    '''
    Predicts stock prices n days in the future for all S&P 500 stocks and saves predictions to specified location
    
    Inputs:
        args (dict) - arguments passed in via argparser
            - weights (str) - path to model weights
            - skip (bool) - skip most recent daily data download
            - steps (int) - future time steps to predict
            - device (str) - device to use for prediction
            - savepath (str) - path to save predictions
    '''
    weights, skip, steps, device, savepath = \
        args.weights, args.skip, args.steps, args.device, args.savepath

    # Get list of stock tickers
    df = pd.read_csv('tools/S&P500-Info.csv')
    symbols = df['Symbol'].tolist()
    
    # Download most recent daily prices data before making predictions
    if not skip:
        download_data('daily_prices', 'tools/S&P500-Info.csv')

    # Create unique folder for predictions
    k, newpath = 2, 'forecasts/' + savepath
    while True:
        if not os.path.isdir(newpath):
            os.mkdir(newpath)
            break
        else:
            newpath = 'predictions/' + savepath + "_" + str(k)
            k += 1

    print(f"\n--> Created folder \"{newpath}\"")

    # Load model
    kwargs, state = torch.load(weights, map_location=torch.device(device))
    model = LSTM(**kwargs)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(f"--> Model loaded from \"{weights}\"\n")

    # Create predictions for desired stocks
    for _, stock in enumerate(symbols):
        stock = stock.replace('.', '-')
        sys.stdout.write('\rPredicting prices for: %s' % stock.ljust(4))

        predictions = torch.rand(1,0,5)   # tensor to store future predictions
        if 'cuda' in device:
            predictions = predictions.cuda()

        # Load input data
        data = pd.read_csv('daily_prices/' + stock + '.csv', index_col=0)
        x = data[['Open', 'High', 'Low', 'Volume', 'Close']] 

        mins, maxs = x.min(), x.max()   # values for normalization

        # Normalize input data
        x = (x-mins)/(maxs-mins)
        x = torch.tensor(x.values)
        x = torch.unsqueeze(x, dim=0)
        
        mins = torch.tensor(mins.values)
        maxs = torch.tensor(maxs.values)

        if 'cuda' in device:
            x, mins, maxs = x.cuda(), mins.cuda(), maxs.cuda()

        for _ in range(steps):
            pred = model(x.float())   # model prediction for one time step
            pred = torch.unsqueeze(pred, dim=0)
            
            predictions = torch.cat((predictions, pred), dim=1)   # append predicition to full predictions tensor

            x = torch.cat((x, pred), dim=1)   # append predicition to input data for next time step

        # Un-normalize input data and save
        predictions = predictions*(maxs-mins)+mins
        predictions = pd.DataFrame(predictions.cpu().squeeze().detach().numpy(), columns=['Open', 'High', 'Low', 'Volume', 'Close'])
        predictions.to_csv(os.path.join(newpath, stock + '.csv'), index = False)

        sys.stdout.write('\rPredicting prices for: %s - DONE' % stock.ljust(5))

    sys.stdout.flush()
    print(f"\nAll predictions saved to \"{newpath}\"")


def parse_args():
    '''
    Saves cmd line arguments for training
    
    Outputs:
        args (dict) - cmd line aruments for training
            - weights (str) - path to model weights
            - skip (bool) - skip most recent daily data download
            - steps (int) - future time steps to predict
            - device (str) - device to use for prediction
            - savepath (str) - path to save predictions
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/rnn/weights/best.pth', help='Path to model weights')

    parser.add_argument('--skip', action='store_true', help='Skip most recent daily data download')

    parser.add_argument('--steps', type=int, default=25, help='Future time steps to predict')

    parser.add_argument('--device', type=str, default='cuda:0', help='device; cuda:n or cpu')

    parser.add_argument('--savepath', type=str, default=str(date.today()), help='Path to save predictions')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    data = pd.read_csv('utils/S&P500-Info.csv', index_col=0)
    symbols = data['Symbol'].to_list()

    args = parse_args()
    predict(args)
    