'''
Trains PyTorch LSTM on S&P 500 daily price data
By: Zachary Pulliam
'''

import os
import json
import time
import torch
import argparse

from tqdm import tqdm
from termcolor import cprint

from rnn import LSTM
from dataset import SP_500
from utils import create_logs, write_logs


def train(control_file):
    '''
    Trains a RNN on each available S&P 500 stock data to forecast daily close price.

    Inputs:
        control_file (dict) - Arguments for training.
            - hidden (int) - Number of hidden layers.
            - layers (int) - Number of recurrent layers.
            
            - data (str) - Path to prices data.
            
            - epochs (int) - Number of training epochs.
            - lr (float) - Learning rate.
            
            - lags (int) - Minimum lags range (252 = 1 year of data)
            - horizon (int) - Number of days to forecast for recursively.
            - stride (int) - Stride to walk through data for training (stride = 1 trains on each day).
            
            - device (str) - Device to use for training; cuda:n or cpu.
            
            - save_path (str) - Path to save models.
    '''

    cprint("\nSTOCKPREDICTOR2.0 TRAINING MANY MODELS SCRIPT", "cyan")

    # unload arguments
    data_path  = control_file['data_path']

    hidden_dim = control_file['model']['hidden']
    layers     = control_file['model']['layers']

    epochs     = control_file['training']['epochs']
    lr         = control_file['training']['lr']
    
    lags       = control_file['forecasting']['lags']
    stride     = control_file['forecasting']['stride']
    horizon    = control_file['forecasting']['horizon']
    
    device     = control_file['device']
    
    save_path   = control_file['save_path']

    # make models folder
    if not os.path.isdir('models'):
        os.mkdir('models')
        cprint("\nCheckpoint: ", "cyan", end='')
        cprint("MODELS folder created.", "green")
    else:
        cprint("\nCheckpoint: ", "cyan", end='')
        cprint(f"MODELS folder exists.", "green")

    # load data
    dataset = SP_500(data_path)    
    cprint("Checkpoint: ", "cyan", end='')
    cprint("S&P500 dataset created for training.", "green")

    # make unique model folder
    k, path = 2, os.path.join('models', save_path)
    while True:
        if not os.path.isdir(path):
            os.mkdir(path)
            break
        else:
            path = os.path.join('models', save_path + "_" + str(k))
            k += 1
    cprint("Checkpoint: ", "cyan", end='')
    cprint(f"{path.upper()} created to store models and logs for this run.", "green")

    # check if cuda is available
    if (torch.cuda.is_available()) and ('cuda' in device):
        cprint("Checkpoint: ", "cyan", end='')
        cprint(f"{device.upper()} is found.", "green")
    elif (not torch.cuda.is_available()) and ('cuda' in device):
        cprint("Checkpoint: ", "cyan", end='')
        cprint(f"{device.upper()} is not found, device set to CPU.", "green")
        device = 'cpu'
    else:
        cprint("Checkpoint: ", "cyan", end='')
        cprint(f"Device set to CPU.", "green")

    cprint("\nStarting training...\n", "green")

    for idx, stock in enumerate(tqdm(dataset.data, desc='Training Many Models', ascii=True, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
        # create folder for timeseries unique model
        uniq_path = os.path.join(path, stock.split('.', 1)[0])
        os.mkdir(uniq_path)
        
        # create logging file and df
        logs = create_logs(os.path.join(uniq_path, 'logs.csv'), horizon)

        # make weights folder
        os.mkdir(os.path.join(uniq_path, 'weights'))

        # load timeseries
        timeseries = dataset.__getitem__(idx)
        if 'cuda' in device:
            timeseries = timeseries.cuda()

        # create model
        model = LSTM(input_dim=5, hidden_dim=hidden_dim, num_layers=layers, output_dim=5)
        model = model.to(device)

        # define loss and optimizer
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best = 100

        for epoch in range(epochs):
            # logging
            start = time.time()

            # initialize train and valid loss and accuracy logging lost
            losses, accuracies = [], []

            # initialize X and Y list of length len(timeseries)-lags with
            X = [] # of size (n, num_feats)
            Y = [] # of size (horizon, 5)

            # create X and Y
            for i in range(lags, timeseries.shape[0]-horizon, stride):
                X.append(timeseries[0:i, :])
                Y.append(timeseries[i:i+horizon, :])

            # train model for each sequence
            for i, x in enumerate(X):

                predictions = torch.empty(0,5)
                if 'cuda' in device:
                    predictions = predictions.cuda()

                # train and validation
                x = x.float()
                y = Y[i].float()

                for _ in range(horizon):
                    # predict one time step
                    prediction = torch.unsqueeze(model(x), dim=0)
                    
                    # append prediction
                    predictions = torch.cat((predictions, prediction), dim=0)

                    # append prediction to input data for next forecast
                    x = torch.cat((x, prediction), dim=0)   

                # calculate loss
                loss = criterion(predictions, y)   
                losses.append(loss.item())
                
                # checking accuracy of close on last day for each day forecasted
                accuracies.extend([1 - torch.abs(predictions[n, 4] - y[n, 4]).tolist() for n in range(horizon)])

                # update model parameters
                optimizer.zero_grad()   
                loss.backward()         
                optimizer.step()        

            # logging time
            end = time.time()

            # save most recent model
            torch.save([model.kwargs, model.state_dict()], os.path.join(uniq_path, "weights\last.pth"))

            # calculate average metrics
            avg_loss = sum(losses) / len(losses)
            avg_accuracies = [sum(col) / len(col) for col in zip(*accuracies)]

            # save best model
            if avg_loss < best:
                best = avg_loss
                torch.save([model.kwargs, model.state_dict()], os.path.join(uniq_path, "weights\\best.pth"))

            # logging
            logs = write_logs(os.path.join(uniq_path, 'logs.csv'), logs, epoch, end-start, avg_loss, avg_accuracies)

    cprint(f"\n\nCheckpoint: ", "cyan", end="")    
    cprint(f"Finished training; models and logging saved to: {path.upper()}\n", "cyan", end="\n")


def read_control_file():
    '''
    Read control_file for training.
    
    Outputs:
        control_file (dict) - Arguments for training.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_file', type=str, default='control_file.json', help='Path to control file')
    args = parser.parse_args()

    # Read JSON data from control file and load as dictionary
    with open(args.control_file, 'r') as json_file:
        control_file = json.load(json_file)
    
    return control_file


if __name__ == "__main__":
    control_file = read_control_file()
    train(control_file)
    