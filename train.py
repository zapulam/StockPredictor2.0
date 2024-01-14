'''
Trains PyTorch LSTM on S&P 500 daily price data
By: Zachary Pulliam
'''

import os
import json
import time
import torch
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from termcolor import cprint
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from rnn import LSTM
from dataset import Stock_Data
from utils import create_logs, write_logs, custom_collate


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

    tickers    = control_file['tickers']

    hidden_dim = control_file['model']['hidden']
    layers     = control_file['model']['layers']

    epochs     = control_file['training']['epochs']
    lr         = control_file['training']['lr']
    bs         = control_file['training']['bs']
    workers    = control_file['training']['workers']
    
    lags       = control_file['forecasting']['lags']
    stride     = control_file['forecasting']['stride']
    horizon    = control_file['forecasting']['horizon']
    
    device     = control_file['device']
    
    save_path   = control_file['save_path']

    # make models folder
    if not os.path.isdir('models'):
        os.mkdir('models')
        cprint("\nCheckpoint: ", "cyan", end='')
        cprint("MODELS folder created.")
    else:
        cprint("\nCheckpoint: ", "cyan", end='')
        cprint(f"MODELS folder exists.")

    # load all data locations
    files = os.listdir(data_path)

    # get intersection between tickers listed and files found
    if tickers:
        tickers = [item + ".csv" for item in tickers]
        tickers = list(set(files) & set(tickers))

    cprint("Checkpoint: ", "cyan", end='')
    cprint("S&P500 files found for training.")

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
    cprint(f"{path.upper()} created to store models and logs for this run.")

    # check if cuda is available
    if (torch.cuda.is_available()) and ('cuda' in device):
        cprint("Checkpoint: ", "cyan", end='')
        cprint(f"{device.upper()} is found.")
    elif (not torch.cuda.is_available()) and ('cuda' in device):
        cprint("Checkpoint: ", "cyan", end='')
        cprint(f"{device.upper()} is not found, device set to CPU.")
        device = 'cpu'
    else:
        cprint("Checkpoint: ", "cyan", end='')
        cprint(f"Device set to CPU.")
    cprint("")

    for i, stock in enumerate(tickers):
        cprint(f"\nStarting training for '", "cyan", end="")
        cprint(f"{stock[:-4]}", "green", end="")
        cprint(f"'", "cyan")

        # create folder for timeseries unique model
        uniq_path = os.path.join(path, stock.split('.', 1)[0])
        os.mkdir(uniq_path)
        
        # create logging file and df
        logs = create_logs(os.path.join(uniq_path, 'logs.csv'), horizon)

        # make weights folder
        os.mkdir(os.path.join(uniq_path, 'weights'))

        # load stock data
        dataset = Stock_Data(os.path.join(data_path, stock), lags, horizon, stride)
        train, val = train_test_split(dataset, test_size=0.1, random_state=42)

        # create dataloaders
        trainloader = DataLoader(dataset=train, batch_size=bs, shuffle=True, num_workers=workers, collate_fn=custom_collate)
        validloader = DataLoader(dataset=val, batch_size=bs, shuffle=True, num_workers=workers, collate_fn=custom_collate)

        # create model
        model = LSTM(input_dim=5, hidden_dim=hidden_dim, num_layers=layers, output_dim=5)
        model = model.to(device)

        # define loss and optimizer
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best = 100

        # start training epoch
        for epoch in range(epochs):
            # start time for logging
            start = time.time()

            cprint("\nEpoch: ", "cyan", end='')
            cprint(f"{epoch+1} ", "green", end='')
            cprint(f"of ", "cyan", end='')
            cprint(f"{epochs} ", "green")


            # initialize train and valid loss and accuracy logging lost
            train_losses, train_accuracies = [], []
            valid_losses, valid_accuracies = [], []

            # training
            for j, data in enumerate(tqdm(trainloader, desc='Train', ascii=True, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
                # load X and Y
                X, Y = data

                # send to CUDA
                if 'cuda' in device:
                    X = X.cuda()
                    Y = Y.cuda()

                # create predictions tensor for recursive forecasting
                predictions = torch.empty(X.size()[0], 0, 5)
                if 'cuda' in device:
                    predictions = predictions.cuda()

                # convert to floats
                X = X.float()
                Y = Y.float()

                for _ in range(horizon):
                    # predict one time step
                    prediction = torch.unsqueeze(model(X), dim=1)
                    
                    # append prediction
                    predictions = torch.cat((predictions, prediction), dim=1)

                    # append prediction to input data for next forecast
                    X = torch.cat((X, prediction), dim=1) 

                # calculate loss                
                loss = criterion(predictions, Y)   
                train_losses.append(loss.item())
                
                # checking accuracy of close on last day for each day forecasted
                for k in range(X.size()[0]):
                    train_accuracies.extend([[1 - torch.abs(predictions[k, n, 4] - Y[k, n, 4]).tolist() for n in range(horizon)]])

                # update model parameters
                optimizer.zero_grad()   
                loss.backward()         
                optimizer.step()

            # validating
            with torch.no_grad():
                for j, data in enumerate(tqdm(validloader, desc='Valid', ascii=True, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
                    # load X and Y
                    X, Y = data

                    # send to CUDA
                    if 'cuda' in device:
                        X = X.cuda()
                        Y = Y.cuda()

                    # create predictions tensor for recursive forecasting
                    predictions = torch.empty(X.size()[0], 0, 5)
                    if 'cuda' in device:
                        predictions = predictions.cuda()

                    # convert to floats
                    X = X.float()
                    Y = Y.float()

                    for _ in range(horizon):
                        # predict one time step
                        prediction = torch.unsqueeze(model(X), dim=1)
                        
                        # append prediction
                        predictions = torch.cat((predictions, prediction), dim=1)

                        # append prediction to input data for next forecast
                        X = torch.cat((X, prediction), dim=1)   

                    # calculate loss
                    loss = criterion(predictions, Y)   
                    valid_losses.append(loss.item())
                    
                    # checking accuracy of close on last day for each day forecasted
                    for k in range(X.size()[0]):
                        valid_accuracies.extend([[1 - torch.abs(predictions[k, n, 4] - Y[k, n, 4]).tolist() for n in range(horizon)]])    

            # logging time
            end = time.time()

            # save most recent model
            torch.save([model.kwargs, model.state_dict()], os.path.join(uniq_path, "weights\last.pth"))

            # calculate average metrics
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_train_accuracies = [sum(col) / len(col) for col in zip(*train_accuracies)]

            avg_valid_loss = sum(valid_losses) / len(valid_losses)
            avg_valid_accuracies = [sum(col) / len(col) for col in zip(*valid_accuracies)]

            # save best model
            if avg_valid_loss < best:
                best = avg_valid_loss
                torch.save([model.kwargs, model.state_dict()], os.path.join(uniq_path, "weights\\best.pth"))

            # logging
            logs = write_logs(os.path.join(uniq_path, 'logs.csv'), logs, epoch, end-start, avg_train_loss, avg_train_accuracies, avg_valid_loss, avg_valid_accuracies)

            # print logging
            cprint(f"Time: ", end="")
            cprint(f"{round(end-start, 3)}   ", "green", end="")
            cprint(f"Train Loss: ", end="")
            cprint(f"{round(avg_train_loss, 5)}   ", "green", end="")
            cprint(f"Train Acc@{horizon}: ", end="")
            cprint(f"{round(avg_train_accuracies[-1], 5)}   ", "green", end="")
            cprint(f"Valid Loss: ", end="")
            cprint(f"{round(avg_valid_loss, 5)}   ", "green", end="")
            cprint(f"Valid Acc@{horizon}: ", end="")
            cprint(f"{round(avg_valid_accuracies[-1], 5)}   ", "green", end="\n")

        cprint("\n")  

    cprint(f"Checkpoint: ", "cyan", end="")    
    cprint(f"Finished training; models and logging saved to: {path.upper()}\n", "green", end="\n")


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
    