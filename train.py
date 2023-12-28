'''
Trains PyTorch LSTM on S&P 500 daily price data
By: Zachary Pulliam
'''

import os
import sys
import time
import torch
import argparse

from tqdm import tqdm
from termcolor import colored, cprint
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from rnn import LSTM
from dataset import SP_500



def train(args):
    '''
    Trains RNN on S&P 500 daily stock prices data to predict future close price and saves model to specified path

    Inputs:
        args (dict) - CMD line arguments for training
            - hidden (int) - Number of hidden layers.
            - layers (int) - Number of recurrent layers.
            
            - data (str) - Path to prices data.
            
            - epochs (int) - Number of training epochs.
            - lr (float) - Learning rate.
            - workers (int) - Number of worker nodes.
            
            - lags (int) - Minimum lags range (252 = 1 year of data)
            - horizon (int) - Number of days to forecast for recursively.
            - stride (int) - Stride to walk through data for training (stride = 1 trains on each day).
            
            - device (str) - Device to use for training; cuda:n or cpu.
            
            - savepath (str) - Path to save models.
    '''
    hidden_dim, num_layers, folder, epochs, \
        lr, workers, lags, stride, horizon, device, savepath = \
        args.hidden, args.layers, args.data, args.epochs, \
        args.lr, args.workers, args.lags, args.stride, args.horizon, args.device, args.savepath
    
    cprint("\nSTOCKPREDICTOR2.0 TRAINING MANY MODELS SCRIPT", "cyan")

    # make models folder
    if not os.path.isdir('models'):
        os.mkdir('models')
        cprint("\nCheckpoint: ", "cyan", end='')
        cprint("MODELS folder created.", "green")
    else:
        cprint("\nCheckpoint: ", "cyan", end='')
        cprint(f"MODELS folder exists.", "green")

    # load data
    dataset = SP_500(folder)    
    cprint("Checkpoint: ", "cyan", end='')
    cprint("S&P500 dataset created for training.", "green")

    # make unique model folder
    k, path = 2, os.path.join('models', savepath)
    while True:
        if not os.path.isdir(path):
            os.mkdir(path)
            break
        else:
            path = os.path.join('models', savepath + "_" + str(k))
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
        
        # create logging file
        with open(os.path.join(uniq_path, 'logs.txt'), 'a') as f:
            f.write('Time, Avg_Loss, Avg_Accuracy\n')

        # make weights folder
        os.mkdir(os.path.join(uniq_path, 'weights'))

        # load timeseries
        timeseries = dataset.__getitem__(idx)
        if 'cuda' in device:
            timeseries = timeseries.cuda()

        # create model
        model = LSTM(input_dim=5, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=5)
        model = model.to(device)

        # define loss and optimizer
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best = 100

        for _ in range(epochs):
            # logging
            start = time.time()

            # initialize train and valid loss and accuracy logging lost
            loss, acc = [], []

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
                l = criterion(predictions, y)   
                loss.append(l.item())
                
                # checking accuracy of close on last day... horizon days 
                # CHANGE TO TRACK EACH DAY INDIVIDUALLY
                #acc.extend([1 - torch.abs(predictions[n, 4] - y[n, 4]).tolist() for n in range(len(horizon))])
                acc.append((1 - torch.abs(predictions[-1, 4] - y[-1, 4]).item()))

                # update model parameters
                optimizer.zero_grad()   
                l.backward()         
                optimizer.step()        

            # logging
            end = time.time()

            # save most recent model
            torch.save([model.kwargs, model.state_dict()], os.path.join(uniq_path, "weights\last.pth"))

            # calculate average metrics
            avg_loss = sum(loss) / len(loss)
            avg_acc = sum(acc) / len(acc)

            # save best model
            if avg_loss < best:
                best = avg_loss
                torch.save([model.kwargs, model.state_dict()], os.path.join(uniq_path, "weights\\best.pth"))

            # logging
            with open(os.path.join(uniq_path, 'logs.txt'), 'a') as f:
                f.write(f"{round(end-start, 3)}, {round(avg_loss, 5)}, {round(avg_acc, 5)}\n")

    cprint(f"\n\nCheckpoint: ", "cyan", end="")    
    cprint(f"Finished training; models and logging saved to: {path.upper()}\n", "cyan", end="\n")


def parse_args():
    '''
    Saves cmd line arguments for training
    
    Outputs:
        args (dict) - CMD line arguments for training
            - hidden (int) - Number of hidden layers.
            - layers (int) - Number of recurrent layers.
            
            - data (str) - Path to prices data.
            
            - epochs (int) - Number of training epochs.
            - lr (float) - Learning rate.
            - workers (int) - Number of worker nodes.
            
            - lags (int) - Minimum lags range (252 = 1 year of data)
            - horizon (int) - Number of days to forecast for recursively.
            - stride (int) - Stride to walk through data for training (stride = 1 trains on each day).
            
            - device (str) - Device to use for training; cuda:n or cpu.
            
            - savepath (str) - Path to save models.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=4, help='Number of hidden layers.')
    parser.add_argument('--layers', type=int, default=2, help='Number of recurrent layers')

    parser.add_argument('--data', type=str, default='training_data', help='Path to prices data')

    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')

    parser.add_argument('--lags', type=int, default=252, help='Minimum lags range (252 = 1 year of data)')
    parser.add_argument('--horizon', type=int, default=5, help='Number of days to forecast for recursively.')
    parser.add_argument('--stride', type=int, default=32, help='Stride to walk through data for training (stride = 1 trains on each day).')

    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training; cuda:n or cpu.')

    parser.add_argument('--savepath', type=str, default='rnn', help='Path to save models.')

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
    