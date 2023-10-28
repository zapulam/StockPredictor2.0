'''
Purpose: trains PyTorch LSTM on S&P 500 daily price data
'''

import os
import time
import torch
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from rnn import LSTM
from dataset import SP_500



def train(args):
    '''
    Trains RNN on S&P 500 daily stock prices data to predict future close price and saves model to specified path

    Inputs:
        args (dict) - CMD line aruments for training
            - hidden (int) - Number of hidden layers.
            - layers (int) - Number of recurrent layers.
            
            - data (str) - Path to prices data.
            
            - epochs (int) - Number of training epochs.
            - lr (float) - Learning rate.
            - bs (int) - Batch size.
            - workers (int) - Number of worker nodes.
            
            - lookback (int) - Minimum lookback range (252 = 1 year of data)
            - horizon (int) - Number of days to forecast for recursively.
            - stride (int) - Stride to walk through data for training (stride = 1 trains on each day).
            
            - device (str) - Device to use for trainng; cuda:n or cpu.
            
            - savepath (str) - Path to save models.
    '''
    hidden_dim, num_layers, folder, epochs, \
        lr, bs, workers, lookback, stride, horizon, device, savepath = \
        args.hidden, args.layers, args.data, args.epochs, \
        args.lr, args.bs, args.workers, args.lookback, args.stride, args.horizon, args.device, args.savepath

    # make models folder
    if not os.path.isdir('models'):
        os.mkdir('models')

    # make unique model folder
    k, newpath = 2, 'models/' + savepath
    while True:
        if not os.path.isdir(newpath):
            os.mkdir(newpath)
            break
        else:
            newpath = 'models/' + savepath + "_" + str(k)
            k += 1

    # make weights folder
    os.mkdir(os.path.join(newpath, 'weights'))

    print(f"\n--> Created folder \"{newpath}\"")

    # create logging file
    with open(os.path.join(newpath, 'logs.txt'), 'a') as f:
        f.write('Time, TrainLoss, TrainAcc, ValidLoss, ValidAcc\n')

    # load data
    dataset = SP_500(folder)
    train, val = train_test_split(dataset, test_size=0.1, random_state=42)

    # create dataloaders
    trainloader = DataLoader(dataset=train, batch_size=bs, shuffle=True, num_workers=workers)
    valloader = DataLoader(dataset=val, batch_size=bs, shuffle=True, num_workers=workers)

    print("--> Dataloaders created for training and validating")

    # create model
    model = LSTM(input_dim=5, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=5)
    model = model.to(device)

    print("--> Model created and sent to device")

    # define loss and optimizer
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    best = 100

    print("\nBeginning training...")

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1} of {epochs}")
        start = time.time()

        # initialize train and valid loss and accuracy logging lost
        t_loss, t_acc = [], [] 
        v_loss, v_acc = [], []

        # Training
        for _, data in enumerate(tqdm(trainloader, desc='Training', ascii=True, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
            inputs = data[0]

            if 'cuda' in device:
                inputs = inputs.cuda()

            # initialize seqs list of length len(series)-lookback with
            #   X of size (bs, n, num_feats) and Y of size (bs, 1, numfeats)
            seqs = []

            # Create seqs
            for i in range(lookback, inputs.shape[1]-horizon, stride):
                seqs.append([inputs[:, 0:i, :], inputs[:, i:i+horizon, :]])

            # train model for each sequence
            for _, seq in enumerate(seqs):
                # batch size
                size = seq[0].size()[0]

                predictions = torch.ones(size,0,5)
                if 'cuda' in device:
                    predictions = predictions.cuda()

                # features and target
                x = seq[0].float()
                y = seq[1].float()

                for _ in range(horizon):
                    # predict one time step
                    pred = torch.unsqueeze(model(x), dim=1)
                    
                    # append prediction
                    predictions = torch.cat((predictions, pred), dim=1)

                    # append predicition to input data for next forecast
                    x = torch.cat((x, pred), dim=1)   

                # calculate loss
                loss = criterion(predictions, y)   
                t_loss.append(loss.item())
                
                # checking accuracy of close on last day... horizon days out
                t_acc.extend((1 - torch.abs(predictions[:, -1, 4] - seq[1][:, -1, 4])).tolist())

                # update model parameters
                optimiser.zero_grad()   
                loss.backward()         
                optimiser.step()        

        # validation
        with torch.no_grad():
            for _, data in enumerate(tqdm(valloader, desc='Validating', ascii=True, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
                inputs = data[0]

                if 'cuda' in device:
                    inputs = inputs.cuda()

                # initialize seqs list of length len(series)-lookback with
                #   X of size (bs, n, num_feats) and Y of size (bs, 1, numfeats)
                seqs = []

                # Create seqs
                for i in range(lookback, inputs.shape[1]-horizon, stride):
                    seqs.append([inputs[:, 0:i, :], inputs[:, i:i+horizon, :]])

                # train model for each sequence
                for _, seq in enumerate(seqs):
                    # batch size
                    size = seq[0].size()[0]

                    predictions = torch.ones(size,0,5)
                    if 'cuda' in device:
                        predictions = predictions.cuda()

                    # features and target
                    x = seq[0].float()
                    y = seq[1].float()

                    for _ in range(horizon):
                        # predict one time step
                        pred = torch.unsqueeze(model(x), dim=1)
                        
                        # append prediction
                        predictions = torch.cat((predictions, pred), dim=1)

                        # append predicition to input data for next forecast
                        x = torch.cat((x, pred), dim=1)  

                    # calculate loss
                    loss = criterion(predictions, y)   

                    # checking accuracy of close on last day... horizon days out
                    v_loss.append(loss.item())
                    v_acc.extend((1 - torch.abs(predictions[:, -1, 4] - seq[1][:, -1, 4])).tolist())

        end = time.time()

        # save most recent model
        torch.save([model.kwargs, model.state_dict()], os.path.join(newpath, "weights\last.pth"))

        # calculate average losses
        avg_t_loss = sum(t_loss) / len(t_loss)
        avg_v_loss = sum(v_loss) / len(v_loss)

        # calculate average accuracies
        avg_t_acc = sum(t_acc) / len(t_acc)
        avg_v_acc = sum(v_acc) / len(v_acc)

        # save best model
        if avg_v_loss < best:
            best = avg_v_loss
            torch.save([model.kwargs, model.state_dict()], os.path.join(newpath, "weights\\best.pth"))

        # print logging
        print(f"Time: {round(end-start, 3)}   ", end="")
        print(f"Train Loss: {round(avg_t_loss, 5)}   ", end="")
        print(f"Train Acc: {round(avg_t_acc, 5)}   ", end="")
        print(f"Valid Loss: {round(avg_v_loss, 5)}   ", end="")
        print(f"Valid Acc: {round(avg_v_acc, 5)}   ", end="\n")

        # logging
        with open(os.path.join(newpath, 'logs.txt'), 'a') as f:
            f.write(f"{round(end-start, 3)}, {round(avg_t_loss, 5)}, {round(avg_t_acc, 5)}, {round(avg_v_loss, 5)}, {round(avg_v_acc, 5)}\n")
        
    print(f'\nFinished Training - Models and metrics saved to: \"{newpath}\"')


def parse_args():
    '''
    Saves cmd line arguments for training
    
    Outputs:
        args (dict) - CMD line aruments for training
            - hidden (int) - Number of hidden layers.
            - layers (int) - Number of recurrent layers.
            
            - data (str) - Path to prices data.
            
            - epochs (int) - Number of training epochs.
            - lr (float) - Learning rate.
            - bs (int) - Batch size.
            - workers (int) - Number of worker nodes.
            
            - lookback (int) - Minimum lookback range (252 = 1 year of data)
            - horizon (int) - Number of days to forecast for recursively.
            - stride (int) - Stride to walk through data for training (stride = 1 trains on each day).
            
            - device (str) - Device to use for trainng; cuda:n or cpu.
            
            - savepath (str) - Path to save models.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=32, help='Number of hidden layers.')
    parser.add_argument('--layers', type=int, default=2, help='Number of recurrent layers')

    parser.add_argument('--data', type=str, default='daily_prices', help='Path to prices data')

    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')

    parser.add_argument('--lookback', type=int, default=252, help='Minimum lookback range (252 = 1 year of data)')
    parser.add_argument('--horizon', type=int, default=5, help='Number of days to forecast for recursively.')
    parser.add_argument('--stride', type=int, default=2, help='Stride to walk through data for training (stride = 1 trains on each day).')

    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for trainng; cuda:n or cpu.')

    parser.add_argument('--savepath', type=str, default='rnn', help='Path to save models.')

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
    