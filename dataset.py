'''
Contains S&P 500 dataset class used for training
By: Zachary Pulliam
'''

import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from decomposition import decompose


class Stock_Data(Dataset):
    '''
    S&P 500 Dataset for training RNN to predict future close prices
    '''
    def __init__(self, path, lags, horizon, stride):
        '''
        Constructor method
        '''
        self.data = []

        data = pd.read_csv(path).drop(columns=['Adj Close'])
        
        # decompose data
        timeseries = decompose(data)['input']

        # convert to tensor
        timeseries = torch.tensor(timeseries.values)

        # create X and Y
        for i in range(lags, timeseries.shape[0]-horizon, stride):
            X = timeseries[0:i, :]
            Y = timeseries[i:i+horizon, :]

            # self.data.append([X, Y])
            self.data.append([X, Y])


    def __len__(self):
        '''
        Returns length of dataset
        
        Outputs:
            - length (int) - length of dataset
        '''
        length = len(self.data)

        return length


    def __getitem__(self, idx):
        '''
        Gets individual stock history for training 

        Inputs:
            idx (int) - index to refernece from self.data

        Outputs:
            - X (tensor) - model input timeseries data
            - Y (tensor) - ground truth forecast data
        '''

        # read data
        X, Y = self.data[idx]

        return X, Y
    