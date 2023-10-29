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


class SP_500(Dataset):
    '''
    S&P 500 Dataset for training RNN to predict future close prices
    '''
    def __init__(self, folder):
        '''
        Constructor method
        '''
        self.data = []
        self.folder = folder

        # create list of files: [A.csv, AAL.csv, ...]
        all_files = os.listdir(folder)
        if '_.txt' in all_files: all_files.remove('_.txt')
        files = []

        # set max file length ( 5 years worth of data )
        max = 0
        for file in all_files:
            df = pd.read_csv(os.path.join(folder, file), index_col=0)
            length = len(df.index) 
            if length > max:
                max = length

        # remove files with less than 5 years of data
        for file in all_files:
            df = pd.read_csv(os.path.join(folder, file), index_col=0)
            if len(df.index) == max:
                files.append(file)

        self.data = files


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
            - x (tensor) - training input data
            - mins (tensor) - minimum values for all input features
            - maxs (tensor) - maximum values for all input features
        '''

        # read data
        file = self.data[idx]
        data = pd.read_csv(os.path.join(self.folder, file)).drop(columns=['Adj Close'])
        
        # decompose data
        input = decompose(data)[0]

        # convert to tensor
        input = torch.tensor(input.values)

        return input
    