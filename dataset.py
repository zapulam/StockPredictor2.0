'''
Contains S&P 500 dataset class used for training
'''

import os
import torch
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy import stats
from torch.utils.data import Dataset
from sklearn.linear_model import LinearRegression


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
        data = pd.read_csv(os.path.join(self.folder, file))[['Open', 'High', 'Low', 'Volume', 'Close']] 
        
        # detrend and deseason all columns
        decomp, forecast = pd.DataFrame(), pd.DataFrame()
        for col in data.columns:
            # detrend and deason column
            result = sm.tsa.seasonal_decompose(data[col], model='additive', period=252, extrapolate_trend=25, two_sided=False)
            
            # add decomposed column to dataframe
            decomp[col] = result.resid
            
            # create seasonal forecast component based on moving avg
            forecast['seasonal_forecast'] = result.seasonal[-252:].reset_index(drop=True).rolling(14, min_periods=1).mean()
            
            # create trend forecast component based on OLS
            model = LinearRegression()
            model.fit(np.array(data.index[-50:]).reshape(-1, 1), result.trend[-50:])
            forecast['trend_forecast'] = model.intercept_ + model.coef_ * range(data.index[-1], data.index[-1]+252)

        # normalize input data
        mins, maxs = decomp.min(), decomp.max() 
        normalized = (decomp-mins)/(maxs-mins)
        
        # difference normalized data
        differenced = normalized.diff().iloc[1: , :]

        # remove outliers
        for col in differenced.columns:
            z_scores = np.abs(stats.zscore(differenced[col]))
            outliers = z_scores > 3
            differenced[col][outliers] = differenced[col].mean()

        # convert to tensor
        input = torch.tensor(differenced.values)

        return input
    