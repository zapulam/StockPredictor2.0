'''
Contains utility functions
By: Zachary Pulliam
'''
import torch
import pandas as pd
import datetime as dt

from datetime import date
from torch.nn.utils.rnn import pad_sequence
from dateutil.relativedelta import relativedelta


def format_yahoo_datetimes():
    '''
    
    '''
    now = dt.datetime.now()

    a = dt.datetime(1970,1,1,23,59,59)
    b = dt.datetime(now.year, now.month, now.day, 23, 59, 59)
    c = b - relativedelta(years=5)

    period1 = str(int((c-a).total_seconds()))   # total seconds from today since Jan. 1, 1970 subracting 5 years
    period2 = str(int((b-a).total_seconds()))   # total seconds from today since Jan. 1, 1970

    return period1, period2


def create_logs(path, horizon):
    '''
    
    '''
    # create list of column names
    columns_list = ['Epoch'] + \
                   ['Time'] + \
                   ['Avg_Train_Loss'] + \
                   ['Train_Accuracy@' + str(i) for i in range(1, horizon + 1)] + \
                   ['Avg_Valid_Loss'] + \
                   ['Valid_Accuracy@' + str(i) for i in range(1, horizon + 1)]

    # create empty dataframe
    logs = pd.DataFrame(columns=columns_list)

    # write logs
    logs.to_csv(path, index=False)

    return logs


def write_logs(path, logs, epoch, time, avg_train_loss, avg_train_accuracies, avg_valid_loss, avg_valid_accuracies):
    '''
    
    '''
    # create list of metrics
    metrics = [epoch] + [time] + [avg_train_loss] + avg_train_accuracies + [avg_valid_loss] + avg_valid_accuracies

    # append current logs to logs df
    logs.loc[len(logs)] = metrics

    # write logs
    logs.to_csv(path, index=False)

    return logs


def custom_collate(batch):
    '''
    
    '''
    # truncate the batch so that all tensors are the same length
    min = batch[0][0].shape[0]
    for i in range(len(batch)):
        if batch[i][0].shape[0] < min: 
            min = batch[i][0].shape[0]
    for i in range(len(batch)):
        if batch[i][0].shape[0] > min:    
            batch[i] = (batch[i][0][-min:, :], batch[i][1])

    xs = [pair[0].unsqueeze(0) for pair in batch]
    ys = [pair[1].unsqueeze(0) for pair in batch]

    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)

    # return the truncated batch
    return X, Y
