'''
Contains utility functions
By: Zachary Pulliam
'''
import pandas as pd
import datetime as dt

from datetime import date
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
    # create list of column names
    columns_list = ['Epoch'] + ['Time'] + ['Avg_Loss'] + ['Accuracy@' + str(i) for i in range(1, horizon + 1)]

    # create empty dataframe
    logs = pd.DataFrame(columns=columns_list)

    # write logs
    logs.to_csv(path, index=False)

    return logs

def write_logs(path, logs, epoch, time, avg_loss, avg_accuracies):
    # create list of metrics
    metrics = [epoch] + [time] + [avg_loss] + avg_accuracies

    # append current logs to logs df
    logs.loc[len(logs)] = metrics

    # write logs
    logs.to_csv(path, index=False)

    return logs
