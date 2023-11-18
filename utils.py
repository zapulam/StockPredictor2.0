'''
Contains utility functions
By: Zachary Pulliam
'''
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
