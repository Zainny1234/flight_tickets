import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.data_ingest import load_dataset
from conf.config import DAY_OF_BOOKING


class MarketShare(BaseEstimator, TransformerMixin):
    """Add Market share feature"""

    def __init__(self, market):
        self.market = market

    def fit(self):
        return self

    def transform(self, X):
        X['Market_Share'] = X['Airline'].map(self.market)
        return X


class BookingClass(BaseEstimator, TransformerMixin):
    """Add  Class feature"""

    def __init__(self, bk_class):
        self.bk_class = bk_class

    def fit(self):
        return self

    def transform(self, X):
        X['Booking_Class'] = X['Airline'].map(bk_class)
        return X


class DaysToDep(BaseEstimator, TransformerMixin):
    """Calculating dyas to departure"""

    def __init__(self, dayofbk):
        self.dayofbk = dayofbk

    def fit(self):
        return self

    def transform(self, X):
        df1 = X.copy()
        df1['Day_of_Booking'] = self.dayofbk
        df1['Day_of_Booking'] = pd.to_datetime(df1['Day_of_Booking'], format='%d/%m/%Y')
        df1['Date_of_Journey'] = pd.to_datetime(df1['Date_of_Journey'], format='%d/%m/%Y')
        df1['Days_to_Departure'] = (df1['Date_of_Journey'] - df1['Day_of_Booking']).dt.days
        X['Days_to_Departure'] = df1['Days_to_Departure']
        return X


class TimeOfDay(BaseEstimator, TransformerMixin):
    def __init(self):
        pass

    def fit(self):
        return self

    def transform(self, X):
        X['Arrival_Time'] = X['Arrival_Time'].str.split(' ').str[0]
        X['Dep_timeofday'] = X['Dep_Time'].apply(get_departure)
        X['Arr_timeofday'] = X['Arrival_Time'].apply(get_departure)
        return X


def get_departure(dep):
    dep = dep.split(':')
    dep = int(dep[0])
    if 6 <= dep < 12:
        return 'Morning'
    elif 12 <= dep < 17:
        return 'Noon'
    elif 17 <= dep < 20:
        return 'Evening'
    else:
        return 'Night'


class TotalStops(BaseEstimator, TransformerMixin):
    def __init(self):
        pass

    def fit(self):
        return self

    def transform(self, X):
        X['Total_Stops'] = X['Total_Stops'].str.replace('non-stop', '0')
        X['Total_Stops'] = X['Total_Stops'].str.replace('stops', '')
        X['Total_Stops'] = X['Total_Stops'].str.replace('stop', '')
        X['Total_Stops'].fillna(0, inplace=True)
        X['Total_Stops'] = X['Total_Stops'].astype(float)
        return X

class Duration(BaseEstimator, TransformerMixin):
    def __init(self):
        pass

    def fit(self):
        return self

    def transform(self, X):
        X['Hours'] = X['Duration'].str.split(' ').str[0]
        X['Hours'] = X['Hours'].str.replace('h', '').astype(float)
        X['Hours'].fillna(0, inplace=True)
        X['Minutes'] = X['Duration'].str.split(' ').str[1]
        X['Minutes'] = X['Minutes'].str.replace('m', '').astype(float)
        X['Minutes'].fillna(0, inplace=True)
        X['Hours'] = X['Hours'] * 60
        X['Duration'] = X['Hours'] + X['Minutes']
        X.drop(['Hours', 'Minutes'], axis=1, inplace=True)
        return X


if __name__ == "__main__":
    x = load_dataset('train.xlsx')
    ms = load_dataset('ms.json')['market']
    bk_class = load_dataset('class.json')['class']

    y = MarketShare(ms).transform(x)
    z = BookingClass(bk_class).transform(y)
    u = DaysToDep(DAY_OF_BOOKING).transform(z)
    v = TimeOfDay().transform(u)
    w = Duration().transform(v)
