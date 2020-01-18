import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.data_ingest import load_dataset


class MarketShare(BaseEstimator, TransformerMixin):
    """Add Market share feature"""

    def __init__(self, data):
        self.data = data

    def fit(self):
        return self

    def transform(self, market):
        self.data['Market_Share'] = self.data['Airline'].map(market)
        return self.data


class BookingClass(BaseEstimator, TransformerMixin):
    """Add  Class feature"""

    def __init__(self, data):
        self.data = data

    def fit(self):
        return self

    def transform(self, bk_class):
        self.data['Booking_Class'] = self.data['Airline'].map(bk_class)
        return self.data


if __name__ == "__main__":
    x = load_dataset('train.xlsx')
    ms = load_dataset('ms.json')
    bk_class = load_dataset('class.json')['class']

    y = MarketShare(x).transform(ms)
    z = BookingClass(y).transform(bk_class)

