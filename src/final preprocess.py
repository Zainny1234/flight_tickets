import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


from src.data_ingest import load_dataset
from conf.config import DAY_OF_BOOKING, CATEGORICAL_COLUMNS
from src.util import *


class Preprocess:
    def __int__(self, X, Y):
        self.x = X
        self.y = Y

    def basefeat(self, ms, bk_class):
        self.x = marketshare(self.x, ms)
        self.x = classshare(self.x, bk_class)
        self.x = daystodep(self.x)
        self.x = timeofday(self.x)
        self.x = totalstops(self.x)
        self.x = X(self.x)
        return self

    def vectoriser(self, col, training=True):
        self.x[col] = self.x[col].apply(clean_route)
        if training:
            tf = TfidfVectorizer(ngram_range=(1, 1), lowercase=False)
            route = tf.fit_transform(self.x[col])
        else:
            #load tf fomr json(write code for it)
            route = tf.fit_transform(self.x['col'])

        self.x = pd.concat([self.x, route], axis=1)
        self.x.drop('Route', axis=1, inplace=True)
        return self

    def create_dummies(self, cols):
        self.x = pd.get_dummies(self.x, cols)

















