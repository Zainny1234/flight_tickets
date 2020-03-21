import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

import os
import joblib
from flight_tickets.src.data_ingest import load_dataset
from flight_tickets.conf.config import DAY_OF_BOOKING, CATEGORICAL_COLUMNS
from flight_tickets.src.util import *

logger = logging.getLogger(__name__)


class PreProcess:
    def __init__(self, X):
        self.x = X

    def basefeat(self, ms, bk_class):
        self.x = marketshare(self.x, ms)
        self.x = classshare(self.x, bk_class)
        self.x = daystodep(self.x)
        self.x = timeofday(self.x)
        self.x = totalstops(self.x)
        self.x = X(self.x)
        return self.x

    def vectoriser(self, training=True):
        self.x['Route'] = self.x['Route'].apply(clean_route)
        if training:
            tf = TfidfVectorizer(ngram_range=(1, 1), lowercase=False)
            route = tf.fit_transform(self.x['Route'])
            joblib.dump(tf, os.path.join(os.getcwd(), 'models', 'tf.sav'))
        else:
            # load tf fomr json(write code for it)
            tf = joblib.load(os.path.join(os.getcwd(), 'models', 'tf.sav'))
            route = tf.transform(self.x['Route'])

        route = pd.DataFrame(data=route.toarray(), columns=tf.get_feature_names())
        self.x = pd.concat([self.x, route], axis=1)
        self.x.drop('Route', axis=1, inplace=True)
        return self.x

    def create_dummies(self, cat_cols):
        self.x = pd.get_dummies(self.x, columns=cat_cols, drop_first=True)
        return self.x

    def column_rename(self):
        self.x.columns = self.x.columns.str.replace(":", "_")
        return self.x

    def preprocess(self, ms, bk_class, cat_cols, training=True):
        self.x = self.basefeat(ms, bk_class)
        self.x = self.create_dummies(cat_cols)
        self.x = self.vectoriser(training)
        self.x = self.column_rename()

        if not training:
            test_cols = self.x.columns
            #print(test_cols)
            #print(len(test_cols))
            dat_cols = joblib.load(os.path.join(os.getcwd(), 'models', 'dat_cols.sav'))
            add_cols = [x for x in dat_cols if x not in test_cols]
            # add_cols = list(set(dat_cols) - set(test_cols))
            #print(add_cols)
            df_new = pd.DataFrame(0, index=range(len(self.x)), columns=add_cols)
            self.x = pd.concat([self.x, df_new], axis=1)
        logger.info('Data preprocessing completed')
        return self.x


if __name__ == "__main__":
    from flight_tickets.src.data_ingest import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from flight_tickets.conf.config import CATEGORICAL_COLUMNS as cat_cols
    import pandas as pd

    x = load_dataset('train.xlsx')
    x.drop(['Price'], axis=1, inplace=True)
    y = load_dataset('test.xlsx')
    # x['label'] = 0
    # y['label'] = 1
    data_in = pd.concat([x, y], axis=0).reset_index(drop=True)
    ms = load_dataset('ms.json')['market']
    bk_class = load_dataset('class.json')['class']

    p = PreProcess(data_in)
    # data1 = p.basefeat(ms, bk_class)
    data = p.preprocess(ms, bk_class, cat_cols, training=True)
    dat_cols = data.columns
    # joblib.dump(dat_cols, os.path.join(os.getcwd(), 'models', 'dat_cols.sav'))
