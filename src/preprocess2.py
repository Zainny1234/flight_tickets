from src.util import *
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline


class BaseFeat(BaseEstimator, TransformerMixin):
    def __init__(self, ms, bkclass):
        self.ms = ms
        self.bkclass = bkclass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = marketshare(X, self.ms)
        X = classshare(X, self.bkclass)
        X = daystodep(X)
        X = timeofday(X)
        X = totalstops(X)
        X = duration(X)
        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """Logarithm transformer."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accomodate the pipeline
        return self

    def transform(self, X, y= None):
        X = X.copy()

        # check that the values are non-negative for log transform
        # if not (X[self.variables] > 0).all().all():
        #     vars_ = self.variables[(X[self.variables] <= 0).any()]
        #     raise InvalidModelInputError(
        #         f"Variables contain zero or negative values, "
        #         f"can't apply log for vars: {vars_}")

        for feature in self.variables:
            X[feature] = np.log(X[feature])

        return X


if __name__ == "__main__":
    x = load_dataset('train.xlsx')
    ms = load_dataset('ms.json')['market']
    bk_class = load_dataset('class.json')['class']
    # df = BaseFeat(ms, bk_class).transform(x)
    # df = LogTransformer(['Price', 'Duration']).transform(df)

    Feat = [
        ('Base Features', BaseFeat(ms, bk_class)),
        ('Log Transform', LogTransformer(['Price', 'Duration'])),
    ]

    pipe_feat = Pipeline(Feat)
    df = pipe_feat.transform(x)
