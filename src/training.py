from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from src.data_ingest import ml_params
from src.final_preprocess import PreProcess
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import joblib
import os
import logging
logger = logging.getLogger(__name__)


def algorithm_pipeline(X_train, X_test, y_train, y_test,
                       model, param_grid, cv=10, scoring_fit='r2',
                       do_probabilities=False):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring=scoring_fit,
        verbose=2
    )
    fitted_model = gs.fit(X_train, y_train)

    if do_probabilities:
        pred = fitted_model.predict_proba(X_test)
    else:
        pred = fitted_model.predict(X_test)

    return fitted_model, pred


def models_training(X_train, X_test, y_train, y_test):


    try:
        assert len(X_train.columns) == 572
    except AssertionError as err:
        print('number of columns should be 572')
        logger.info("File not found", exc_info=True)
        raise

    ## Training random forest
    logger.info('Starting Random Forest')
    rf = ml_params['random forest']['estimators']
    model_rf, pred_rf = algorithm_pipeline(X_train, X_test, y_train, y_test, model=RandomForestRegressor(),
                                           param_grid=rf, cv=5, scoring_fit='r2')
    best_rf_score = model_rf.best_score_
    best_rf_es = model_rf.best_estimator_
    df_score_rf = pd.DataFrame(model_rf.cv_results_).sort_values('rank_test_score')
    logger.info('Random forest model done')

    ## Training Light GBM
    logger.info('Starting Light GBM')
    lgbm = ml_params['lightGBM']['estimators']
    model_lgbm, pred_lgbm = algorithm_pipeline(X_train, X_test, y_train, y_test, model=lgb.LGBMRegressor(),
                                               param_grid=lgbm, cv=5, scoring_fit='r2')
    best_lgbm_score = model_lgbm.best_score_
    best_lgbm_es = model_lgbm.best_estimator_
    df_score_lgbm = pd.DataFrame(model_lgbm.cv_results_).sort_values('rank_test_score')
    logger.info('GBM model done')

    ##Training xgboost
    logger.info('Starting XGBOOST')
    xgbt = ml_params['xgboost']['estimators']
    model_xgbt, pred_xgbt = algorithm_pipeline(X_train, X_test, y_train, y_test, model=xgb.XGBRegressor(),
                                               param_grid=xgbt, cv=5, scoring_fit='r2')
    best_xgbt_score = model_xgbt.best_score_
    best_xgbt_es = model_xgbt.best_estimator_
    df_score_xgbt = pd.DataFrame(model_xgbt.cv_results_).sort_values('rank_test_score')
    logger.info('XGBOOST model done')

    if not os.path.exists('models'):
        os.mkdir('models')

    joblib.dump(best_rf_es, os.path.join(os.getcwd(), 'models', 'rf_es.sav'))
    joblib.dump(best_lgbm_es, os.path.join(os.getcwd(), 'models', 'lgbm_es.sav'))
    joblib.dump(best_xgbt_es, os.path.join(os.getcwd(), 'models', 'xgbt_es.sav'))

    score_dict = {'RF': best_rf_score,
                  'LightGBM': best_lgbm_score,
                  'Xgboost': best_xgbt_score}
    return score_dict


if __name__ == '__main__':
    from src.data_ingest import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from conf.config import CATEGORICAL_COLUMNS as cat_cols
    import pandas as pd

    x = load_dataset('train.xlsx')
    y_dat = x['Price'].values
    x.drop(['Price'], axis=1, inplace=True)
    y = load_dataset('test.xlsx')
    ms = load_dataset('ms.json')['market']
    bk_class = load_dataset('class.json')['class']

    x['label'] = 0
    y['label'] = 1
    data_in = pd.concat([x, y], axis=0).reset_index(drop=True)

    p = PreProcess(data_in)
    # data1 = p.basefeat(ms, bk_class)
    data = p.preprocess(ms, bk_class, cat_cols, training=True)

    data.columns = data.columns.str.replace(":", "_")

    x_dat = data.loc[data['label'] == 0]
    x_dat = x_dat.drop('label', 1)

    X_train, X_test, y_train, y_test = train_test_split(x_dat, y_dat, test_size=0.20, random_state=1)
    scores = models_training(X_train, X_test, y_train, y_test)
    # model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model=xgb.XGBRegressor(),
    #                                  param_grid=xgbt, cv=2, scoring_fit='r2')
