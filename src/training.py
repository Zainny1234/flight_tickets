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
    ## Training random forest
    rf = ml_params['random forest']['estimators']
    model_rf, pred_rf = algorithm_pipeline(X_train, X_test, y_train, y_test, model=RandomForestRegressor(),
                                           param_grid=rf, cv=5, scoring_fit='r2')
    best_rf_score = model_rf.best_score_
    best_rf_es = model_rf.best_estimator_
    df_score_rf = pd.DataFrame(model_rf.cv_results_).sort_values('rank_test_score')

    ## Training Light GBM
    lgbm = ml_params['lightGBM']['estimators']
    model_lgbm, pred_lgbm = algorithm_pipeline(X_train, X_test, y_train, y_test, model=lgb.LGBMRegressor(),
                                               param_grid=lgbm, cv=5, scoring_fit='r2')
    best_lgbm_score = model_lgbm.best_score_
    best_lgbm_es = model_lgbm.best_estimator_
    df_score_lgbm = pd.DataFrame(model_lgbm.cv_results_).sort_values('rank_test_score')

    ##Training xgboost
    xgbt = ml_params['xgboost']['estimators']
    model_xgbt, pred_xgbt = algorithm_pipeline(X_train, X_test, y_train, y_test, model=xgb.XGBRegressor(),
                                               param_grid=xgbt, cv=5, scoring_fit='r2')
    best_xgbt_score = model_xgbt.best_score_
    best_xgbt_es = model_xgbt.best_estimator_
    df_score_xgbt = pd.DataFrame(model_xgbt.cv_results_).sort_values('rank_test_score')

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

    x = load_dataset('train.xlsx')
    ms = load_dataset('ms.json')['market']
    bk_class = load_dataset('class.json')['class']

    p = PreProcess(x)
    data = p.preprocess(ms, bk_class, cat_cols, training=False)
    data.columns = data.columns.str.replace(":", "_")
    y = data['Price'].values
    x = data.drop(labels=['Price'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
    scores = models_training(X_train, X_test, y_train, y_test)
    # model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model=xgb.XGBRegressor(),
    #                                  param_grid=xgbt, cv=2, scoring_fit='r2')
