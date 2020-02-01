from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from src.data_ingest import ml_params
from src.vectorizertemp import train_df, test_df

rf = ml_params['random forest']['estimators']
#train_df.head(56)


def algorithm_pipeline(X_train, X_test, y_train, y_test,
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
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


if __name__ == '__main__':
    y = train_df['Price'].values
    x = train_df.drop(labels=['Price'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
    model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model=RandomForestClassifier(),
                                     param_grid=rf, cv=5, scoring_fit='neg_mean_squared_error')
