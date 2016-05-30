#!/usr/bin/env python3

import pandas
from sklearn.linear_model import LogiticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

def gen_lr_base(X_train, y_train, **params):
    """
    params should include:
        scoring
    """
    # convert to pandas.Series for convinience
    if isinstance(y_train, 'list'):
        y_train = pandas.Series(y_train)

    # gen clf list for training
    clfs = []
    if 'class_weight' in params:
        class_weight = params['class_weight']
    else:
        class_weight = 0

    if (len(y_train[y_train > 1])):
        clfs.append(LogiticRegression(multi_class='ovr', class_weight=class_weight))
        clfs.append(LogiticRegression(multi_class='ovo', class_weight=class_weight))
    else:
        clfs.append(LogiticRegression(penalty='l1', class_weight=class_weight))
        clfs.append(LogiticRegression(penalty='l2', class_weight=class_weight))

    best_score = 0
    best_clf = None
    for clf in clfs:
        res = cross_val_score(clf, X_train, y_train, scoring=params['scoring'], cv=5, n_jobs=-1)
        cur_score = sum(res) / 5.0
        if cur_score > best_score:
            best_score = cur_score
            best_clf = clf

    return best_clf


def gen_best_xgboost(X_train, y_train, **params):
    """
    params should include:
        scoring,
        objective
    """
    if isinstance(y_train, 'list'):
        y_train = pandas.Series(y_train)

    # modify parameters, it will benefit the score at 0.0*
    base_params = {
            'learning_rate' : 0.1,
            'n_estimators' : 200,
            'gamma' : 0,
            'subsample' : 0.8,
            'colsample_bytree' : 0.8,
            'objective' : params['objective'],
            'nthread' : 8,
            'scale_pos_weight' : 1,
            'seed' : 27
            }

    # tree parameters
    scoring = params['scoring']
    param_test = {
            'max_depth' : list(range(3, 10, 1)),
            'min_child_weight' : list(range(1, 6, 1))
            }
    gsearch = GridSearchCV(estimator=xgb.XGBClassifier(base_params),
            param_grid=param_test,
            scoring=scoring,
            n_jobs=4,
            iid=False,
            cv=5)
    gsearch.fit(X_train, y_train)
    print('best tree params: ', gsearch.best_params_)
    print('best score currently: ', gsearch.best_score_)
    base_params.update(gsearch.best_params)

    # gamma parameters
    param_test = {
            'gamma' : [i/10.0 for i in range(1, 10)]
            }
    del base_params['gamma']
    gsearch = GridSearchCV(estimator=xgb.XGBClassifier(base_params),
            param_grid=param_test,
            scoring=scoring,
            n_jobs=4,
            iid=False,
            cv=5)
    gsearch.fit(X_train, y_train)
    print('best gamma params: ', gsearch.best_params_)
    print('best score currently: ', gsearch.best_score_)
    base_params.update(gsearch.best_params_)

    # subsample parameters
    param_test = {
            'subsample' : [i/100.0 for i in range(80, 100, 1)],
            }
    del base_params['subsample']
    gsearch = GridSearchCV(estimator=xgb.XGBClassifier(base_params),
            param_grid=param_test,
            scoring=scoring,
            n_jobs=4,
            iid=False,
            cv=5)
    gsearch.fit(X_train, y_train)
    print('best subsample: ', gsearch.best_params_)
    print('best score currently: ', gsearch.best_score_)
    base_params.update(gsearch.best_params_)

    # reg parameters
    param_test = {
        'reg_alpha' : [1e-5, 1e-2, 0.1, 1, 100, 0, 0.001, 0.005, 0.01, 0.05]
        }
    gsearch = GridSearchCV(estimator=xgb.XGBClassifier(base_params),
            param_grid=param_test,
            scoring=scoring,
            n_jobs=4,
            iid=False,
            cv=5)
    gsearch.fit(X_train, y_train)
    print('best reg: ', gsearch.best_params_)
    print('best score currently: ', gsearch.best_score_)
    base_params.update(gsearch.best_params_)

    return xgb.XGBClassifier(base_params)


def gen_best_rf(X_train, y_train, **params):
    base_params = {
            'n_estimators' : 200,
            'criterion' :  'entropy',
            'min_samples_split' : 3,
            'max_features' : 'auto',
            'max_leaf_nodes' : None,
            'bootstrap' : True,
            'oob_score' : False,
            'n_jobs' : 4,
            'random_state' : None,
            'verbose' : 1,
            'warm_start' : False,
            'class_weight' : None
            }
    scoring = params['scoring']
    param_test = {
            'max_depth' : list(range(15, 31)),
            'min_weight_fraction_leaf' : [0.05, 0.1, 0.15, 0.2, 0.25]
            }
    gsearch = GridSearchCV(estimator=RandomForestClassifier(base_params),
            param_grid=param_test,
            scoring=scoring,
            n_jobs=4,
            iid=False,
            cv=5)
    gsearch.fit(X_train, y_train)
    base_params.update(gsearch.best_params_)
    return RandomForestClassifier(base_params)
