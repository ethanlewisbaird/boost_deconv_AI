import os
import numpy as np
from joblib import dump
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from pipeline.models.factory import ModelFactory


def train_model(pseudo_bulk: dict, config: dict):
    X = pseudo_bulk['pseudobulk'].T.values
    y = pseudo_bulk['meta']['dominant_cluster'].values
    # Ensure labels are encoded consistently (handles string/integer labels)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    model_type = config.get('model_type','xgboost')
    estimator = ModelFactory.get(model_type, config)
    if config.get('hyperparameter_search', True):
        if model_type == 'xgboost':
            param_grid = {'n_estimators':[50,100], 'max_depth':[3,6], 'learning_rate':[0.1,0.01]}
        elif model_type == 'rf':
            param_grid = {'n_estimators':[100,200], 'max_depth':[None,20]}
        else:
            param_grid = {'C':[0.1,1.0]}
        gs = GridSearchCV(estimator, param_grid, cv=config.get('cv_folds',5), n_jobs=1, verbose=1)
        gs.fit(X,y_enc)
        model = gs.best_estimator_
        model._cv_results_ = gs.cv_results_
    else:
        model = estimator.fit(X,y_enc)
    outdir = config.get('output_dir','results')
    os.makedirs(os.path.join(outdir,'models'), exist_ok=True)
    dump(model, os.path.join(outdir,'models', f'model_{model_type}.joblib'))
    return model
