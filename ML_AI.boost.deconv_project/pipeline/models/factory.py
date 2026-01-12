from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class ModelFactory:
    @staticmethod
    def get(model_type:str, config:dict=None):
        if model_type == 'xgboost':
            return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=config.get('random_state',42))
        if model_type == 'rf':
            return RandomForestClassifier(random_state=config.get('random_state',42))
        if model_type == 'svm':
            return SVC(probability=True)
        raise ValueError('Unknown model type')
