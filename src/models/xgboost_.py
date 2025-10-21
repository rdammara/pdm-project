
from xgboost import XGBRegressor, XGBClassifier

def make_xgb(task='regression', **params):
    if task == 'regression':
        return XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9, random_state=params.get('random_state', 42))
    elif task == 'classification':
        scale_pos_weight = params.get('scale_pos_weight', 1.0)
        return XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, random_state=params.get('random_state', 42),
                             scale_pos_weight=scale_pos_weight, eval_metric='logloss')
    else:
        raise ValueError("task must be 'regression' or 'classification'")
