from metrics.evaluate_model import evaluate_model
import xgboost as xgb

def xgboost_fold(X_train, y_train, X_test, y_test, device='cpu'):
    """
    Train the XGBoost model for one fold

    Parameters
    ----------
    X_train : list
        Train features.
    y_train : list
        Train targets.
    X_test : list
        Train features.
    y_test : list
        Test targets.
    device : str, optional
        Device to train the TabNet model on. 'gpu' is recommended.
        The default is 'cpu'.

    Returns
    -------
    dict
        Model metrics.

    """
    model_xgb = xgb.XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1
        )
    model_xgb.fit(X_train, y_train, eval_metric='error')
    mcc, acc = evaluate_model(model_xgb, X_test, y_test)
    # put metrics in logger
    return {'model': 'xgb', 'mcc': mcc, 'acc': acc}