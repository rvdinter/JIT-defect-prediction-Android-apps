from metrics.evaluate_model import evaluate_model
from metrics.tabnet_matthews_correlation_coefficient import TabNetMCC
from pytorch_tabnet.tab_model import TabNetClassifier


def tabnet_fold(X_train, y_train, X_test, y_test, X_val, y_val, 
                batch_size=1024, device='cpu'):
    """
    Train the TabNet model for one fold

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
    X_val : list
        Validation features.
    y_val : list
        Validation targets.
    batch_size : int, optional
        Batch size for the TabNet model. The default is 1024.
    device : str, optional
        Device to train the TabNet model on. 'gpu' is recommended.
        The default is 'cpu'.

    Returns
    -------
    dict
        Model metrics.

    """
    # Check if one remains, otherwise there occurs an error
    drop_last = False
    if len(y_train) % 1024 == 1:
        drop_last = True
    # get the model
    model = TabNetClassifier(
        optimizer_params=dict(lr=2e-3)
    )
    # Train on the first half
    # define the network
    model.fit(X_train, y_train,
              weights=1,
              max_epochs=150,
              patience=20,
              eval_set=[(X_train, y_train), (X_val, y_val)],
              eval_name=['train', 'val'],
              eval_metric=[TabNetMCC],
              batch_size=batch_size,
              drop_last=drop_last
              )
    mcc, acc = evaluate_model(model, X_test, y_test)
    # put metrics in logger
    return {'model': 'tabnet', 'mcc': mcc, 'acc': acc}
