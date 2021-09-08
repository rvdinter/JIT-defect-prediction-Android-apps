from sklearn.metrics import matthews_corrcoef, accuracy_score


def evaluate_model(model, X_test, y_test):
    """ Evaluate a model
    Parameters
    ----------
    model : sklearn object
        a sklearn-style model.
    X_test : list
        test features.
    y_test : list
        test targets.

    Returns
    -------
    mcc : float
        Matthews Correlation Coefficient.
    acc : float
        Accuracy.

    """
    y_hat = model.predict(X_test)
    mcc = matthews_corrcoef(y_test, y_hat)
    acc = accuracy_score(y_test, y_hat)
    return mcc, acc
