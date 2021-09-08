import numpy as np
from sklearn.metrics import matthews_corrcoef
from pytorch_tabnet.metrics import Metric


class TabNetMCC(Metric):
    def __init__(self):
        """
        Init fn

        Returns
        -------
        None.

        """
        self._name = "mcc"
        self._maximize = True

    def __call__(self, y_true, y_score):
        """
        Call TabNet Metric

        Parameters
        ----------
        y_true : list
            truth values.
        y_score : list
            predicted values.

        Returns
        -------
        float
            Matthews Correlation Coefficient.

        """
        y_pred = np.argmax(y_score, axis=1)
        return matthews_corrcoef(y_true, y_pred)
