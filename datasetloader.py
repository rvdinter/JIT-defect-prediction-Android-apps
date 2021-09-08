import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DatasetLoader():
    def __init__(self, path, csvfile):
        """
        Load a CSV file and save the features and targets

        Parameters
        ----------
        path : str
            Path of the file.
        csvfile : str
            Filename.

        Returns
        -------
        None.

        """
        df = pd.read_csv(f'{path}/{csvfile}', index_col=None,
                         header=0,  sep='[:,;]', engine='python')
        df.dropna(inplace=True)
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1].astype('int')

    def slice_dataset(self, train_index, test_index, val=False):
        """
        Slice dataset

        Parameters
        ----------
        train_index : list
            Train indices to slice.
        test_index : list
            Test indices to slice.
        val : bool, optional
            Select whether to split the train set into train and val. 
            Required for TabNet model. The default is False.

        Returns
        -------
        lists
            Lists of features and targets.

        """
        X_train, X_test = self.X[train_index], self.X[test_index]
        y_train, y_test = self.y[train_index], self.y[test_index]
        if val:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
            return X_train, X_test, X_val, y_train, y_test, y_val
        else:
            return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test, X_val=None):
        """
        Normalize the data

        Parameters
        ----------
        X_train : list
            Train targets.
        X_test : list
            Test targets.
        X_val : list, optional
            Validation targets. Include for TabNet model. The default is None.

        Returns
        -------
        lists
            Lists of targets.

        """
        if X_val is not None:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            X_val = scaler.transform(X_val)
            return X_train, X_test, X_val
        else:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            return X_train, X_test
