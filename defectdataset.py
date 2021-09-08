from torch.utils.data import Dataset


class DefectDataset(Dataset):
    def __init__(self, X, y):
        """
        Store inputs and outputs

        Parameters
        ----------
        X : Numpy array
            Features.
        y : Numpy array
            targets.

        Returns
        -------
        None.

        """
        # store the inputs and outputs
        self.X = X.astype('float32')
        self.y = y
        self.y = self.y.reshape((len(self.y), 1))

    def __len__(self):
        """
        Number of rows in the dataset

        Returns
        -------
        int
            Length of dataset.

        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get a row at an index

        Parameters
        ----------
        idx : int
            Index to get.

        Returns
        -------
        list
            features.
        int
            target.

        """
        return self.X[idx], self.y[idx]
