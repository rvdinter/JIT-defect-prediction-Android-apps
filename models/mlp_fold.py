from models.mlp_utils import train_mlp, evaluate_mlp
from models.mlp import MLP, weights_init
from defectdataset import DefectDataset
from torch.utils.data import DataLoader


def mlp_fold(X_train, y_train, X_test, y_test, device='cpu'):
    """
    Train the MLP for one fold

    Parameters
    ----------
    X_train : list
        features.
    y_train : list
        targets.
    X_test : list
        test features.
    y_test : list
        test targets.
    device : str, optional
        device to train the model on. 'gpu' is recommended.
        The default is 'cpu'.

    Returns
    -------
    dict
        metrics.

    """
    # put data in dataset
    train_ds = DefectDataset(X_train, y_train)
    test_ds = DefectDataset(X_test, y_test)
    # prepare data loaders
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)
    model = MLP(6)
    model.apply(weights_init)
    # train the model
    train_mlp(train_dl, model, max_iterations=10000, device=device)
    # evaluate the model
    acc, mcc = evaluate_mlp(test_dl, model, device=device)
    # put metrics in logger
    return {'model': 'mlp', 'mcc': mcc, 'acc': acc}
