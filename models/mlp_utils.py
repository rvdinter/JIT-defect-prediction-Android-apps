from torch.nn import BCELoss
import torch
from numpy import vstack
from sklearn.metrics import matthews_corrcoef, accuracy_score


def train_mlp(train_dl, model, max_iterations, device='cpu'):
    """
    Train the MLP model

    Parameters
    ----------
    train_dl : PyTorch DataLoader
        train dataloader.
    model : PyTorch Model
        MLP.
    max_iterations : int
        max number of iterations.
    device : str, optional
        device to train the model on. 'cpu' is recommended.
        The default is 'cpu'.

    Returns
    -------
    None.

    """
    # define the optimization
    criterion = BCELoss()
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=0.001)  # , weight_decay=0.99
    # optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    total_iterations = 0
    # enumerate iterations
    while total_iterations < max_iterations:
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets.float())
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            # Check num iterations
            total_iterations += 1


def evaluate_mlp(test_dl, model, device):
    """
    evaluate the model

    Parameters
    ----------
    test_dl : PyTorch DataLoader
        test dataloader.
    model : PyTorch Model
        MLP.
    device : str, optional
        device to train the model on. 'cpu' is recommended.
        The default is 'cpu'.

    Returns
    -------
    acc : float
        Accuracy.
    mcc : float
        Matthews Correlation Coefficient.

    """
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    mcc = matthews_corrcoef(actuals, predictions)
    acc = accuracy_score(actuals, predictions)
    print(mcc)
    return acc, mcc
