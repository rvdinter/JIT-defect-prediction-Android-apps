from torch import nn
from torch.nn.init import kaiming_uniform_
from torch.nn import Module


class MLP(Module):
    def __init__(self, n_inputs):
        """
        Define model elements

        Parameters
        ----------
        n_inputs : int
            Number of input features.

        Returns
        -------
        None.

        """
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_inputs, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid())

    def forward(self, X):
        """
        Forward propagate input

        Parameters
        ----------
        X : Torch Tensor
            Tensor of features.

        Returns
        -------
        y : Torch Tensor
            Predicted outputs.

        """
        y = self.mlp(X)
        return y


def weights_init(m):
    """
    Initialize the linear weights of MLP 

    Parameters
    ----------
    m : Torch Model
        MLP model.

    Returns
    -------
    None.

    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        kaiming_uniform_(m.weight, nonlinearity='relu')
