import numpy as np

from benchopt.base import BaseObjective


class Objective(BaseObjective):
    # Name of the Objective function
    name = "Lasso Regression"

    # parametrization of the objective with various regularization parameters.
    parameters = {
        'reg': [1]
    }

    def __init__(self, reg=.1):
        "Store the value of the parameters used to compute the objective."
        self.reg = reg

    def set_data(self, X, y):
        """Set the data from a Dataset to compute the objective.

        The argument are the key in the data dictionary returned by
        get_data.
        """
        self.X, self.y = X, y
        self.lmbd = self.reg

    def to_dict(self):
        "Returns a dict to pass to the set_objective method of a solver."
        return dict(X=self.X, y=self.y, lmbd=self.lmbd)

    def compute(self, theta):
        "Compute the objective value given the output x of  a solver."
        signal = np.concatenate([np.array([np.convolve(theta_k, d_k, mode="full")
                                          for theta_k, d_k in zip(theta[:, :, i], self.X)]).sum(axis=0).reshape(-1, 1)
                                for i in range(theta.shape[2])], axis=1)
        diff = self.y - signal
        return .5 * (diff * diff).sum() + self.lmbd * abs(theta).sum()
