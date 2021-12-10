import numpy as np

from benchopt.base import BaseObjective


class Objective(BaseObjective):
    # Name of the Objective function
    name = "Convolutional Sparse Coding"

    # parametrization of the objective with various regularization parameters.
    parameters = {
        'reg': [1]
    }

    def __init__(self, reg=.1):
        "Store the value of the parameters used to compute the objective."
        self.reg = reg

    def set_data(self, D, y):
        """Set the data from a Dataset to compute the objective.

        The argument are the key in the data dictionary returned by
        get_data.
        """
        self.D, self.y = D, y
        self.lmbd = self.reg

    def to_dict(self):
        "Returns a dict to pass to the set_objective method of a solver."
        return dict(D=self.D, y=self.y, lmbd=self.lmbd)

    def compute(self, theta):
        "Compute the objective value given the output x of  a solver."
        signal = np.concatenate([
            np.sum([
                np.convolve(theta_k, d_k, mode="full")
                for theta_k, d_k in zip(theta[:, :, i], self.D)
            ], axis=0).reshape(1, -1)
            for i in range(theta.shape[2])
        ], axis=0)
        diff = self.y - signal
        return .5 * (diff * diff).sum() + self.lmbd * abs(theta).sum()
