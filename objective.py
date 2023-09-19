from benchopt import BaseObjective, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    # Name of the Objective function
    name = "Convolutional Sparse Coding"
    min_benchopt_version = "1.5"

    # parametrization of the objective with various regularization parameters.
    parameters = {
        'reg': [.5],
        'positive': [True, False]  # If true, add a constraints on z >= 0.
    }

    def set_data(self, D, y):
        """Set the data from a Dataset to compute the objective.

        The argument are the key in the data dictionary returned by
        get_data.

        Parameters
        ----------
        D : array, shape (n_atoms, kernel_size)
            Dictionary for the CSC problem.
        y : array, shape (n_times, n_samples)
            Signals to run the CSC on.
        """
        self.D, self.y = D, y
        self.lmbd = self.reg * get_lambda_max(D, y)

    def get_one_result(self):
        n_times, n_samples = self.y.shape
        n_atoms, kernel_size = self.D.shape
        return dict(
            theta=np.zeros((n_atoms, n_times - kernel_size + 1, n_samples))
        )

    def get_objective(self):
        "Returns a dict to pass to the set_objective method of a solver."
        return dict(D=self.D, y=self.y, lmbd=self.lmbd, positive=self.positive)

    def evaluate_result(self, theta):
        """Compute the objective value given the output theta of a solver.

        Parameters
        ----------
        theta: array, shape (n_atoms, n_times_valid, n_samples)
            Solution of the CSC.
        """
        if self.positive:
            theta = theta * (theta > 0)
        signal = np.concatenate([
            np.sum([
                np.convolve(theta_k, d_k, mode="full")
                for theta_k, d_k in zip(theta[:, :, i], self.D)
            ], axis=0).reshape(-1, 1)
            for i in range(theta.shape[2])
        ], axis=1)
        diff = self.y - signal
        return .5 * (diff * diff).sum() + self.lmbd * abs(theta).sum()


def get_lambda_max(y, D_hat):

    return np.max([[
        np.correlate(D_k, y_i, mode='valid') for y_i in y.T
    ] for D_k in D_hat])
