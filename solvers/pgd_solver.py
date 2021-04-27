import numpy as np

from scipy.fft import fft
from benchopt import BaseSolver


class Solver(BaseSolver):
    name = 'Python-PGD'  # proximal gradient, optionally accelerated

    # Store the information to compute the objective. The parameters of this
    # function are the eys of the dictionary obtained when calling
    # ``Objective.to_dict``.
    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

    # Main function of the solver, which computes a solution estimate.
    def run(self, n_iter):
        fourier_dico = fft(self.X, axis=1)
        L = np.max(np.real(fourier_dico * np.conj(fourier_dico)), axis=1).sum()
        step_size = 1. / L
        mu = step_size * self.lmbd

        n_features = self.X.shape[0]
        signal_length = self.y.shape[0]
        kernel_size = self.X.shape[1]
        n_samples = self.y.shape[1]
        w = np.zeros((n_features, signal_length - kernel_size + 1, n_samples))

        w_old = w.copy()
        t_old = 1.

        for _ in range(n_iter):
            signal = signal = np.concatenate(
                [np.array([np.convolve(w_k, x_k, mode="full")
                           for w_k, x_k in zip(w[:, :, i], self.X)]
                          ).sum(axis=0).reshape(-1, 1)
                 for i in range(w.shape[2])], axis=1)
            diff = signal - self.y
            corr = np.concatenate(
                [np.array([np.correlate(diff[:, i], d_k, mode="valid")
                           for d_k in self.X])[:, :, None]
                 for i in range(diff.shape[1])], axis=2)
            w -= step_size * corr
            w = np.maximum(0, np.abs(w) - mu)
            # w = np.sign(w) * np.maximum(0, np.abs(w) - mu)

            t = 0.5 * (1 + np.sqrt(1 + 4 * t_old * t_old))
            z = w + ((t_old-1) / t) * (w - w_old)
            w_old = w.copy()
            t_old = t
            w = z

        self.w = w

    # Return the solution estimate computed.
    def get_result(self):
        return self.w
