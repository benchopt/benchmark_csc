from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.fft import fft


class Solver(BaseSolver):
    name = 'Python-PGD'  # proximal gradient, optionally accelerated

    stop_strategy = "callback"

    # Store the information to compute the objective. The parameters of this
    # function are the eys of the dictionary obtained when calling
    # ``Objective.to_dict``.
    def set_objective(self, D, y, lmbd):
        self.D, self.y, self.lmbd = D, y, lmbd

    # Main function of the solver, which computes a solution estimate.
    def run(self, callback):
        fourier_dico = fft(self.D, axis=1)
        L = np.max(np.real(fourier_dico * np.conj(fourier_dico)), axis=1).sum()
        step_size = 1. / L
        mu = step_size * self.lmbd

        n_atoms = self.D.shape[0]
        signal_length = self.y.shape[0]
        kernel_size = self.D.shape[1]
        n_samples = self.y.shape[1]
        w = np.zeros((n_atoms, signal_length - kernel_size + 1, n_samples))

        w_old = w.copy()
        t_old = 1.

        while callback(w):
            signal = signal = np.concatenate([
                np.array([np.convolve(w_k, d_k, mode="full")
                          for w_k, d_k in zip(w[:, :, i], self.D)]
                         ).sum(axis=0).reshape(-1, 1)
                for i in range(w.shape[2])
            ], axis=1)
            diff = signal - self.y
            corr = np.concatenate([
                np.array([np.correlate(diff[:, i], d_k, mode="valid")
                          for d_k in self.D])[:, :, None]
                for i in range(diff.shape[1])
            ], axis=2)
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
