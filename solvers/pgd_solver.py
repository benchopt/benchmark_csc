from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.fft import fft


class Solver(BaseSolver):
    name = 'Python-PGD'

    sampling_strategy = "callback"

    # Store the information to compute the objective. The parameters of this
    # function are the eys of the dictionary obtained when calling
    # ``Objective.to_dict``.
    def set_objective(self, D, y, lmbd, positive):
        self.D, self.y, self.lmbd, self.positive = D, y, lmbd, positive

    # Main function of the solver, which computes a solution estimate.
    def run(self, callback):
        # Computes Lipschitz upper bound with FFT

        fourier_dico = fft(self.D, axis=1)
        L = np.max(np.real(fourier_dico * np.conj(fourier_dico)), axis=1).sum()
        step_size = 1. / L
        mu = step_size * self.lmbd

        n_atoms, kernel_size = self.D.shape
        signal_length, n_samples = self.y.shape
        w = np.zeros((n_atoms, signal_length - kernel_size + 1, n_samples))

        self.w_old = w.copy()
        t_old = 1.

        while callback():
            signal = np.concatenate([
                np.array([np.convolve(w_k, d_k, mode="full")
                          for w_k, d_k in zip(w[:, :, i], self.D)]
                         ).sum(axis=0)[:, None]
                for i in range(w.shape[2])
            ], axis=1)
            diff = signal - self.y
            grad = np.concatenate([
                np.array([np.correlate(diff[:, i], d_k, mode="valid")
                          for d_k in self.D])[:, :, None]
                for i in range(diff.shape[1])
            ], axis=2)
            w -= step_size * grad
            if self.positive:
                w = np.maximum(0, w - mu)
            else:
                w -= np.clip(w, -mu, mu)

            t = 0.5 * (1 + np.sqrt(1 + 4 * t_old * t_old))
            z = w + ((t_old-1) / t) * (w - self.w_old)
            self.w_old = w.copy()
            t_old = t
            w = z

    # Return the solution estimate computed.
    def get_result(self):
        return dict(theta=self.w_old)
