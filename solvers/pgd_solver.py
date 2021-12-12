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
        """
        PGD with FISTA

        Parameters
        ----------
        callback : function
            callback from benchopt
        """
        # Computes Lipschitz upper bound with FFT
        fourier_dico = fft(self.D, axis=1)
        L = np.max(np.real(fourier_dico * np.conj(fourier_dico)), axis=1).sum()

        # Step size and regularization
        step_size = 1. / L
        mu = step_size * self.lmbd

        n_atoms = self.D.shape[0]
        n_samples, signal_length = self.y.shape
        kernel_size = self.D.shape[1]
        w = np.zeros((n_samples, n_atoms, signal_length - kernel_size + 1))

        iterate_old = w.copy()
        t_old = 1.

        while callback(iterate_old):
            # Computes the gradient
            signal = np.concatenate([
                np.array([np.convolve(w_k, d_k, mode="full")
                          for w_k, d_k in zip(w[i, :, :], self.D)]
                         ).sum(axis=0).reshape(1, -1)
                for i in range(w.shape[0])
            ], axis=0)
            diff = signal - self.y
            grad = np.concatenate([
                np.array([np.correlate(diff[i, :], d_k, mode="valid")
                          for d_k in self.D])[None, :, :]
                for i in range(diff.shape[0])
            ], axis=0)

            # Gradient descent
            w -= step_size * grad

            # Positive sparse codes
            iterate = np.maximum(0, w - mu)

            # FISTA
            t = 0.5 * (1. + np.sqrt(1. + 4. * t_old * t_old))
            w = iterate + ((t_old - 1.) / t) * (iterate - iterate_old)
            iterate_old = iterate.copy()
            t_old = t

        self.w = iterate_old

    # Return the solution estimate computed.
    def get_result(self):
        return self.w
