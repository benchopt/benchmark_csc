from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from alphacsc.update_z_multi import update_z_multi


class Solver(BaseSolver):
    name = 'alphacsc'  # alphacsc

    install_cmd = 'conda'
    requirements = ['numpy', 'cython', 'pip:alphacsc']

    # Store the information to compute the objective. The parameters of this
    # function are the keys of the dictionary obtained when calling
    # ``Objective.to_dict``.
    def set_objective(self, X, y, lmbd):
        self.X = X[:, None]
        self.y = np.transpose(y, (1, 0))[:, None]
        self.lmbd = lmbd

    # Main function of the solver, which computes a solution estimate.
    def run(self, n_iter):

        w, *_ = update_z_multi(
            self.y, self.X[:, None], self.lmbd, solver="lgcd",
            solver_kwargs=dict(max_iter=n_iter, tol=1e-12),
            n_jobs=1
        )
        self.w = np.transpose(w, (1, 2, 0))

    # Return the solution estimate computed.
    def get_result(self):
        return self.w
