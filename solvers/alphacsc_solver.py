from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from alphacsc.update_z import update_z


class Solver(BaseSolver):
    name = 'alphacsc'  # alphacsc

    install_cmd = 'conda'
    requirements = ['numpy', 'cython', 'pip:alphacsc']

    # Store the information to compute the objective. The parameters of this
    # function are the keys of the dictionary obtained when calling
    # ``Objective.to_dict``.
    def set_objective(self, D, y, lmbd):
        self.D = D
        self.y = y
        self.lmbd = lmbd

    # Main function of the solver, which computes a solution estimate.
    def run(self, n_iter):
        """
        Univariate sparse coding based on alphacsc

        Parameters
        ----------
        n_iter : int
            Number of iterations
        """

        w = update_z(
            self.y, self.D, self.lmbd, solver="fista",
            solver_kwargs=dict(max_iter=n_iter, tol=1e-12)
        )
        self.w = np.transpose(w, (1, 0, 2))

    # Return the solution estimate computed.
    def get_result(self):
        return self.w
