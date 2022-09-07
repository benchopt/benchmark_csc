from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from alphacsc.update_z_multi import update_z_multi


class Solver(BaseSolver):
    name = 'alphacsc'  # alphacsc

    install_cmd = 'conda'
    requirements = ['numpy', 'cython', 'pip:alphacsc']

    parameters = {
        'solver': ['lgcd', 'fista', 'l-bfgs']
    }

    # Store the information to compute the objective. The parameters of this
    # function are the keys of the dictionary obtained when calling
    # ``Objective.to_dict``.
    def set_objective(self, D, y, lmbd):
        self.D = D[:, None]  # shape (n_atoms, n_channels, n_times_atom)
        self.y = np.transpose(  # shape (n_samples, n_channels, n_times)
            y, (1, 0)
        )[:, None]
        self.lmbd = lmbd

        self.solver_kwargs = {}
        if self.solver == 'lgcd':
            self.solver_kwargs['tol'] = 1e-12

    # Main function of the solver, which computes a solution estimate.
    def run(self, n_iter):
        self.solver_kwargs['max_iter'] = n_iter

        self.w, *_ = update_z_multi(
            self.y, self.D, self.lmbd, solver=self.solver,
            solver_kwargs=self.solver_kwargs,
            n_jobs=1
        )  # shape (n_samples, n_atoms, n_times_valid)

    # Return the solution estimate computed.
    def get_result(self):
        return np.transpose(self.w, (1, 2, 0))
