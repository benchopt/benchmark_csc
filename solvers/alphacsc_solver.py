import numpy as np

from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from alphacsc.update_z import update_z


class Solver(BaseSolver):
    name = 'alphacsc'  # alphacsc

    install_cmd = 'conda'
    requirements = ['pip:numpy', 'pip:cython', 'pip:alphacsc']

    # Store the information to compute the objective. The parameters of this
    # function are the keys of the dictionary obtained when calling
    # ``Objective.to_dict``.
    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

    # Main function of the solver, which computes a solution estimate.
    def run(self, n_iter):

        self.w = update_z(self.y.T, self.X, self.lmbd, solver="fista")
        self.w = np.transpose(self.w, (0, 2, 1))

    # Return the solution estimate computed.
    def get_result(self):
        return self.w
