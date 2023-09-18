from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from sporco.admm.cbpdn import ConvBPDN


class Solver(BaseSolver):
    name = 'sporco'

    install_cmd = 'conda'
    requirements = ['pip:sporco']

    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy='iteration'
    )

    # Store the information to compute the objective. The parameters of this
    # function are the keys of the dictionary obtained when calling
    # ``Objective.to_dict``.
    def set_objective(self, D, y, lmbd, positive):
        self.D = D.T[:, None]  # shape (n_times_atom, n_channel, n_atoms)
        self.y = y[:, None]  # shape (n_times, n_channel, n_samples)
        self.lmbd = lmbd
        self.positive = positive

    # Main function of the solver, which computes a solution estimate.
    def run(self, n_iter):

        opt = ConvBPDN.Options({
            'Verbose': False, 'MaxMainIter': n_iter, 'FastSolve': True,
            'RelStopTol': 0, 'AbsStopTol': 0, 'AuxVarObj': False,
            'NonNegCoef': self.positive
        })
        cbpdn = ConvBPDN(
            self.D, self.y, self.lmbd, opt, dimN=1
        )
        self.w = cbpdn.solve()  # shape (n_times, n_samples, n_atoms)

    def get_result(self):
        # truncate codes as the objective is defined for full convolution and
        # not 'same' and reorder axis for compat with objective.
        return dict(theta=np.transpose(
            self.w[:-self.D.shape[0]+1, 0], (2, 0, 1)
        ))
