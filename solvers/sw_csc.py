# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#         Laurent Dragoni
#
# License: MIT License

from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:

    import numpy as np
    import spams
    import celer


class Solver(BaseSolver):
    name = 'sliding_windows'  # alphacsc

    install_cmd = 'conda'
    requirements = ['pip:celer']
    stopping_strategy = 'tolerance'
    parameters = {
        'window': ['full', 'sliding'],
        'solver': ['celer']
    }

    def skip(self, D, y, lmbd, positive):
        if positive:
            return True, "SW can only handle positive=False"

        return False, None

    # Store the information to compute the objective. The parameters of this
    # function are the keys of the dictionary obtained when calling
    # ``Objective.to_dict``.
    def set_objective(self, D, y, lmbd, positive):
        self.D = np.transpose(D[:, None, :], (1, 2, 0))
        self.y = y[None, :, 0]
        self.lmbd = lmbd

    # Main function of the solver, which computes a solution estimate.
    def run(self, tol):

        itermax = 10000

        if self.window == 'full':
            self.w = working_set_convolutional(
                self.y, self.D, self.lmbd, itermax=itermax,
                verbose=False, kkt_stop=tol, log=False,
                solver=self.solver
            )

        elif self.window == 'sliding':
            self.w = sliding_window_working_set(
                self.y, self.D, self.lmbd, itermax=itermax,
                verbose=False, kkt_stop=tol, log=False,
                solver=self.solver
            )

    # Return the solution estimate computed.
    def get_result(self):
        return np.reshape(
            self.w, (self.D.shape[2], self.y.shape[1], 1)
        )[:, :-self.D.shape[1]+1, :]


fmt_verb = '| {:4d} | {:4d} | {:1.5e} |'
fmt_verb2 = '| {:4s} | {:4s} | {:11s} |'


def solve_lasso(y, H, a0, lambd, mode="C", verbose=False, solver='celer'):

    """
    Wrapper of spams.fistaFlat for the Lasso.

    Parameters
    ----------
        y : vector
         vector of observations (size: ET),
        H : matrix
         design matrix (size: ETxNT),
        a0 : vector
         first iterate (size: NT),
        lambd : float
         regularization parameter of the Lasso,
        mode : string
         "C" by default, "F" if the inputs are already in Fortran mode,
        verbose : boolean, optional
         verbose mode of spams.fistaFlat, False by default.

    Returns
    -------
        vector,
         solution of the Lasso given by spams.fistaFlat (size: NT).
    """
    if solver == 'spams':

        if mode == "C":
            # then we convert everything in Fortran mode
            return spams.fistaFlat(
                y.astype(np.float64, order='F'),
                H.astype(np.float64, order='F'),
                a0.astype(np.float64, order='F'),
                loss='square', regul='l1',
                lambda1=lambd, verbose=verbose
            )
        elif mode == "F":
            # inputs are already in Fortran mode
            return spams.fistaFlat(
                y, H, a0, loss='square', regul='l1',
                lambda1=lambd, verbose=verbose
            )
        else:
            raise ValueError('mode="C" or "F"')
    elif solver == 'celer':
        # we need to use lambd / n_samples as the l2 loss is scaled by this
        # term in celer.
        clf = celer.Lasso(
            lambd / y.shape[0], warm_start=True,
            fit_intercept=False
        )
        clf.coef_ = a0[:, 0]
        clf.fit(H, y[:, 0])

        return clf.coef_[:, None]
    else:
        raise ValueError('Bad solver')


def optimality_conditions(y, H, a):
    """Computation of the optimality conditions of the Lasso.

    This computation simply performs products of big matrices without taking
    advantage of the particular structure of H, not efficient in practice.

    Parameters
    ----------
    y : vector
        vector of observations (size: ET),
    H : matrix
        design matrix (size: ET, NT),
    a : vector
        current estimation of the activations (size: NT).

    Returns
    -------
    vector,
        the optimality conditions for the current estimation of
        the activations (size: NT).
    """
    return np.dot(H.T, (np.dot(H, a) - y))


def H_column_full(W, T, j):

    """
    Computes the full j-th column of matrix H.

    Parameters
    ----------
        W : matrix
         matrix of the shapes of the action potentials (size: E, l, N),
        T : int
         number of time steps,
        j : int
         number of the column in H we want to compute.

    Returns
    -------
        Hj : vector
         the j-th column of matrix H (size: ET).
    """

    E = W.shape[0]  # number of electrodes
    # number of sampling times for the shapes of the action potentials
    L = W.shape[1]

    r = int(j / T)  # neuron corresponding to j
    time = int(j % T)  # time corresponding to j

    # column storing the relevant action potential shapes
    Hj = np.zeros((E*T, 1))

    for p in range(E):
        shape_p = W[p, :, r]
        Hj[p*T+time:np.min((p*T+time+L, (p+1)*T))] = (
            shape_p[0:np.min((p*T+time+L, (p+1)*T)) - (p*T+time), None]
        )

    return Hj


def H_column_window(W, w, neuron, time):

    """Computes the column of matrix H.

    This corresponds to the given neuron and time, for a temporal
    window of size w.

    Parameters
    ----------
    W : matrix
        matrix of the shapes of the action potentials (size: E, l, N),
    w : int
        size of the temporal window,
    neuron : int
        neuron number of the desired column,
    time : int
        time of the desired column.

    Returns
    -------
    Hj : vector
        the column of matrix H which corresponds to the given neuron and time,
        for a temporal window of size w (size: Ew).
    """

    E = W.shape[0]  # number of electrodes
    # number of sampling times for the shapes of the action potentials
    L = W.shape[1]

    Hj = np.zeros(E*w)  # column storing the relevant action potential shapes

    for p in range(E):

        shape_p = W[p, :, neuron]
        left_bound = p*w+time
        right_bound = np.min((p*w+time+L, (p+1)*w))
        length = right_bound - left_bound
        Hj[left_bound:right_bound] = shape_p[0:length]

    return Hj


def optimality_conditions_corr(y, H, a, W, T):

    """Computation of the optimality conditions of the Lasso.

    This uses efficient product $H.T * (Hx - y)$ as a correlation.

    Parameters
    ----------
    y : vector
        vector of observations (size: ET),
    H : matrix
        design matrix (size: ET, NT),
    a : vector
        current estimation of the activations (size: NT),
    W : matrix
        matrix of the shapes of the action potentials (size: E, l, N),
    T : int
        number of time steps.

    Returns
    -------
    grad : vector
        the vector of optimality conditions for the current estimation of
        the activations (size: NT).
    """

    E = W.shape[0]  # number of electrodes
    L = W.shape[1]  # number of sampling times for the action potentials
    N = W.shape[2]  # number of neurons
    R = np.dot(H, a) - y  # residual
    grad = np.zeros((N*T, 1))

    for r in range(N):
        for p in range(E):
            shapepr = W[p, :, r]  # shape of neuron r on electrode d
            Rp = R[p*T: (p+1)*T, 0]
            grad[r*T: (r+1)*T, 0] += np.correlate(
                Rp, shapepr, mode="full"
            )[L-1:]

    return grad


def optimality_conditions_corr_window(y, H, a, W, w):

    """Computation of the optimality conditions of the Lasso.

    Restricted on a temporal window of size w, wth efficient product
    $H.T * (Hx - y)$ as a correlation.

    Parameters
    ----------
    y : vector
        vector of observations (size: ET),
    H : matrix
        design matrix (size: ET, NT),
    a : vector
        current estimation of the activations (size: NT),
    W : matrix
        matrix of the shapes of the action potentials (size: E, l, N),
    w : int
        size of the temporal window.

    Returns
    -------
    grad : vector
        the optimality conditions for the current estimation of the
        activations, restricted on a temporal window of size w (size: Nw).
    """

    E = W.shape[0]  # number of electrodes
    # number of sampling times for the shapes of the action potentials
    L = W.shape[1]
    N = W.shape[2]  # number of neurons
    R = np.dot(H, a) - y  # residual
    grad = np.zeros((N, w))

    for r in range(N):
        for p in range(E):
            shapepr = W[p, :, r]  # shape of neuron r on electrode p
            Rp = R[p*w:(p+1)*w, 0]
            grad[r, :] += np.correlate(Rp, shapepr, mode="full")[L-1:]

    return grad


def generic_working_set(S, H, N, lambd, itermax=1000, verbose=False,
                        kkt_stop=1e-3, log=False):

    """Generic working set.

    Naive computations of the optimality conditions.
    Works with the whole matrix H: very high memory cost in high dimension.

    Parameters
    ----------
    S : matrix
        matrix of the measured signals (size: E, T),
    H : matrix
        design matrix (size: ET, NT),
    N : int
        number of neurons,
    lambd : float
        regularization parameter of the Lasso,
    itermax : int, optional
        maximal number of iterations,
    verbose : boolean, optional
        print optimality conditions and iterations if set to True,
    kkt_stop : float, optional
        tolerance parameter for the stopping criterion based on the
        optimality conditions,
    log : boolean, optional
        also returns LOG if set to True.

    Returns
    ----------
    asol : vector,
        estimated activations (size: NT),
    LOG : dict, optional
        informations about the execution of the function.
    """

    T = S.shape[1]  # number of time steps

    # Vectorization step
    y = S.reshape((-1, 1)).astype(np.float64, order='F')  # signal vector
    # vector candidate as solution of the initial problem
    asol = np.zeros((N*T, 1)).astype(np.float64, order='F')

    # Computation of the optimality conditions and initialization
    # of the working set
    gd = optimality_conditions(y, H, asol)
    new_index = int(np.argmax(np.abs(gd)))
    J = [new_index]  # working set

    kkt_viol = [np.abs(gd[new_index])]

    loop = True
    niter = 1

    while loop:

        # Update of the Lasso
        Htilde = H[:, J]  # reduction of the problem on J
        atilde0 = asol[J]
        # computation of the solution on the subproblem
        atildesol = solve_lasso(y, Htilde, atilde0, lambd, mode="F")

        # Computation of the optimality conditions
        gd = np.dot(H.T, (np.dot(Htilde, atildesol) - y))
        # Remove the coordinates already in J
        gd[J, 0] = 0

        # Checking the violation of the optimality conditions
        ind = np.argmax(np.abs(gd), axis=0)[0]
        viol = abs(gd[ind])[0]

        kkt_viol.append(viol)

        if verbose:
            if ((niter-1) % 20) == 0:
                print(fmt_verb2.format('It', 'N AS', 'KKT viol'))
                print('-'*len(fmt_verb2.format('It', 'NbAS', 'KKT viol')))
            print(fmt_verb.format(niter, np.sum(np.abs(atildesol) > 0),
                                  viol / lambd))

        if viol > lambd*(1+kkt_stop):
            J.append(ind)
        else:
            loop = False
            if verbose:
                print('Convergence in optimality conditions')

        if niter >= itermax:
            loop = False
            if verbose:
                print('Max iteration reached')

        niter += 1

    asol = np.zeros((N*T, 1)).astype(np.float64, order='F')
    asol[J] = atildesol  # new candidate as solution of the initial problem

    if log:
        LOG = {}
        LOG['kkt_viol'] = kkt_viol
        LOG['J'] = J
        LOG['atildesol'] = atildesol

        return asol, LOG
    else:
        return asol


def working_set_convolutional(S, W, lambd, itermax=1000, kkt_stop=1e-4,
                              verbose=False, log=False, solver='celer'):

    """
     Working set, computing the optimality conditions with the convolution.

     Parameters
    ----------
    S : matrix
        matrix of the measured signals (size: E, T),
    W : matrix
        matrix of the shapes of the action potentials (size: E, l, N),
    lambd : float
        regularization parameter of the Lasso,
    itermax : int, optional
        maximal number of iterations,
    verbose : boolean, optional
        print optimality conditions and iterations if set to True,
    kkt_stop : float, optional
        tolerance parameter for the stopping criterion based on the
        optimality conditions,
    log : boolean, optional
        also returns LOG if set to True.
    solver :str, optional
        Solver to solve lasso sub problems.

    Returns
    ----------
    asol : vector,
        estimated activations (size: NT),
    LOG : dict, optional
        informations about the execution of the function.
    """

    N = W.shape[2]  # number of neurons
    T = S.shape[1]  # number of time steps

    # Vectorization steps
    y = S.reshape((-1, 1)).astype(np.float64, order='F')  # signal vector
    Htilde = np.zeros((y.shape[0], 0)).astype(np.float64, order='F')
    atilde = np.zeros((0, 1)).astype(np.float64, order='F')

    # Initialization of the working set
    J = []  # working set
    kkt_viol = []

    loop = True
    niter = 1
    while loop:

        # Computation of the optimality condition and initialization
        # of the working set
        gd = optimality_conditions_corr(y, Htilde, atilde, W, T)
        gd[J, 0] = 0  # remove the coordinates already in the working set
        new_index = np.argmax(np.abs(gd), axis=0)[0]

        # Checking the violation of the optimality conditions
        viol = abs(gd[new_index])[0]
        kkt_viol.append(viol)
        if viol <= lambd*(1+kkt_stop):
            if verbose:
                print('Convergence in optimality conditions')
            break

        # Add this new index
        Htilde = np.column_stack(
            (Htilde, H_column_full(W, T, new_index))
        ).astype(np.float64, order='F')
        atilde = np.row_stack((atilde, 0)).astype(np.float64, order='F')

        # Extract non zero lines of Htilde
        sel = np.any(Htilde, 1)
        Htilde2 = Htilde[sel, :].astype(np.float64, order='F')
        y2 = y[sel, :].astype(np.float64, order='F')

        # Solve the Lasso with the new index
        atilde = solve_lasso(
            y2, Htilde2, atilde, lambd, mode="F", solver=solver
        )

        if verbose:
            if ((niter-1) % 20) == 0:
                print(fmt_verb2.format('It', 'N AS', 'KKT viol'))
                print('-'*len(fmt_verb2.format('It', 'NbAS', 'KKT viol')))
            print(
                fmt_verb.format(niter, np.sum(np.abs(atilde) > 0), viol/lambd)
            )

        J.append(new_index)

        if niter >= itermax:
            loop = False
            if verbose:
                print('Max iteration reached')

        niter += 1

    asol = np.zeros((N*T, 1)).astype(np.float64, order='F')
    asol[J] = atilde

    if log:
        LOG = {}
        LOG['kkt_viol'] = kkt_viol
        LOG['J'] = J
        LOG['atilde'] = atilde

        return asol, LOG
    else:
        return asol


def sliding_window_working_set(S, W, lambd, itermax=1000, kkt_stop=1e-3,
                               verbose=False, log=False, solver='celer'):

    """Sliding Window Working set.

    Presented in the article: https://doi.org/10.1007/s10440-022-00494-x.
    Solves the Lasso problem by exploring the signal with a sliding window.

     Parameters
    ----------
    S : matrix
        matrix of the measured signals (size: E, T),
    W : matrix
        matrix of the shapes of the action potentials (size: E, l, N),
    lambd : float
        regularization parameter of the Lasso,
    itermax : int, optional
        maximal number of iterations,
    verbose : boolean, optional
        print optimality conditions and iterations if set to True,
    kkt_stop : float, optional
        tolerance parameter for the stopping criterion based on the
        optimality conditions,
    log : boolean, optional
        also returns LOG if set to True.
    solver :str, optional
        Solver to solve lasso sub problems.

    Returns
    ----------
    x : vector,
        estimated activations (size: NT),
    LOG : dict, optional
        informations about the execution of the function.
    """

    E = W.shape[0]  # number of electrodes
    L = W.shape[1]  # size of the shapes
    N = W.shape[2]  # number of neurons
    T = S.shape[1]  # number of time steps

    # A solution
    A_sol = np.zeros((N, T))

    # Initial window borders
    w1 = 0  # left border of the window
    w2 = 4*L  # right border of the window

    J = []  # working set
    Omega = []  # list of windows infos
    kkt_viol = []  # violation values of the optimality conditions

    niter = 0  # while counter on a window

    S_loc = S[:, w1:w2]  # signal on the current window
    # vectorization of S_loc
    y_loc = S_loc.reshape((-1, 1)).astype(np.float64, order='F')
    # design matrix on the current window
    Htilde_loc = np.zeros((E*(w2-w1), 0)).astype(np.float64, order='F')
    # solution of the current window
    xtilde_loc = np.zeros((0, 1)).astype(np.float64, order='F')
    activ_loc = []  # list of activations represented as tuples (neuron, time)
    # Caution: variables having the "loc" suffix are related
    # to the current window. Therefore note that a local time t corresponds
    # to a global time t + w1

    loop_global = True

    while loop_global:

        loop_window = True

        while loop_window:

            if niter >= itermax:
                loop_window = False
                if verbose:
                    print('Max window iteration achieved')
            else:
                niter += 1

            gd = optimality_conditions_corr_window(
                y_loc, Htilde_loc, xtilde_loc, W, w2-w1
            )

            if w2 == T:  # if end of signal
                gd_loc = gd[:, :]  # search activation in the whole window
            else:
                # search activation not in the last l indices
                gd_loc = gd[:, :w2-w1-L]
            # gd_loc with zeroes on the coordinates already activated
            gd_loc_zeroed = gd_loc
            for activ in activ_loc:
                gd_loc_zeroed[activ[0], activ[1]] = 0.0

            new_activ = np.unravel_index(
                np.abs(gd_loc_zeroed).argmax(), gd_loc_zeroed.shape
            )
            viol = abs(gd_loc_zeroed[new_activ])
            kkt_viol.append(viol)

            # if the optimality conditions are verified, quit current window
            if viol <= lambd*(1+kkt_stop):
                if verbose:
                    print('Convergence in KKT condition')
                break
            # else use this new activation
            else:
                # neuron corresponding to this activation
                new_activ_neuron = new_activ[0]
                # local time corresponding to this activation
                new_activ_time = new_activ[1]
                activ_loc.append((new_activ_neuron, new_activ_time))
                # store the activation in the global domain
                J.append(new_activ_neuron*T + w1 + new_activ_time)

                # Ajout du nouvel indice dans les objets
                Htilde_loc = np.column_stack((
                    Htilde_loc,
                    H_column_window(W, w2-w1, new_activ_neuron, new_activ_time)
                )).astype(np.float64, order='F')
                xtilde_loc = np.row_stack(
                    (xtilde_loc, 0)
                ).astype(np.float64, order='F')

                # Sélection des lignes ne contenant pas que des zéros
                sel = np.any(Htilde_loc, 1)
                Htilde_loc2 = Htilde_loc[sel, :].astype(np.float64, order='F')
                y_loc2 = y_loc[sel, :].astype(np.float64, order='F')

                # Résolution du Lasso sur le problème réduit
                xtilde_loc = solve_lasso(
                    y_loc2, Htilde_loc2, xtilde_loc, lambd, mode="F",
                    solver=solver
                )

        # after convergence on current window, three cases:
        #   1.create new window,
        #   2.merge current window with last window,
        #   or 3.extend current window
        if activ_loc:  # if there is at least one activation

            first_activ_time_loc = min([activ[1] for activ in activ_loc])
            last_activ_time_loc = max([activ[1] for activ in activ_loc])

            # CASE 1: activations are far away from borders of current
            # window, can create a new window
            if first_activ_time_loc >= L and last_activ_time_loc <= w2-w1-2*L:

                # if there is enough space to create a new window of size 4*L,
                # do it
                if w2+3*L+1 <= T:
                    # for activ in range(len(activ_neurons_loc)):
                    for activ_idx in range(len(activ_loc)):
                        A_sol[
                            activ_loc[activ_idx][0], w1+activ_loc[activ_idx][1]
                        ] = xtilde_loc[activ_idx]
                    Omega.append([w1, w2, activ_loc])
                    w1 = w2-L+1
                    w2 = w2+3*L+1
                    niter = 0
                    S_loc = S[:, w1:w2]
                    y_loc = S_loc.reshape(
                        (-1, 1)
                    ).astype(np.float64, order='F')
                    Htilde_loc = np.zeros(
                        (E*(w2-w1), 0)
                    ).astype(np.float64, order='F')
                    xtilde_loc = np.zeros(
                        (0, 1)
                    ).astype(np.float64, order='F')
                    activ_loc = []
                # if there is not enough space to create a new window
                # of size 4*L, extend the current one to the end of
                # the signal with new w2=T
                else:
                    # converged on a window which reaches end of signal,
                    # stop global loop
                    if w2 == T:
                        loop_global = False
                        for activ_idx in range(len(activ_loc)):
                            loc = activ_loc[activ_idx]
                            A_sol[loc[0], w1+loc[1]] = xtilde_loc[activ_idx]
                    w2_new = T

                    # Create a new Htilde for the extended window
                    Htilde_loc = np.zeros(
                        (E*(w2_new-w1), 0)
                    ).astype(np.float64, order='F')
                    for activ_idx in range(len(activ_loc)):
                        loc = activ_loc[activ_idx]
                        Htilde_loc = np.column_stack((
                            Htilde_loc,
                            H_column_window(W, w2_new-w1, loc[0], loc[1])
                        )).astype(np.float64, order='F')

                    w2 = w2_new
                    niter = 0
                    S_loc = S[:, w1:w2]
                    y_loc = S_loc.reshape(
                        (-1, 1)
                    ).astype(np.float64, order='F')

            # CASE 2: at least an activation close to the left border of
            # current window, need to merge with last window
            elif (first_activ_time_loc <= L-1 and
                  last_activ_time_loc <= w2-w1-2*L):

                # If there is at least a window in Omega
                if Omega:
                    # extract the infos of the previous window from Omega
                    previous_window_infos = Omega.pop()
                    previous_w1 = previous_window_infos[0]
                    previous_activ_loc = previous_window_infos[2]

                    # recompute Htilde_loc and xtilde_loc for the merging of
                    # the windows
                    Htilde_loc = np.zeros(
                        (E*(w2-previous_w1), 0)
                    ).astype(np.float64, order='F')
                    previous_xtilde_loc = np.zeros(
                        (0, 1)
                    ).astype(np.float64, order='F')
                    # add H columns and x estimates corresponding
                    # to previous activations
                    for activ_idx in range(len(previous_activ_loc)):
                        loc = previous_activ_loc[activ_idx]
                        Htilde_loc = np.column_stack((
                            Htilde_loc,
                            H_column_window(W, w2-previous_w1, loc[0], loc[1])
                        )).astype(np.float64, order='F')
                        previous_xtilde_loc = np.row_stack((
                            previous_xtilde_loc,
                            A_sol[loc[0], previous_w1+loc[1]]
                        )).astype(np.float64, order='F')
                    # convert the activation times of the current window for
                    # the merging of the windows: just need to translate
                    # these times with the correct shift
                    activ_loc_new = [(activ[0], activ[1]+w1-previous_w1)
                                     for activ in activ_loc]
                    # add H columns of current activs, have been shifted
                    # because previous_w1 is the new left border w1
                    for activ_idx in range(len(activ_loc)):
                        loc = activ_loc_new[activ_idx]
                        Htilde_loc = np.column_stack((
                            Htilde_loc,
                            H_column_window(W, w2-previous_w1, loc[0], loc[1])
                        )).astype(np.float64, order='F')
                    # merge the two estimated vectors
                    xtilde_loc = np.row_stack((
                        previous_xtilde_loc, xtilde_loc
                    )).astype(np.float64, order='F')
                    # merge the lists of activations
                    activ_loc = previous_activ_loc + activ_loc_new
                    # new left border of the merged window
                    w1 = previous_w1
                    niter = 0
                    S_loc = S[:, w1:w2]
                    y_loc = S_loc.reshape(
                        (-1, 1)
                    ).astype(np.float64, order='F')
                # If there is no window in Omega
                else:
                    # if there is enough space to create a new window
                    # of size 4*L, do it
                    if w2+3*L+1 <= T:
                        for activ_idx in range(len(activ_loc)):
                            loc = activ_loc[activ_idx]
                            A_sol[loc[0], w1+loc[1]] = xtilde_loc[activ_idx]
                        Omega.append([w1, w2, activ_loc])
                        w1 = w2-L+1
                        w2 = w2+3*L+1
                        niter = 0
                        S_loc = S[:, w1:w2]
                        y_loc = S_loc.reshape(
                            (-1, 1)
                        ).astype(np.float64, order='F')
                        Htilde_loc = np.zeros(
                            (E*(w2-w1), 0)
                        ).astype(np.float64, order='F')
                        xtilde_loc = np.zeros(
                            (0, 1)
                        ).astype(np.float64, order='F')
                        activ_loc = []
                    # if there is not enough space to create a new window of
                    # size 4*L, extend the current one to the end of the
                    # signal with new w2=T
                    else:
                        # converged on a window which reaches end of signal,
                        # stop global loop
                        if w2 == T:
                            loop_global = False
                            for activ_idx in range(len(activ_loc)):
                                loc = activ_loc[activ_idx]
                                A_sol[loc[0], w1+loc[1]] = xtilde_loc[activ_idx]  # noqa: E501
                        w2_new = T
                        # Extend lines of Htilde with zeros to match
                        # new size of window
                        # Create a new Htilde for the extended window
                        Htilde_loc = np.zeros(
                            (E*(w2_new-w1), 0)
                        ).astype(np.float64, order='F')
                        for activ_idx in range(len(activ_loc)):
                            loc = activ_loc[activ_idx]
                            Htilde_loc = np.column_stack((
                                Htilde_loc,
                                H_column_window(W, w2_new-w1, loc[0], loc[1])
                            )).astype(np.float64, order='F')

                        w2 = w2_new
                        niter = 0
                        S_loc = S[:, w1:w2]
                        y_loc = S_loc.reshape(
                            (-1, 1)
                        ).astype(np.float64, order='F')

            # CASE 3: at least an activation close to the right border of
            # current window, extend it
            elif (L <= first_activ_time_loc and
                  w2-w1-2*L+1 <= last_activ_time_loc):
                # converged on a window which reaches end of signal,
                # stop global loop
                if w2 == T:
                    loop_global = False
                    for activ_idx in range(len(activ_loc)):
                        loc = activ_loc[activ_idx]
                        A_sol[loc[0], w1+loc[1]] = xtilde_loc[activ_idx]
                w2_new = min(w2+L, T)

                # Create a new Htilde for the extended window
                Htilde_loc = np.zeros(
                    (E*(w2_new-w1), 0)
                ).astype(np.float64, order='F')
                for activ_idx in range(len(activ_loc)):
                    loc = activ_loc[activ_idx]
                    Htilde_loc = np.column_stack((
                        Htilde_loc,
                        H_column_window(W, w2_new-w1, loc[0], loc[1])
                    )).astype(np.float64, order='F')

                w2 = w2_new
                niter = 0
                S_loc = S[:, w1:w2]
                y_loc = S_loc.reshape((-1, 1)).astype(np.float64, order='F')

            # CASE 4
            elif (first_activ_time_loc <= L-1 and
                  w2-w1-2*L+1 <= last_activ_time_loc):
                # boolean indicating if the current window has been fusioned
                # with the previous one
                fusioned = 0

                # If there is at least a window in Omega
                if Omega:
                    fusioned = 1
                    # extract the infos of the previous window from Omega
                    previous_window_infos = Omega.pop()
                    previous_w1 = previous_window_infos[0]
                    previous_activ_loc = previous_window_infos[2]

                    # recompute Htilde_loc and xtilde_loc for the merging
                    # of the windows
                    Htilde_loc = np.zeros(
                        (E*(w2-previous_w1), 0)
                    ).astype(np.float64, order='F')
                    previous_xtilde_loc = np.zeros(
                        (0, 1)
                    ).astype(np.float64, order='F')
                    # add H columns and x estimates corresponding
                    # to previous activations
                    for activ_idx in range(len(previous_activ_loc)):
                        loc = previous_activ_loc[activ_idx]
                        Htilde_loc = np.column_stack((
                            Htilde_loc,
                            H_column_window(W, w2-previous_w1, loc[0], loc[1])
                        )).astype(np.float64, order='F')
                        previous_xtilde_loc = np.row_stack((
                            previous_xtilde_loc,
                            A_sol[loc[0], previous_w1+loc[1]]
                        )).astype(np.float64, order='F')
                    # convert the activation times of the current window
                    # for the merging of the windows: just need to translate
                    # these times with the correct shift
                    activ_loc_new = [(activ[0], activ[1]+w1-previous_w1)
                                     for activ in activ_loc]
                    # add H columns of current activs, have been shifted
                    # because previous_w1 is the new left border w1
                    for activ_idx in range(len(activ_loc)):
                        loc = activ_loc_new[activ_idx]
                        Htilde_loc = np.column_stack((
                            Htilde_loc,
                            H_column_window(W, w2-previous_w1, loc[0], loc[1])
                        )).astype(np.float64, order='F')
                    # merge the two estimated vectors
                    xtilde_loc = np.row_stack((
                        previous_xtilde_loc, xtilde_loc
                    )).astype(np.float64, order='F')
                    # merge the lists of activations activ_times_loc_new
                    activ_loc = previous_activ_loc + activ_loc_new
                    # new left border of the merged window
                    w1 = previous_w1
                    niter = 0
                    S_loc = S[:, w1:w2]
                    y_loc = S_loc.reshape(
                        (-1, 1)
                    ).astype(np.float64, order='F')

                # Want to extend the window

                # converged on a window which reaches end of signal,
                # and did not fusion with previous window, stop global loop
                if w2 == T:
                    if not fusioned:
                        loop_global = False
                        for activ_idx in range(len(activ_loc)):
                            loc = activ_loc[activ_idx]
                            A_sol[loc[0], w1+loc[1]] = xtilde_loc[activ_idx]
                w2_new = min(w2+L, T)

                # Create a new Htilde for the extended window
                Htilde_loc = np.zeros(
                    (E*(w2_new-w1), 0)
                ).astype(np.float64, order='F')
                for activ_idx in range(len(activ_loc)):
                    loc = activ_loc[activ_idx]
                    Htilde_loc = np.column_stack((
                        Htilde_loc,
                        H_column_window(W, w2_new-w1, loc[0], loc[1])
                    )).astype(np.float64, order='F')

                w2 = w2_new
                niter = 0
                S_loc = S[:, w1:w2]
                y_loc = S_loc.reshape((-1, 1)).astype(np.float64, order='F')

            # CASE 5: should not enter here
            else:
                print('Warning: Case 5')
                print("first bound=", L)
                print("first_activ_time_loc=", first_activ_time_loc)
                print("last_activ_time_loc=", last_activ_time_loc)
                print("last bound=", w2-w1-2*L)
                break
        # if there is no activation
        else:
            # if there is enough space to create a new window of size 4*L,
            # do it
            if w2+3*L+1 <= T:
                Omega.append([w1, w2, []])
                w1 = w2-L+1
                w2 = w2+3*L+1
                niter = 0
                S_loc = S[:, w1:w2]
                y_loc = S_loc.reshape((-1, 1)).astype(np.float64, order='F')
                Htilde_loc = np.zeros(
                    (E*(w2-w1), 0)
                ).astype(np.float64, order='F')
                xtilde_loc = np.zeros((0, 1)).astype(np.float64, order='F')
                activ_loc = []
            # if there is not enough space to create a new window of size 4*L,
            # extend the current one to the end of the signal with new w2=T
            else:
                # converged on a window which reaches end of signal,
                # stop global loop
                if w2 == T:
                    loop_global = False
                    for activ_idx in range(len(activ_loc)):
                        loc = activ_loc[activ_idx]
                        A_sol[loc[0], w1+loc[1]] = xtilde_loc[activ_idx]
                w2_new = T

                # Create a new Htilde for the extended window
                Htilde_loc = np.zeros(
                    (E*(w2_new-w1), 0)
                ).astype(np.float64, order='F')
                for activ_idx in range(len(activ_loc)):
                    loc = activ_loc[activ_idx]
                    Htilde_loc = np.column_stack((
                        Htilde_loc,
                        H_column_window(W, w2_new-w1, loc[0], loc[1])
                    )).astype(np.float64, order='F')

                w2 = w2_new
                niter = 0
                S_loc = S[:, w1:w2]
                y_loc = S_loc.reshape((-1, 1)).astype(np.float64, order='F')

    x = A_sol.reshape((-1, 1)).astype(np.float64, order='F')

    if log:
        LOG = {}
        LOG['kkt_viol'] = kkt_viol
        LOG['J'] = J
        LOG['xtilde'] = xtilde_loc
        return x, LOG
    else:
        return x
