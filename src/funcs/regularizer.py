import numpy as np
from numba import jit
from numba.typed import List
from scipy.sparse import coo_matrix
import spams
import sys
sys.path.append("../")
from src.utils import l2_norm


class GL1:
    def __init__(self, groups, penalty=None, weights=None):
        """
        !!Warning: need `groups` be ordered in a consecutive manner, i.e.,
        groups: array([1., 1., 1., 2., 2., 2., 3., 3., 3., 3.])
        Then:
        unique_groups: array([1., 2., 3.])
        group_frequency: array([3, 3, 4]))
        """
        self.penalty = penalty
        if penalty is None:
            raise ValueError("Initialization failed!")
        if type(groups) == np.ndarray:
            self.unique_groups, self.group_frequency = np.unique(
                groups, return_counts=True)
            if weights is not None:
                self.weights = weights
            else:
                self.weights = np.sqrt(self.group_frequency)
            self.weights = penalty * self.weights
            self.K = len(self.unique_groups)
            self.group_size = -1 * np.ones(self.K)
            p = groups.shape[0]
            full_index = np.arange(p)
            starts = []
            ends = []
            for i in range(self.K):
                G_i = full_index[np.where(groups == self.unique_groups[i])]
                # record the `start` and `end` indices of the group G_i to avoid fancy indexing in numpy
                # in the example above, the start index and end index for G_1 is 0 and 2 respectively
                # since python `start:end` will include `start` and exclude `end`, so we will add 1 to the `end`
                # so the G_i-th block of X is indexed by X[start:end]
                start, end = min(G_i), max(G_i) + 1
                starts.append(start)
                ends.append(end)
                self.group_size[i] = end - start
            # wrap as np.array for jit compile purpose
            self.starts = np.array(starts)
            self.ends = np.array(ends)
        else:
            print("Use fast initialization for the GL1 object", flush=True)
            if weights is not None:
                self.weights = weights
            else:
                self.weights = np.sqrt(groups['group_frequency'])
            self.weights = penalty * self.weights
            self.K = len(groups['group_frequency'])
            self.group_size = groups['group_frequency']
            self.starts = groups['starts']
            self.ends = groups['ends']
            self.unique_groups = np.linspace(1, self.K, self.K)

    def __str__(self):
        return("Group L1")

    def func(self, X):
        """
            X here is not the data matrix but the variable instead
        """
        return self._func_jit(X, self.K, self.starts, self.ends, self.weights)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _func_jit(X, K, starts, ends, weights):
        fval = 0.0
        for i in range(K):
            start, end = starts[i], ends[i]
            XG_i = X[start:end]
            # don't call l2_norm for jit to complie
            fval += weights[i] * np.sqrt(np.dot(XG_i.T, XG_i))[0][0]
        return fval

    def grad(self, X):
        """
            compute the gradient. If evaluate at the group whose value is 0, then
            return np.inf for that group
        """
        return self._grad_jit(X, self.K, self.starts, self.ends, self.weights)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _grad_jit(X, K, starts, ends, weights):
        grad = np.full(X.shape, np.inf)
        for i in range(K):
            start, end = starts[i], ends[i]
            XG_i = X[start:end]
            norm_XG_i = np.sqrt(np.dot(XG_i.T, XG_i))[0][0]
            if (np.abs(norm_XG_i) > 1e-15):
                grad[start:end] = (weights[i] / norm_XG_i) * XG_i
        return grad

    def _prepare_hv_data(self, X, subgroup_index):
        """
        make sure the groups in subgroup_index are non-zero
        """
        self.hv_data = {}
        start = 0
        for i in subgroup_index:
            start_x, end_x = self.starts[i], self.ends[i]
            XG_i = X[start_x:end_x]
            XG_i_norm = np.sqrt(np.dot(XG_i.T, XG_i))[0][0]
            end = start + end_x - start_x
            self.hv_data[i] = {}
            self.hv_data[i]['XG_i'] = XG_i
            self.hv_data[i]['XG_i_norm'] = XG_i_norm
            self.hv_data[i]['start'] = start
            self.hv_data[i]['end'] = end
            self.hv_data[i]['XG_i_norm_cubic'] = XG_i_norm**3
            start = end

    def hessian_vector_product_fast(self, v, subgroup_index):
        """
        call _prepare_hv_data before call hessian_vector_product_fast
        """
        hv = np.empty_like(v)
        for i in subgroup_index:
            start = self.hv_data[i]['start']
            end = self.hv_data[i]['end']
            vi = v[start:end]
            temp = np.matmul(self.hv_data[i]['XG_i'].T, vi)
            hv[start:end] = self.weights[i] * (1 / self.hv_data[i]['XG_i_norm'] * vi -
                                               (temp / self.hv_data[i]['XG_i_norm_cubic']) *
                                               self.hv_data[i]['XG_i'])
        return hv

    def _dual_norm(self, y):
        """
            compute the dual of r(x), which is r(y): max ||y_g||/lambda_g
            reference: https://jmlr.org/papers/volume18/16-577/16-577.pdf section 5.2
        """
        return self._dual_norm_jit(y, self.K, self.starts, self.ends, self.weights)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _dual_norm_jit(y, K, starts, ends, weights):
        max_group_norm = 0.0
        for i in range(K):
            start, end = starts[i], ends[i]
            yG_i = y[start:end]
            temp_i = (np.sqrt(np.dot(yG_i.T, yG_i))[0][0]) / weights[i]
            max_group_norm = max(max_group_norm, temp_i)
        return max_group_norm

    ##############################################
    #      exact proximal gradient calculation   #
    ##############################################
    def compute_proximal_gradient_update(self, xk, alphak, dk):
        return self._compute_proximal_gradient_update_jit(xk, alphak, dk, self.starts,
                                                          self.ends, self.weights)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _compute_proximal_gradient_update_jit(X, alpha, gradf, starts, ends, weights):
        proximal = np.zeros_like(X)
        nonZeroGroup = []
        zeroGroup = []
        for i in range(len(starts)):
            start, end = starts[i], ends[i]
            XG_i = X[start:end]
            gradfG_i = gradf[start:end]
            gradient_step = XG_i - alpha * gradfG_i
            gradient_step_norm = np.sqrt(
                np.dot(gradient_step.T, gradient_step))[0][0]
            if gradient_step_norm != 0:
                temp = 1 - ((weights[i] * alpha) / gradient_step_norm)
            else:
                temp = -1
            if temp > 0:
                nonZeroGroup.append(i)
            else:
                zeroGroup.append(i)
            proximal[start:end] = max(temp, 0) * gradient_step
        return proximal, len(zeroGroup), len(nonZeroGroup)

    def _get_group_structure(self, X):
        return self._get_group_structure_jit(X, self.K, self.starts, self.ends)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _get_group_structure_jit(X, K, starts, ends):
        nz = 0
        for i in range(K):
            start, end = starts[i], ends[i]
            X_Gi = X[start:end]
            if (np.sum(np.abs(X_Gi)) == 0):
                nz += 1
        nnz = K - nz
        return nnz, nz


class NatOG:
    def __init__(self, groups, penalty, config, weights=None):
        """
            groups: [[0,1,3,4], [0,2,4,5], [1,3,6]]
                index begins with 0
                all coordinates need to be included
                within each group the index needs to be sorted
        """
        self.config = config
        if type(groups) != list:
            groups = groups['groups']
        if weights is not None:
            assert len(groups) == len(weights), "groups and weights should be of the same length"
            assert isinstance(weights, (np.ndarray, np.generic)), "weights should be a numpy array"
        else:
            weights = np.array([np.sqrt(len(g)) for g in groups])
        self.penalty = penalty            
        self.weights = self.penalty * weights
        self.K = len(groups)
        # group structure to matrix representation
        self.lifted_dimension = 0
        self.groups_dict = {}
        groups_flattern = []
        group_size = []
        for (i, g) in enumerate(groups):
            groups_flattern += g
            self.lifted_dimension += len(g)
            group_size.append(len(g))
            self.groups_dict[i] = g
        self.groups = List()
        for g in groups:
            self.groups.append(np.array(g))
        # actual dimension
        p = max(groups_flattern) + 1
        self.xdim = p
        rows, cols = [], []
        for (colidx, rowidx) in enumerate(groups_flattern):
            rows.append(rowidx)
            cols.append(colidx)
        vals = [1.0] * len(rows)
        self.A = coo_matrix((vals, (rows, cols)),shape=(p, self.lifted_dimension))
        self.A = self.A.tocsc()
        self.ATA = self.A.T @ self.A

        # relabel groups in lifted space
        self.starts, self.ends = [], []
        start = 0
        for i in group_size:
            self.starts.append(start)
            end = start + i
            self.ends.append(end)
            start = end
        self.starts, self.ends = np.array(self.starts), np.array(self.ends)

    def __str__(self):
        return("Overlapping Group L1")

    def func(self, x):
        return self._func_jit(x, self.groups, self.weights)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _func_jit(x, groups, weights):
        ans = 0.0
        for i, g in enumerate(groups):
            xg = x[g]
            ans += np.sqrt(np.dot(xg.T, xg))[0][0] * weights[i]
        return ans

    ##############################################
    #    inexact proximal gradient calculation   #
    ##############################################

    def compute_inexact_proximal_gradient_update(self, xk, alphak, dk, y_init, stepsize_init, ipg_kwargs):
        """
            implement the fixed stepsize projected  gradient descent
        """
        uk = xk - alphak * dk
        ATuk = self.A.T @ uk
        if y_init is None:
            y_current = np.zeros((self.lifted_dimension, 1))
        else:
            assert y_init.shape[0] == self.lifted_dimension, f'y_init dimension:{y_init.shape[0]} mismacth with the desired lifted dimenson:{self.lifted_dimension}'
            y_current = y_init
        if stepsize_init is None:
            self.stepsize = 1 / alphak
        else:
            self.stepsize = stepsize_init
        self.inner_its = 0
        if not self.config.exact_pg_computation:
            if self.config.ipg_strategy == 'diminishing':
                k = ipg_kwargs['iteration']
                self.targap = self.config.ipg_diminishing_c * np.log(k+1) / (k+1)**self.config.ipg_diminishing_delta
            else:
                raise ValueError(f"Unrecognized ipg_strategy value:{self.config.ipg_strategy}")
        else:
            self.targap = self.config.exact_pg_computation_tol

        dual_val_ycurrent = self.prox_dual(self.A @ y_current, uk, alphak)[0][0]
        grad_psi_ycurrent = (alphak * self.ATA @ y_current + ATuk)
        self.total_bak = 0
        while True:
            self.inner_its += 1
            # perform arc search to find suitable stepsize
            bak = 0
            if self.config.ipg_do_linesearch:
                while True:
                    ytrial, projected_group = self._proj_norm_ball(y_current - self.stepsize * grad_psi_ycurrent)
                    dual_val = self.prox_dual(self.A @ ytrial, uk, alphak)[0][0]
                    LHS = -(dual_val - dual_val_ycurrent)
                    RHS = (self.config.ipg_linesearch_eta *(grad_psi_ycurrent.T @ (ytrial - y_current)))[0][0]
                    if (LHS <= RHS) or (np.abs(np.abs(LHS) - np.abs(RHS)) < 1e-15):
                        self.flag = 'numertol'
                        self.total_bak += bak
                        break
                    if self.stepsize < 1e-15:
                        self.flag = 'smallstp'
                        self.total_bak += bak
                        break
                    self.stepsize *= self.config.ipg_linesearch_xi
                    bak += 1
            else:
                ytrial, projected_group = self._proj_norm_ball(y_current - self.stepsize * grad_psi_ycurrent)
                dual_val = self.prox_dual(self.A @ ytrial, uk, alphak)[0][0]

            # get the primal approximate solution from the dual
            xtrial = alphak * (self.A @ ytrial) + uk
            # attempt to project the primal approximate solution
            xtrial_proj = xtrial.copy()
            for i in range(self.K):
                if i not in projected_group:
                    xtrial_proj[self.groups_dict[i]] = 0.0
            ######################### check for termination ###############################
            # first check the projected primal
            rxtrial_proj = self.func(xtrial_proj)
            primal_val_proj = self.prox_primal(xtrial_proj, uk, alphak, rxtrial_proj)
            gap = (primal_val_proj - dual_val)
            if gap < self.targap:
                xtrial = xtrial_proj
                self.flag = 'desired'
                self.rxtrial = rxtrial_proj
                break
            # then check the un-projected primal
            rxtrial = self.func(xtrial)
            primal_val = self.prox_primal(xtrial, uk, alphak, rxtrial)
            gap = (primal_val - dual_val)
            if gap < self.targap:
                self.flag = 'desired'
                self.rxtrial = rxtrial
                break
            if self.inner_its > self.config.ipg_linesearch_limits:
                self.flag = 'maxiter'
                # attemp a correction step
                x_correction = self.correction_step(xtrial, ipg_kwargs['xref'])
                rx_correction = self.func(x_correction)
                primal_val_correction = self.prox_primal(x_correction, uk, alphak, rx_correction)
                gap_corrected = primal_val_correction - dual_val
                if primal_val_correction <= primal_val:
                    xtrial = x_correction
                    gap = gap_corrected
                    rxtrial = rx_correction
                    self.flag = 'correct'
                self.rxtrial = rxtrial
                break
            ################ proceed to the next iteration ###############################
            y_current = ytrial
            grad_psi_ycurrent = (alphak * self.ATA @ y_current + ATuk)
            dual_val_ycurrent = dual_val
            # if no backtracking is performed, increase the stepsize for the next iteration
            if bak == 0:
                self.stepsize *= self.config.ipg_linesearch_beta
        # post-processing
        self.gap = gap
        self.aoptim = l2_norm(xtrial - xk)
        self.xtrial = xtrial
        return xtrial, ytrial
    def _proj_norm_ball(self, y):
        return self._proj_norm_ball_jit(y, self.K, self.starts, self.ends, self.weights)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _proj_norm_ball_jit(y, K, starts, ends, weights):
        projected_group = {}
        for i in range(K):
            start, end = starts[i], ends[i]
            y_Gi = y[start:end]
            norm_y_Gi = np.sqrt(np.dot(y_Gi.T, y_Gi))[0][0]
            if norm_y_Gi > weights[i]:
                y[start:end] = (weights[i] / norm_y_Gi) * y_Gi
                projected_group[i] = i
        return y, projected_group
    
    def _get_group_structure(self, X):
        return self._get_group_structure_jit(X, self.K, self.groups)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _get_group_structure_jit(X, K, groups):
        nz = 0
        for g in groups:
            X_Gi = X[g]
            if (np.sum(np.abs(X_Gi)) == 0):
                nz += 1
        nnz = K - nz
        return nnz, nz

    def correction_step(self, x, xref):
        return self._correction_step_jit(x, xref, self.groups)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _correction_step_jit(x, xref, groups):
        x_corrected = x.copy()
        for g in groups:
            if np.sum(np.abs(xref[g])) == 0:
                x_corrected[g] = 0.0
        return x_corrected

    @staticmethod
    def prox_primal(xk, uk, alphak, rxk):
        return (0.5 / alphak) * l2_norm(xk - uk) ** 2 + rxk

    @staticmethod
    def prox_dual(y, uk, alphak):
        return -(alphak / 2 * l2_norm(y) ** 2 + uk.T @ y)

    def print_header(self, **kwargs):
        header = " Epoch/batch   iters.   Flag    Stepsize   baks    Gap      tarGap"
        header += "\n"
        with open(kwargs['filename'], "a") as logfile:
            logfile.write(header)
    
    def print_iteration(self, **kwargs):
        contents = f" {kwargs['epoch']:4d}/{kwargs['batch']:5d}  {self.inner_its:5d}    {self.flag}   {self.stepsize:.3e}  {self.total_bak:4d} {self.gap:+.3e} {self.targap:+.3e} "
        contents += "\n"
        with open(kwargs['filename'], "a") as logfile:
            logfile.write(contents)


class TreeOG:
    def __init__(self, groups, tree, penalty, weights=None):
        """
        Taken from https://thoth.inrialpes.fr/people/mairal/spams/doc-python/html/doc_spams006.html#sec26
            # Example 1 of tree structure
            # tree structured groups:
            # g1= {0 1 2 3 4 5 6 7 8 9}
            # g2= {2 3 4}
            # g3= {5 6 7 8 9}
            own_variables =  np.array([0,2,5],dtype=np.int32) # pointer to the first variable of each group
            N_own_variables =  np.array([2,3,5],dtype=np.int32) # number of "root" variables in each group
            # (variables that are in a group, but not in its descendants).
            # for instance root(g1)={0,1}, root(g2)={2 3 4}, root(g3)={5 6 7 8 9}
            eta_g = np.array([1,30,1],dtype=np.float64) # weights for each group, they should be non-zero to use fenchel duality
            groups = np.asfortranarray([[0,0,0],
                                        [1,0,0],
                                        [1,0,0]],dtype = np.bool)
            # first group should always be the root of the tree
            # non-zero entriees mean inclusion relation ship, here g2 is a children of g1,
            # g3 is a children of g1
            groups = csc_matrix(groups,dtype=np.bool)
            tree = {'eta_g': eta_g,'groups' : groups,'own_variables' : own_variables,
                    'N_own_variables' : N_own_variables}
        """
        assert len(groups) == len(weights), "groups and weights should be of the same length"
        self.penalty = penalty
        self.tree = tree
        self.K = len(groups)
        if weights is None:
            weights = np.array([np.sqrt(len(g)) for g in groups])
        self.weights = self.penalty * weights
        self.groups = List()
        for g in groups:
            self.groups.append(np.array(g))

    def __str__(self):
        return("Tree Group L1")

    def func(self, x):
        return self._func_jit(x, self.groups, self.weights)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _func_jit(x, groups, weights):
        ans = 0.0
        for i, g in enumerate(groups):
            xg = x[g]
            ans += np.sqrt(np.dot(xg.T, xg))[0][0] * weights[i]
        return ans

    ##############################################
    #    exact proximal gradient calculation   ###
    ##############################################

    def compute_exact_proximal_gradient_update(self, xk, alphak, dk, compute_structure=False):
        """
            implement the fixed stepsize projected  gradient descent
        """
        param = {'numThreads': -1, 'verbose': False, 'pos': False, 'intercept': False, 'lambda1': alphak, 'regul': 'tree-l2'}
        uk = xk - alphak * dk
        xtrial = spams.proximalTree(uk, self.tree, False, **param)
        self.optim = l2_norm(xtrial - xk)
        if compute_structure:
            nnz, nz = self._get_group_structure(xtrial)
            return xtrial, nnz, nz
        else:
            return xtrial, _, _


    def _get_group_structure(self, X):
        return self._get_group_structure_jit(X, self.K, self.groups)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _get_group_structure_jit(X, K, groups):
        nz = 0
        for g in groups:
            X_Gi = X[g]
            if (np.sum(np.abs(X_Gi)) == 0):
                nz += 1
        nnz = K - nz
        return nnz, nz            