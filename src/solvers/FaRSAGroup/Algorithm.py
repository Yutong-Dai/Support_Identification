import sys
import os
import time
sys.path.append("../")
sys.path.append("../..")
import numpy as np
from numba import jit
from copy import deepcopy
import src.solvers.FaRSAGroup.utils as utils


class AlgoBase:
    def __init__(self, f, r, proxStepsize, params, kappa_1, kappa_2, debug=False):
        """
        f: loss function
        r: regularizer
        proxStepsize: proximal stepsize
        params: tons of parameters related to linesearch, termination tolerance etc
        """
        self.f = f
        self.r = r
        # number of groups
        self.K = self.r.K
        # n: number of data points; p: point dimension
        self.n, self.p = self.f.n, self.f.p
        # array; start index of each group
        self.starts = self.r.starts
        # array; end index of each group
        self.ends = self.r.ends
        self.Lambda_group = self.r.weights
        self.proxStepsize = proxStepsize
        self.params = params
        self.pg_flag = False
        self.fval = None
        self.F = None
        self.F_old = None
        self.f_old = None
        self.dirder = None
        self.newProxStepsize = np.copy(self.proxStepsize)
        self.gradf = None
        self.gradF = None
        self.debug = debug
        self.cg_big_scale = 1e3
        self.kappa_1 = kappa_1
        self.kappa_2 = kappa_2
        self.kappa_3 = 1
        self.sin_theta = np.sin(np.pi / 4)
        self.group_size = self.r.group_size
        self.rangeK = np.arange(self.K)
        self.num_cg0_steps = 0
        self.num_cgdesc_steps = 0
        self.num_pg_steps = 0
        self.newton_time_iter = 0
        self.ls_time_iter = 0

    def proximal_step(self, X):
        """
        calculate proximal stepsize depending on the type of proxStepsize being passed.
        currently accelerates the for loop use numba njit mode.
        If number of groups is large, then the gain is obvious.
        """
        self.gradf = self.f.gradient(X)
        (self.proximal, self.gradF,
         self.geGradFGroup, self.group_X_norm,
         self.group_gradF_norm,
         self.zeroGroup, self.nonZeroGroup,
         self.zeroProxGroup, self.nonZeroProxGroup,
         self.zeroBlock, self.nonZeroBlock,
         self.zeroProxBlock, self.nonZeroProxBlock,
         self.geGradFBlock, self.group_proximal_norm) = _proximal_uniform_alpha_jit(X, self.proxStepsize, self.gradf, self.K, self.p,
                                                                                    self.starts, self.ends, self.r.weights, self.kappa_1)

    def set_cg(self):
        """
        calculate chi_cg
        find index of variables to be used in compuation of cg-newton direction
        """
        self.proximal_squared = self.proximal ** 2
        self.bar_I_cg_index = self.geGradFBlock & self.nonZeroProxBlock
        self.bar_I_cg_group = utils.intersection(
            self.geGradFGroup, self.nonZeroProxGroup)
        bar_I_cg_group_size = self.group_size[self.bar_I_cg_group]
        # self.bar_chi_cg = utils.l2_norm(self.proximal[self.bar_I_cg_index].reshape(-1, 1))
        self.bar_chi_cg = np.sqrt(
            np.sum(self.proximal_squared[self.bar_I_cg_index]))
        self.I_cg_index = np.copy(self.bar_I_cg_index)
        self.I_cg_group = deepcopy(self.bar_I_cg_group)
        self.chi_cg = self.bar_chi_cg
        self.inner_radius_lst = None
        self.outter_radius_lst = None
        if len(self.bar_I_cg_group) != 0:
            bar_I_cg_group_norm = self.group_X_norm[self.bar_I_cg_group]
            gradF_bar_I_cg = self.gradF[self.bar_I_cg_index].reshape(-1, 1)
            # self.norm_gradF_bar_I_cg = utils.l2_norm(gradF_bar_I_cg)
            self.norm_gradF_bar_I_cg = np.sqrt(
                np.dot(gradF_bar_I_cg.T, gradF_bar_I_cg))[0][0]
            self.bar_I_cg_size = np.sum(self.bar_I_cg_index)
            vec = (bar_I_cg_group_size / self.bar_I_cg_size) * \
                self.norm_gradF_bar_I_cg
            small_radius = np.minimum(
                self.kappa_2 * vec, self.kappa_3 * np.ones_like(vec))
            small_idx = bar_I_cg_group_norm < small_radius
            rm_group_idx = np.array(self.bar_I_cg_group)[small_idx]
            # for print_more purpose
            self.small_radius_lst = -1 * np.ones(self.K)
            self.small_radius_lst[self.bar_I_cg_group] = small_radius
            if len(rm_group_idx) > 0:
                for i in rm_group_idx:
                    start, end = self.starts[i], self.ends[i]
                    self.I_cg_index[start:end] = False
                    self.I_cg_group.remove(i)
                # update the actual chicg
                # self.chi_cg = utils.l2_norm(self.proximal[self.I_cg_index].reshape(-1, 1))
                self.chi_cg = np.sqrt(
                    np.sum(self.proximal_squared[self.I_cg_index]))

    def select_cg_frac(self, fraction):
        if fraction == 1:
            self.I_cgs = self.I_cg_index.reshape(-1,)
            self.I_cgs_group = self.I_cg_group
        else:
            removed = np.setdiff1d(self.rangeK, self.I_cg_group)
            self.group_proximal_norm[removed] = 0
            idx_large_to_small = np.argsort(self.group_proximal_norm)[::-1]
            sorted_norm = self.group_proximal_norm[idx_large_to_small]
            end_pos = np.argmax(np.cumsum(sorted_norm**2) >
                                fraction * (self.chi_cg)**2)
            self.I_cgs_group = idx_large_to_small[:end_pos + 1]
            self.I_cgs = np.copy(self.I_cg_index)
            drop_group = np.setdiff1d(self.I_cg_group, self.I_cgs_group)
            # for i in drop_group:
            #     start, end = self.starts[i], self.ends[i]
            #     self.I_cgs[start:end] = False
            self.I_cgs = remove_groups_jit(
                drop_group, self.starts, self.ends, self.I_cgs)
            self.I_cgs = self.I_cgs.reshape(-1,)
            self.I_cgs_group = utils.intersection(
                self.I_cg_group, self.I_cgs_group)
        # cg_space dimension
        self.nI_cgs = np.sum(self.I_cgs)
        self.I_cg_group.sort()
        self.subgroup_index_cg = np.array(
            self.I_cgs_group)  # for hessian vector product

    def select_cg(self):
        # has to reshape, because I use it to index columns
        self.I_cgs = self.I_cg_index.reshape(-1,)
        self.nI_cgs = np.sum(self.I_cgs)
        self.I_cgs_group = self.I_cg_group
        self.I_cgs_group.sort()  # must sort
        self.subgroup_index_cg = np.array(
            self.I_cgs_group)  # for hessian vector product

    def set_pg(self):
        """
        calculate chi_pg
        find index of variables to be used in compuation of line search method
        """
        self.I_pg_index = ~self.I_cg_index
        # self.chi_pg = utils.l2_norm(self.proximal[self.I_pg_index].reshape(-1, 1))
        self.chi_pg = np.sqrt(np.sum(self.proximal_squared[self.I_pg_index]))

    def select_pg(self):
        # to do: allow to select a subset of I_pg
        # no need to reshape, this is just used to indexing on the search direction, which is a vector
        self.I_pgs = self.I_pg_index
        self.nI_pgs = np.sum(self.I_pgs)
        # setdiff1d will do the sorting for you
        self.subgroup_index_pg = np.setdiff1d(self.rangeK, self.I_cg_group)

    def newton_cg(self, X):
        """
        Perform Hessian Vector Product.
        self.d_cg: newton direction in the full space
        self.newton_counter: number of hessian vector product performed
        """
        self.newton_counter = 0
        termination = False
        # initialize the residual to the gradident of f+r at the selected reduced space
        r0 = self.gradF_Icgs
        # normr0 = utils.l2_norm(r0)
        normr0 = np.sqrt(np.dot(r0.T, r0))[0][0]
        residual = r0
        normr = normr0
        p = -r0
        d = np.zeros((self.nI_cgs, 1))
        # do not confuse yourself; p is different from self.p, which is problem dimension
        self.d_cg = np.zeros((self.p, 1))
        max_loop = min(self.params['maxCG_iter'], self.nI_cgs)
        # self.f._prepare_hv_approx_data(self.I_cgs)
        self.f._prepare_hv_data(self.I_cgs)
        self.r._prepare_hv_data(X, self.subgroup_index_cg)
        # self.gradF_Icgs_norm = utils.l2_norm(gradF_Icgs)
        self.gradF_Icgs_norm = np.sqrt(
            np.dot(self.gradF_Icgs.T, self.gradF_Icgs))[0][0]
        # if True:
        #     fH = self.f.hessian()
        #     rH = self.r.hessian(X)
        #     fHsub = fH[self.I_cgs, :][:, self.I_cgs]
        #     rHsub = rH[self.I_cgs, :][:, self.I_cgs]
        # result = {'fH': fHsub, 'rH': rHsub}
        # np.save('../hessian.npy', result)
        self.res_target = max(
            min(self.params['eta_r'], normr0**0.5) * normr0, 1.0e-10)
        while True:
            # Compute next linear CG iterate.
            Hp1 = self.f.hessian_vector_product_fast(p)
            Hp2 = self.r.hessian_vector_product_fast(p, self.subgroup_index_cg)
            Hp = Hp1 + Hp2
            self.newton_counter += 1
            pTHp = np.matmul(p.T, Hp)
            alphaCG = normr**2 / pTHp
            d = d + alphaCG * p
            # self.normd = utils.l2_norm(d)
            self.normd = np.sqrt(np.dot(d.T, d))[0][0]
            residual += alphaCG * Hp
            normr_old = normr
            # normr = utils.l2_norm(residual)
            normr = np.sqrt(np.dot(residual.T, residual))[0][0]
            # Check for termination of CG
            if normr <= self.res_target:
                # if (self.nI_cgs >= 2 and self.newton_counter >= 1) or (self.nI_cgs == 1):
                self.subprobFlag = "CGtol"
                termination = True
            elif self.normd > self.cg_big_scale * min(self.gradF_Icgs_norm, 1):
                # if (self.nI_cgs >= 2 and self.newton_counter >= 1) or (self.nI_cgs == 1):
                self.subprobFlag = 'CGbig'
                termination = True
            elif self.newton_counter >= max_loop:
                self.subprobFlag = "CGmax"
                termination = True
            if termination:
                self.res = normr
                self.d_cg[self.I_cgs] = d
                # res1 = np.max(np.abs(self.f.hessian_vector_product_fast(d) + self.r.hessian_vector_product_fast(d, self.subgroup_index_cg) + self.gradF_Icgs))
                # res2 = np.max(np.abs((fHsub + rHsub)@d + self.gradF_Icgs))
                # print(f"Hd+g: res1:{res1:3.4e} | res2:{res2:3.4e}")
                break
            # Get next conjugate direction
            betaCG = np.power(normr, 2) / np.power(normr_old, 2)
            p = -residual + betaCG * p

    def get_taus(self, X):
        taus_vector = np.ones(X.shape) * np.inf
        self.inner_radius_lst = [0] * self.K
        self.outter_radius_lst = [0] * self.K
        self.tau_k_intersect = [1]
        for i in self.I_cgs_group:
            start, end = self.starts[i], self.ends[i]
            d_Gi = self.d_cg[start:end]
            X_Gi = X[start:end]
            a = np.dot(d_Gi.T, d_Gi)
            b = 2 * np.dot(d_Gi.T, X_Gi)
            temp = min(self.kappa_3, self.kappa_2 *
                       self.gradF_Icgs_norm * (end - start) / self.nI_cgs)
            rho_i = max(self.kappa_1 * self.group_gradF_norm[i], temp)
            radius_i = min(rho_i, self.sin_theta * self.group_X_norm[i])
            self.inner_radius_lst[i] = radius_i
            self.outter_radius_lst[i] = rho_i
            c = np.dot(X_Gi.T, X_Gi) - radius_i ** 2
            tau_k_i = utils.get_first_intersetion(a, b, c)
            if tau_k_i > 0:
                taus_vector[start:end] = tau_k_i
                if tau_k_i < 1:
                    self.tau_k_intersect.append(tau_k_i[0][0])
        return taus_vector

    def line_search_cg(self, X, strategy='cutfrac'):
        """
        backtrack linesearch along the newton direction in the full space
        strategy:
            cutfrac: every time do backtracking, cut the stepsize by a constant fraction.
            jump: only try 1, tau_1, tau_2, tau_min in the projection stage to avoid wasted backtracking
                  specifically, 1 > tau_1 > tau_2 > tau_min
        """
        self.status = 0
        self.cg_backtrack = 0
        self.cg_stepsize = 1
        self.cg_feval = 0
        self.cg_flag = False
        self.f_old = self.f.evaluate_function_value(X)
        self.r_old = self.r.func(X)
        self.cg_feval += 1
        taus_vector = self.get_taus(X)
        self.projection_attempts = 0
        self.newZB = 0
        tau_min = min(taus_vector)
        if strategy == 'jump':
            self.tau_k_intersect.sort(reverse=True)
            self.tau_k_intersect = np.array(self.tau_k_intersect)
        while self.cg_stepsize >= tau_min:
            proj_vec = (self.cg_stepsize > taus_vector)
            y = X + self.cg_stepsize * self.d_cg * \
                (self.cg_stepsize <= taus_vector) + (-X) * proj_vec
            self.fval = self.f.evaluate_function_value(y)
            self.rval = self.r.func(y)
            self.F = self.fval + self.rval
            self.cg_feval += 1
            self.projection_attempts += 1
            # avoid numerical cancellation
            if (self.fval - self.f_old + self.rval - self.r_old) <= 0:
                self.cg_type = 'proj'
                self.num_cg0_steps += 1
                self.d_use = y - X
                self.stepSize_use = 1
                self.dirder = np.matmul(self.gradF.T, self.d_use)[0][0]
                for i in self.I_cgs_group:
                    start, end = self.starts[i], self.ends[i]
                    temp = proj_vec[start:end]
                    if np.sum(temp) != 0:
                        self.newZB += 1
                return y
            if self.projection_attempts >= len(self.I_cgs_group):
                break
            if strategy == 'cutfrac':
                self.cg_stepsize *= self.params['xi']
            else:
                # turn off the projection for the group according to the decreasing
                # order of tau_k_i
                self.tau_k_intersect = self.tau_k_intersect[self.tau_k_intersect <
                                                            self.cg_stepsize]
                self.cg_stepsize = min(
                    self.tau_k_intersect[0] - 1e-9, self.cg_stepsize * self.params['xi'])
        self.dirder = np.matmul(self.gradF.T, self.d_cg)[0][0]
        ratio = np.abs(self.dirder) / (1 + np.abs(self.f_old + self.r_old))
        if ratio < 1e-15:
            # no further progress can be made
            self.cg_type = 'npgs'
            self.status = 4
            return X
        while True:
            y = X + self.cg_stepsize * self.d_cg
            self.fval = self.f.evaluate_function_value(y)
            self.rval = self.r.func(y)
            self.F = self.fval + self.rval
            self.cg_feval += 1
            LHS = self.fval - self.f_old + self.rval - self.r_old
            # print(f"F-F_old:{(self.fval+ self.rval - self.f_old  - self.r_old):3.4e},  LHS:{LHS:3.4e}, RHS:{self.params['eta'] * self.cg_stepsize * self.dirder:3.4e}, dirder:{self.dirder:3.4e}")
            if (LHS <= self.params['eta'] * self.cg_stepsize * self.dirder):
                self.cg_type = 'desc'
                self.num_cgdesc_steps += 1
                self.d_use = self.d_cg
                self.stepSize_use = self.cg_stepsize
                return y
            if self.cg_backtrack >= self.params['maxback']:
                self.status = -1
                return y
            self.cg_backtrack += 1
            self.cg_stepsize *= self.params['xi']

    def line_search_pg(self, X, d):
        """
        backtrack linesearch along the proximal gradient direction either in the full space, but
        only with nonzero components in the search direction, either defined by the cg_space or pg_space
        """
        self.status = 0
        self.pg_backtrack = 0
        self.pg_stepsize = 1
        self.pg_feval = 0
        self.pg_flag = False
        y = X + d
        self.f_old = self.f.evaluate_function_value(X)
        # self.F_old = self.f_old + self.r.evaluate_function_value_jit(X)
        self.F_old = self.f_old + self.r.func(X)
        self.fval = self.f.evaluate_function_value(y)
        # self.F = self.fval + self.r.evaluate_function_value_jit(y)
        self.F = self.fval + self.r.func(y)
        self.pg_feval += 2
        const = (-np.matmul(d.T, d) / self.proxStepsize)[0][0]
        while True:
            if (self.F - self.F_old <= self.params['eta'] * self.pg_stepsize * const):
                if self.pg_backtrack == 0:
                    self.pg_flag = True
                self.d_use = d
                self.stepSize_use = self.pg_stepsize
                self.num_pg_steps += 1
                break
            if self.pg_backtrack >= self.params['maxback']:
                self.status = -1
                print("LHS:{:8.6e} | RHS:{:8.6e}".format(
                    self.F - self.F_old, self.params['eta'] * self.pg_stepsize * const))
                break
            self.pg_backtrack += 1
            self.pg_stepsize *= self.params['xi']
            y = X + self.pg_stepsize * d
            self.fval = self.f.evaluate_function_value(y)
            # self.F = self.fval + self.r.evaluate_function_value_jit(y)
            self.F = self.fval + self.r.func(y)
            self.pg_feval += 1
        return y

    def cg_step(self, X, strategy='cutfrac'):
        """
        perform one cg_step
        """
        self.typeofIteration = 'cg'
        self.subits = 0
        self.gradF_Icgs = self.gradF[self.I_cgs]
        self.newton_time_iter = time.time()
        self.newton_cg(X)
        self.newton_time_iter = time.time() - self.newton_time_iter
        self.subits += self.newton_counter  # number of newton-cg steps
        self.ls_time_iter = time.time()
        y = self.line_search_cg(X, strategy)
        self.ls_time_iter = time.time() - self.ls_time_iter
        return y

    def pg_step(self, X):
        """
        perform one pg_step
        """
        self.typeofIteration = 'pg'
        self.subits = 0  # this is meaningless just use as a place holder.
        d_original = np.zeros(X.shape)
        d_original[self.I_pgs] = self.proximal[self.I_pgs]
        # normd = utils.l2_norm(d_original)
        normd = np.sqrt(np.dot(d_original.T, d_original))[0][0]
        scaleFactor = 1.0e8
        if normd > scaleFactor:
            d = scaleFactor * d_original / normd
            self.normd = scaleFactor
        else:
            d = d_original
            self.normd = normd
        self.ls_time_iter = time.time()
        y = self.line_search_pg(X, d)
        self.ls_time_iter = time.time() - self.ls_time_iter
        return y

    # def get_proxStepsize(self, X, y, f_old, f_new, search_d, stepsize):
    #     """
    #     update proxStep size according to the paper Curtis-Robinson2019_Article_ExploitingNegativeCurvatureInD
    #     using equation (19) and the one follows.
    #     """
    #     actual_decrease = f_new - f_old
    #     step_take = stepsize * search_d
    #     step_take_norm_square = np.matmul(step_take.T, step_take)[0][0]
    #     L_k = 1 / self.proxStepsize
    #     dirder = np.matmul(self.gradf.T, step_take)[0][0]
    #     model_decrease = dirder + 0.5 * step_take_norm_square * L_k
    #     gap = 2 * (actual_decrease - model_decrease) / step_take_norm_square
    #     L_k_hat = L_k + gap
    #     L_kplus1 = 2 * L_k_hat
    #     self.newProxStepsize = min(1 / np.float64(L_kplus1), np.float64(1))

    def get_proxStepsize(self, X, y, f_old, f_new, search_d, stepsize):
        """
        update proxStep size according to the paper Curtis-Robinson2019_Article_ExploitingNegativeCurvatureInD
        using equation (19) and the one follows.
        """
        actual_decrease = f_new - f_old
        step_take = stepsize * search_d
        step_take_norm_square = np.matmul(step_take.T, step_take)[0][0]
        L_k = 1 / self.proxStepsize
        dirder = np.matmul(self.gradf.T, step_take)[0][0]
        model_decrease = dirder + 0.5 * step_take_norm_square * L_k
        diff = actual_decrease - model_decrease
        gap = 2 * diff / step_take_norm_square
        # L_k_hat makes the model decreases match with the actual decreases.
        L_k_hat = L_k + gap
        if diff > 0:
            L_k = max(2 * L_k, min(1e3 * L_k, L_k_hat))
        L_kplus1 = max(1e-3, 1e-3 * L_k, L_k_hat)
        self.newProxStepsize = np.float64(min(1 / L_kplus1, 1))


def set_prox_stepsize(f, r, alpha_type='onealpha', method='gradient'):
    """
    Initialize the proximal gradient stepsize.

    Args:
        f: loss function object
        r: regurlizer object
        strategy:
            'onealpha': estimate one stepsize for all groups
            'groupalpha': estimate stepsize for each group separately
        method:
            'gradient': use the difference of gradients at two selected points to
                        have a lower bound estimation of the Lipschitiz constant
    Returns:
        proxStepsize(np.float64/np.array)
    """
    p = f.p
    K = r.K
    starts = r.starts
    ends = r.ends
    x = np.zeros((p, 1))
    if method == 'gradient':
        s = 1e-2
        fx = f.evaluate_function_value(x)
        gradfx = f.gradient(x)
        y = x - s * gradfx
        while True:
            if utils.l2_norm(y - x) > 1e-8:
                fy = f.evaluate_function_value(y)
                gradfy = f.gradient(y)
                if alpha_type == 'groupalpha':
                    proxStepsize = np.zeros(K)
                    for i in range(K):
                        start, end = starts[i], ends[i]
                        grad_diff = gradfx[start:end] - gradfy[start:end]
                        point_diff = x[start:end] - y[start:end]
                        point_diff_norm = utils.l2_norm(point_diff)
                        grad_diff_norm = utils.l2_norm(grad_diff)
                        if (grad_diff_norm == 0):
                            grad_diff_norm += 1e-8
                        if (point_diff_norm == 0):
                            point_diff_norm += 1e-8
                        proxStepsize[i] = point_diff_norm / grad_diff_norm
                else:
                    grad_diff = gradfx - gradfy
                    point_diff = x - y
                    proxStepsize = utils.l2_norm(
                        x - y) / (1 * utils.l2_norm(gradfx - gradfy))
                break
            else:
                s *= 10
                y = x - s * gradfx
    elif method == 'one':
        if alpha_type == 'groupalpha':
            proxStepsize = np.ones(K)
        else:
            proxStepsize = np.float64(1)
    return proxStepsize


@jit(nopython=True, cache=True)
def _proximal_uniform_alpha_jit(X, alpha, gradf, K, p, starts, ends, Lambda_group, kappa_1):
    """
    calculate the proximal step based on only one alpha for all groups.
    """
    gradr = np.zeros((p, 1))
    proximal = np.zeros((p, 1))
    nonZeroProxBlock = np.full(X.shape, False)
    nonZeroBlock = np.full(X.shape, False)
    geGradFBlock = np.full(X.shape, False)
    zeroGroup = []
    nonZeroGroup = []
    zeroProxGroup = []
    nonZeroProxGroup = []
    geGradFGroup = []
    group_X_norm = -1 * np.ones(K)
    group_gradF_norm = -1 * np.ones(K)
    group_size = -1 * np.ones(K)
    group_proximal_norm = -1 * np.ones(K)
    for i in range(K):
        start, end = starts[i], ends[i]
        group_size[i] = end - start
        XG_i = X[start:end]
        gradfG_i = gradf[start:end]
        gradient_step = XG_i - alpha * gradfG_i
        gradient_step_norm = np.sqrt(
            np.dot(gradient_step.T, gradient_step))[0][0]
        if gradient_step_norm != 0:
            temp = 1 - ((Lambda_group[i] * alpha) / gradient_step_norm)
        else:
            temp = -1
        sG_i = max(temp, 0) * gradient_step - XG_i
        proximal[start:end] = sG_i
        group_proximal_norm[i] = np.sqrt(np.dot(sG_i.T, sG_i))[0][0]
        if temp > 0:
            # predicted to be nonzero
            nonZeroProxBlock[start:end] = True
            nonZeroProxGroup.append(i)
        else:
            # predicted to be zero
            zeroProxGroup.append(i)
        XG_i_norm = np.sqrt(np.dot(XG_i.T, XG_i))[0][0]
        group_X_norm[i] = XG_i_norm
        if (XG_i_norm > 0):
            gradrG_i = Lambda_group[i] * XG_i / XG_i_norm
            gradr[start:end] = gradrG_i
            nonZeroBlock[start:end] = True
            nonZeroGroup.append(i)
            gradFG_i = gradfG_i + gradrG_i
            gradFG_i_norm = np.sqrt(np.dot(gradFG_i.T, gradFG_i))[0][0]
            group_gradF_norm[i] = gradFG_i_norm
            if (XG_i_norm > kappa_1 * gradFG_i_norm):
                geGradFBlock[start:end] = True
                geGradFGroup.append(i)
        else:
            zeroGroup.append(i)
    zeroBlock = ~nonZeroBlock
    zeroProxBlock = ~nonZeroProxBlock
    gradF = gradf + gradr
    return (proximal, gradF, geGradFGroup, group_X_norm, group_gradF_norm,
            zeroGroup, nonZeroGroup, zeroProxGroup, nonZeroProxGroup,
            zeroBlock, nonZeroBlock, zeroProxBlock, nonZeroProxBlock,
            geGradFBlock, group_proximal_norm)


@jit(nopython=True, cache=True)
def remove_groups_jit(drop_group, starts, ends, I_cgs):
    for i in drop_group:
        start, end = starts[i], ends[i]
        I_cgs[start:end] = False
    return I_cgs
