'''
# File: InexactProxGD.py
# Project: solvers
# Created Date: 2023-08-29 8:36
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2023-08-31 11:48
# Modified By: Yutong Dai yutongdai95@gmail.com
# 
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
'''


import numpy as np
import datetime
import time
import os
import sys
sys.path.append("../../")
from src.utils import update_nu_rho


class InexactProxGD:
    def __init__(self, f, r, config):
        self.f = f
        self.r = r
        self.version = "0.1 (2023-08-29)"
        self.n = self.f.n
        self.p = self.f.p
        self.config = config
        self.datasetname = self.f.datasetName.split("/")[-1]
        if config.save_log:
            self.filename = '{}.txt'.format(self.config.tag)
        else:
            self.filename = None
        if self.config.print_level > 0:
            self.print_problem()
            self.print_config()

    def solve(self, x_init=None, alpha_init=None):
        self.datasetid = None
        # configure mainsolver
        if x_init is None:
            xk = np.zeros((self.p, 1))
        else:
            xk = x_init
        x_best_so_far = xk
        if not alpha_init:
            self.alphak = 1.0
        else:
            self.alphak = alpha_init
        tol = self.config.accuracy
        # dual variable
        yk = None
        # collect stats
        self.iteration = 0
        self.fevals = 0
        self.gevals = 0
        self.subits = 0
        self.time_so_far = 0
        print_cost = 0
        iteration_start = time.time()
        fxk = self.f.func(xk)
        rxk = self.r.func(xk)
        self.Fxk = fxk + rxk
        Fxk_best_so_far = self.Fxk
        gradfxk = self.f.gradient(xk)
        self.fevals += 1
        self.gevals += 1
        self.status = 404
        self.subsolver_consequtive_maxiter = 0
        self.subsolver_total_correct_iters = 0
        # config subsolver
        stepsize_init = None
        if self.config.rebuttal:
            nuk, rhok = 1.0, 1.0
        while True:
            xaprox, ykplus1 = self.r.compute_inexact_proximal_gradient_update(
                xk, self.alphak, gradfxk, yk, stepsize_init, ipg_kwargs={'iteration':self.iteration, 'xref':x_best_so_far})
            if self.config.rebuttal:
                nuk, rhok, dist_to_sol_set_up = update_nu_rho(xk, self.alphak, gradfxk, ykplus1, self.r, nuk, rhok)
                # print("nuk: {}, rhok: {}, dist_to_sol_set_up: {}".format(nuk, rhok, dist_to_sol_set_up))
            self.aoptim = self.r.aoptim
            # if self.r.flag == 'lscfail':
            #     self.status = -2
            #     break
            self.time_so_far += time.time() - iteration_start - print_cost
            # print current iteration information
            self.subits += self.r.inner_its
            if self.config.print_level > 0:
                if self.iteration % self.config.print_head_every == 0:
                    self.print_header()
                self.print_iteration()
            # the abs is added avoid negetive self.r.gap due numerical cancellation
            self.optim_ub = self.aoptim + \
                np.sqrt(2 * np.abs(self.r.gap) * self.alphak)
            # check termination
            if self.config.optim_scaled:
                tol_scaled = tol * min(1.0, self.alphak)
            else:
                tol_scaled = tol
            if self.optim_ub <= tol_scaled:
                self.status = 0
                break

            if self.iteration > self.config.max_iters:
                self.status = 1
                break
            if self.time_so_far >= self.config.max_time:
                self.status = 2
                break
            if self.r.flag == 'correct':
                self.subsolver_total_correct_iters += 1
                if self.subsolver_total_correct_iters >= 2:
                    self.status = 4
                    break
            if self.r.flag == 'maxiter':
                self.subsolver_consequtive_maxiter += 1
                if self.subsolver_consequtive_maxiter == 2:
                    self.status = 3
                    break
            else:
                # reset counter
                self.subsolver_consequtive_maxiter = 0

            # new iteration
            iteration_start = time.time()
            self.iteration += 1
            # line search
            if self.config.do_linesearch:
                xtrial = xaprox
                fxtrial = self.f.func(xtrial)
                rxtrial = self.r.rxtrial
                self.bak = 0
                self.stepsize = 1
                self.d = xtrial - xk
                self.d_norm = self.aoptim
                pass_L_test = True
                # construct the upper-bound of the directional derivative
                if self.config.ipg_strategy == 'yd':
                    dirder_upper = -((self.d_norm**2 / self.alphak) - np.sqrt
                                     (2.0 * self.r.targap / self.alphak) * self.d_norm - self.r.targap)
                elif self.config.ipg_strategy == 'lee':
                    dirder_upper = - \
                        (rxk - rxtrial - np.sum(gradfxk * self.d))
                else:
                    dirder_upper = -np.inf
                const = self.config.linesearch_eta * dirder_upper
                # begin backtracking
                while True:
                    if self.config.ipg_strategy == 'yd' or self.config.ipg_strategy == 'lee':
                        lhs = fxtrial - fxk + rxtrial - rxk
                        rhs = self.stepsize * const
                        # if self.r.flag == 'maxiter' or  self.iteration == 1697:
                        #     print(f"its:{self.iteration:3d} | LHS:{lhs:.4e} | RHS:{rhs:.4e} | LHS-RHS:{lhs-rhs:.4e} | dirder_upper:{dirder_upper:.4e}")
                        if lhs <= rhs:
                            self.step_take = xtrial - xk
                            xk = xtrial
                            break
                        if self.stepsize <= 1e-20:
                            # linesearch failure
                            # self.status = -1
                            # print("mainsolver: small stepsize encountered in the linesearch for alphak")
                            self.step_take = xtrial - xk
                            xk = xtrial
                            break
                        self.bak += 1
                        self.stepsize *= self.config.linesearch_xi
                        xtrial = xk + self.stepsize * self.d
                        fxtrial = self.f.func(xtrial)
                        rxtrial = self.r.func(xtrial)
                    else:
                        # test Lipschtiz inequality for exact solve and the schimdt inexact solve
                        l_lhs = fxtrial - fxk
                        l_rhs = np.sum(gradfxk * self.d) + 1.0 / (2.0 * self.alphak) * (self.d_norm**2)
                        if (l_lhs <= l_rhs) or (np.abs(l_lhs - l_rhs) <= 1e-15):
                            self.step_take = xtrial - xk
                            xk = xtrial
                        else:
                            # stay at the current point and reduce alphak, or equivalently enlarge L estimate
                            pass_L_test = False
                            self.alphak *= 0.8
                            fxtrial = self.f.func(xk)
                            rxtrial = rxk
                            self.step_take = 0.0
                        if self.aoptim <= 1e-8:
                            self.config.linesearch_beta = 1.0
                        break
                # # terminate the whole algorithm as linesearch failed
                # if self.status == -1:
                #     break
            else:
                self.step_take = xaprox - xk
                xk = xaprox
                fxtrial = self.f.func(xk)
                rxtrial = self.r.rxtrial
                self.bak = 0
            self.step_take_size = np.sqrt(
                np.sum(self.step_take * self.step_take))
            # prepare quantities for the next dual iteration
            yk = ykplus1
            stepsize_init = self.r.stepsize

            # print line search
            print_start = time.time()
            if self.config.print_level > 0:
                self.print_linesearch()
            print_cost = time.time() - print_start

            # update parameter for two in
            if self.config.do_linesearch and pass_L_test:
                if self.config.prox_step_strategy == "frac":
                    if self.bak > 0:
                        self.alphak *= 0.8
                elif self.config.prox_step_strategy == "model":
                    # L_k_hat makes the model decreases match with the actual decreases.
                    # print(self.iteration, self.step_take_size)
                    L_k = 1 / self.alphak
                    L_k_hat = 2 * \
                        (fxtrial - fxk - np.sum(gradfxk * self.step_take)) / \
                        (self.step_take_size**2)
                    L_k = max(2 * L_k, min(1e3 * L_k, L_k_hat))
                    L_kplus1 = max(1e-3, 1e-3 * L_k, L_k_hat)
                    self.alphak = 1.0 / L_kplus1
                elif self.config.prox_step_strategy == "heuristic":
                    if self.bak == 0:
                        # add a safeguard
                        self.alphak = min(self.alphak * self.config.linesearch_beta, 100)
                    else:
                        self.alphak = max(0.8 * self.alphak, 1e-20)
                    if self.aoptim <= 1e-4:
                        self.config.linesearch_beta = 1.0
                elif self.config.prox_step_strategy == "const":
                    pass
                else:
                    raise ValueError(f"Unrecognized stepsize_strategy value: {self.config.prox_step_strategy}")

            # move to new iterate
            fxk = fxtrial
            rxk = rxtrial
            self.Fxk = fxk + rxk
            gradfxk = self.f.gradient(xk)
            self.fevals += 1 + self.bak
            self.gevals += 1
            if self.r.flag != 'maxiter':
                x_best_so_far = xk
                Fxk_best_so_far = self.Fxk
        if self.status != 3:
            self.solution = xk
        else:
            self.solution = x_best_so_far
            self.Fxk = Fxk_best_so_far
        self.print_exit()

        nnz, nz = self.r._get_group_structure(self.solution)
        info = {'iteration': self.iteration, 'time': self.time_so_far,
                'x': xk, 'F': self.Fxk, 'nnz': nnz, 'nz': nz,
                'status': self.status,
                'fevals': self.fevals, 'gevals': self.gevals, 'aoptim': self.aoptim,
                'n': self.n, 'p': self.p, 'Lambda': self.r.penalty,
                'K': self.r.K, 'subits': self.subits, 'datasetid': self.datasetid,
                'optim': self.optim_ub, 'nuk':nuk, 'rhok':rhok
                }
        return info

    def print_problem(self):
        contents = "\n" + "=" * 80
        contents += "\n       Inexact Proximal Gradient Type Method   (version:{})  \n".format(
            self.version)
        time = datetime.datetime.now()
        contents += "=" * 80 + '\n'
        contents += f"Problem Summary: Excuted at {time.year}-{time.month}-{time.day} {time.hour}:{time.minute}\n"

        problem_attribute = self.f.__str__()
        problem_attribute += "Regularizer:{:.>56}\n".format(
            self.r.__str__())
        problem_attribute += "Penalty Parameter:{:.>30}lambda={:3.4f}\n".format(
            '', self.r.penalty)
        problem_attribute += "Number of groups:{:.>32}\n".format(self.r.K)
        contents += problem_attribute
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)

    def print_config(self):
        contents = "\n" + "Algorithm Parameters:\n"
        contents += 'Termination Conditions:'
        contents += f" accuracy: {self.config.accuracy} | optim scaled: {self.config.optim_scaled} | time limits:{self.config.max_time} | iteration limits:{self.config.max_iters}\n"
        if self.config.exact_pg_computation:
            contents += f"Evaluate proximal operator with high accuracy: {self.config.exact_pg_computation_tol}\n"
        else:
            contents += f"Inexact Strategy: {self.config.ipg_strategy}:"
            # if config.ipg_strategy == "schimdt":
            #     contents += f" delta:{self.config.ipg_schimdt_delta:.3e} | c:{self.config.ipg_schimdt_c:.3e}\n"
            # else:
            #     contents += f" gamma:{self.config.ipg_yd_gamma:.3e}\n"
        if self.config.do_linesearch or self.config.ipg_do_linesearch:
            contents += 'Lineserch Parameters:'
            contents += f" eta:{self.config.linesearch_eta} | xi:{self.config.linesearch_xi} | beta:{self.config.linesearch_beta}\n"
        contents += f"Proximal Stepsize update: {self.config.prox_step_strategy}\n"
        contents += "*" * 100 + "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)

    def print_header(self):
        header = " Iters.   Obj.    alphak   |"
        header += "  aoptim   its.   Flag   Stepsize  baks    Gap       tarGap    #pz  #pnz |"
        header += self.print_linesearch_header()
        header += "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(header)
        else:
            print(header)

    def print_iteration(self):
        contents = f" {self.iteration:5d} {self.Fxk:.3e} {self.alphak:.3e} |"
        contents += f" {self.aoptim:.3e} {self.r.inner_its:4d} {self.r.flag}"
        contents += f" {self.r.stepsize:.3e} {self.r.total_bak:4d} {self.r.gap:+.3e} {self.r.targap:+.3e}"
        nnz, nz = self.r._get_group_structure(self.r.xtrial)
        contents += f"  {nz:4d}  {nnz:4d} |"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)

    def print_linesearch_header(self):
        if not self.config.do_linesearch:
            contents = ""
        else:
            contents = " baks    stepsize  |dtaken| |"
        return contents

    def print_linesearch(self):
        if not self.config.do_linesearch:
            contents = "\n"
        else:
            contents = f"  {self.bak:3d}   {self.stepsize:.3e} {self.step_take_size:.3e} |\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)

    def print_exit(self):
        contents = '\n' + "=" * 30 + '\n'
        if self.status == -2:
            contents += 'Exit: Proximal Problem Solver Failed\n'
        if self.status == -1:
            contents += 'Exit: Line Search Failed\n'
        elif self.status == 0:
            contents += 'Exit: Optimal Solution Found\n'
        elif self.status == 1:
            contents += 'Exit: Iteration limit reached\n'
        elif self.status == 2:
            contents += 'Exit: Time limit reached\n'
        elif self.status == 3:
            contents += 'Exit: Early stoppiong. (2 Consequtive subsolver maxiters).\n'
        elif self.status == 4:
            contents += 'Exit: Early stoppiong. (2 correction steps cap reached).\n'
        print(contents)
        contents += "\nFinal Results\n"
        contents += "=" * 30 + '\n'
        contents += f'Iterations:{"":.>65}{self.iteration:d}\n'
        contents += f'CPU seconds:{"":.>64}{self.time_so_far:.4f}\n'
        nnz, nz = self.r._get_group_structure(self.solution)
        contents += f'# zero groups:{"":.>62}{nz:d}\n'
        contents += f'Objective function:{"":.>57}{self.Fxk:8.6e}\n'
        contents += f'Optimality error:{"":.>59}{self.optim_ub:8.6e}\n'
        contents += f'Function evaluations:{"":.>55}{self.fevals:d}\n'
        contents += f'Gradient evaluations:{"":.>55}{self.gevals:d}\n'

        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)