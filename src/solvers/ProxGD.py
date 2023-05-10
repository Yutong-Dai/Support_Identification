import numpy as np
import datetime
import time
import os
import copy
import sys
sys.path.append("../../")
from src.funcs.regularizer import GL1, NatOG, TreeOG
import src.utils as utils
from scipy.sparse import csr_matrix

class ProxGD:
    def __init__(self, f, r, config):
        self.version = "0.1 (2023-05-09)"
        self.f = f
        self.r = r
        self.n = self.f.n
        self.p = self.f.p
        self.K = self.r.K
        self.config = config
        self.datasetname = self.f.datasetName.split("/")[-1]
        self.solver = f"{config.solver}:{config.proxgd_method}"
        if config.proxgd_method == 'FISTA':
            raise ValueError("Not implemented yet.")
        self.stepsize_strategy = config.proxgd_stepsize
        if config.save_log:
            self.filename = '{}.txt'.format(self.config.tag)
        else:
            self.filename = None
        # solver mode
        if isinstance(r, NatOG):
            raise ValueError("Not implemented yet.")
            self.solve_mode = 'inexact'
            self.yk = None
            self.stepsize_init = None
            self.ipg_log_filename = '{}_ipg.txt'.format(self.config.tag)
        elif isinstance(r, GL1) or isinstance(r, TreeOG):
            self.solve_mode = 'exact'
        else:
            raise ValueError("Unknown regularizer type.")
        if self.config.print_level > 0:
            self.print_problem()
            self.print_config()  
        
        # reproted stats
        self.iteration = 0
        self.nnz, self.nz, self.nnz_best, self.nz_best = -1, -1, -1, -1
        self.optim = np.inf    
        self.time_so_far = 0     
        self.header_printed = False
        self.status = None
    
    def solve(self, x_init=None, alpha_init=None):
        if x_init is None:
            xk = np.zeros([self.p, 1])
        else:
            xk = x_init
        if alpha_init is None:
            self.alphak = self.set_init_alpha(xk)
        else:
            self.alphak = alpha_init

        F_seq = []
        nz_seq = []
        x_seq = []
        time_seq = []
        self.fevals = 0
        self.gevals = 0
        self.baks = 0
        self.fevals = 0
        self.gevals = 0
        start = time.time()

        while True:
            self.time_so_far = time.time() - start
            # gradfxk, xprox are all computed in check_termination
            signal = self.check_termination(xk)

            if self.config.save_seq:
                F_seq.append(self.Fxk)
                nz_seq.append(self.nz)
                time_seq.append(self.time_so_far)
            if self.config.save_xseq:
                x_seq.append(csr_matrix(xk))
            if signal == "terminate":
                if self.header_printed == False:
                    self.print_header()
                self.print_iterate()
                break

            # print current iterate           
            if self.config.print_level > 0:
                if self.iteration % self.config.print_head_every == 0:
                    self.print_header()
                self.print_iterate()
            # new iteration
            self.iteration += 1
            if self.stepsize_strategy == 'linesearch':
                self.linesearch_stepsize = 1.0
                self.bak = 0
                self.linesearch_status = 'N.A'
                # search direction
                d = self.xprox - xk
                self.d_norm = utils.l2_norm(d)
                self.dirder_up = -(np.dot(d.T, d)/self.alphak)[0][0]
                # print(self.iteration, (d.T@self.gradfxk)[0][0], self.dirder_up)
                xkp1 = self.xprox
                while True:
                    fxkp1 = self.f.func(xkp1)
                    Fxkp1 = fxkp1 + self.r.func(xkp1)
                    self.fevals += 1
                    LHS = Fxkp1 - self.Fxk
                    RHS = self.config.linesearch_eta * self.linesearch_stepsize * self.dirder_up
                    # print(f'{self.iteration}/{self.bak:3d}: self.fxk: {self.fxk:.3e} | fxkp1:{fxkp1:.3e} | {fxkp1 < self.fxk} | LHS:{LHS:.3e} | RHS:{RHS:.3e} | distance:{utils.l2_norm(xkp1 - xk):.3e}')
                    if LHS <= RHS:
                        self.linesearch_status = 'desired'
                        break
                    else:
                        if np.abs(LHS-RHS) < 1e-15:
                            self.linesearch_status = 'numtol'
                            break
                        elif self.bak == self.config.linesearch_limits:
                            self.linesearch_status = 'maxiter'
                            print(f"LHS:{LHS:8.6e} | RHS:{RHS:8.6e}")
                            self.status = -1
                            break
                    # start backtrack
                    self.bak += 1
                    self.linesearch_stepsize *= self.config.linesearch_xi
                    xkp1 = xk + self.linesearch_stepsize * d
                    
                
                self.baks += self.bak
                xk = xkp1
                if self.bak > 0:
                    self.alphak *= self.config.linesearch_xi
                # if self.bak == 0:
                #     self.alphak *= self.config.linesearch_beta
            elif self.stepsize_strategy == 'const':
                xk = self.xprox
            else:
                raise ValueError("Unknown stepsize strategy.")
        
            if self.stepsize_strategy == 'linesearch':
                self.print_linesearch()

        self.print_exit()

        info = {'iteration': self.iteration, 'time': self.time_so_far, 'optim': self.optim,
                'Fend': self.Fxk, 'xend': csr_matrix(xk),
                'nnz': self.nnz, 'nz': self.nz,
                'status': self.status, 'fevals':self.fevals, 'gevals':self.gevals, 'baks':self.baks,
                'n': self.n, 'p': self.p, 'Lambda': self.r.penalty, 'K': self.r.K}
        if self.config.save_seq:
            info['F_seq'] = F_seq
            info['nz_seq'] = nz_seq
            info['time_seq'] = time_seq
        if self.config.save_xseq:
            info['x_seq'] = x_seq

        return info

    def set_init_alpha(self, x):
        s = 1e-2
        _ = self.f.func(x)
        gradfx = self.f.gradient(x)
        y = x - s * gradfx
        while True:
            if utils.l2_norm(y - x) > 1e-8:
                _ = self.f.func(y)
                gradfy = self.f.gradient(y)
                alpha = utils.l2_norm(x - y) / (1 * utils.l2_norm(gradfx - gradfy))
                break
            else:
                s *= 10
                y = x - s * gradfx
        return alpha

    def check_termination(self, xk):
        self.fxk = self.f.func(xk)
        self.gradfxk = self.f.gradient(xk)
        self.Fxk = self.fxk + self.r.func(xk)
        self.nnz, self.nz = self.r._get_group_structure(xk)
        self.fevals += 1
        self.gevals += 1
        self.xprox, self.pnz, self.pz = self.r.compute_exact_proximal_gradient_update(xk, self.alphak, self.gradfxk, compute_structure=True)
        self.optim = utils.l2_norm(self.xprox - xk)
        if self.iteration == 0:
            self.optim_init = self.optim
        if self.config.optim_scaled:
            self.optim = self.optim / max(1e-15, self.alphak)
        if self.optim <= self.config.accuracy * max(1, self.optim_init):
            self.status = 0
            return 'terminate'
        elif self.iteration >= self.config.max_iters:
            self.status = 1
            return 'terminate'
        elif self.time_so_far >= self.config.max_time:
            self.status = 2
            return 'terminate'
        elif self.status == -1:
            return 'terminate'
        else:
            return 'continue'

    def print_problem(self):
        contents = "\n" + "=" * 80
        contents += f"\n       Solver:{self.solver}    Version: {self.version}  \n"
        contents += "=" * 80 + '\n'
        now = datetime.datetime.now()
        contents += f"Problem Summary: Excuted at {now.date()} {now.time()}\n"

        problem_attribute = self.f.__str__()
        problem_attribute += "Regularizer:{:.>56}\n".format(
            self.r.__str__())
        problem_attribute += "Penalty Parameter:{:.>30}lambda={:3.4f}\n".format(
            '', self.r.penalty)
        problem_attribute += "Number of groups:{:.>32}\n".format(self.r.K)
        contents += problem_attribute
        if self.filename is not None:
            with open(self.filename, "w") as logfile:
                logfile.write(contents)
        else:
            print(contents)
    def print_config(self):
        contents = "\n" + "Algorithm Parameters:\n"
        contents += ' Termination Conditions:\n'
        contents += f"    accuracy: {self.config.accuracy} | optim scaled: {self.config.optim_scaled} | time limits:{self.config.max_time} | epoch limits:{self.config.max_iters}\n"
        contents += f" Proximal Stepsize update: {self.stepsize_strategy}\n"
        if self.solve_mode == 'inexact':
            raise ValueError("Not implemented yet.")
        contents += "*" * 100 + "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)

    def print_exit(self):
        contents = '\n' + "=" * 30 + '\n'
        if self.status == 0:
            contents += 'Exit: Optimal Solution Found\n'
        elif self.status == 1:
            contents += 'Exit: Iteration limit reached\n'
        elif self.status == 2:
            contents += 'Exit: Time limit reached\n'
        elif self.status == -1:
            contents += 'Exit: Line search failed\n'
        else:
            contents += f"Exits: {self.status} Something goes wrong"
        print(contents)
        contents += "\nFinal Results\n"
        contents += "=" * 30 + '\n'
        contents += f'CPU seconds:{self.time_so_far:.4f}\n'
        contents += f'# zero groups(end):{self.nz:d}\n'
        contents += f'Obj. F(end):{self.Fxk:8.6e}\n'
        contents += f'Optim.(end):{self.optim:8.6e}\n'

        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)
    def print_header(self):
        self.header_printed = True
        header = " Iters.   Obj.    alphak      #z   #nz  |   optim     #pz     #pnz |"
        if self.stepsize_strategy == 'linesearch':
            header += " status   #bak    |d|      stepsize |"
        header += "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(header)
        else:
            print(header)

    def print_iterate(self):
        contents = f" {self.iteration:5d} {self.Fxk:.3e} {self.alphak:.3e} {self.nz:5d} {self.nnz:5d}  | {self.optim:8.3e} {self.pz:5d}    {self.pnz:5d} |"
        if self.stepsize_strategy != 'linesearch':
            contents += "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)

    def print_linesearch(self):
        contents = f" {self.linesearch_status} {self.bak:5d} {self.d_norm:.3e}  {self.linesearch_stepsize:.3e} |\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)