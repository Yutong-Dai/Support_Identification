import numpy as np
import datetime
import time
import os
from copy import deepcopy
from scipy.sparse import csr_matrix
import sys
sys.path.append("../../")
import src.utils as utils
from src.solvers.BaseSolver import StoBaseSolver


class RDA(StoBaseSolver):
    def __init__(self, f, r, config):
        self.stepsize_strategy = config.rda_stepsize
        self.solver = "RDA"
        super().__init__(f, r, config)

    def solve(self, x_init=None, alpha_init=None, stepconst=1, **kwargs):
        # process argument
        if x_init is None:
            xk = np.zeros((self.p, 1))
        else:
            xk = x_init

        if self.n % self.config.batchsize == 0:
            self.num_batches = self.n // self.config.batchsize
        else:
            self.num_batches = self.n // self.config.batchsize + 1

        # collect stats
        self.iteration = 0
        self.time_so_far = 0
        start = time.time()
        F_seq = []
        nz_seq = []
        grad_error_seq = []
        x_seq = []
        time_seq = []
        self.best_sol_so_far = xk

        if self.stepsize_strategy == "const":
            self.alphak = alpha_init
        elif self.stepsize_strategy == "increasing":
            if alpha_init is not None:
                print("Warning: alpha_init is provided but will not used in increasing stepsize strategy")
            self.alphak = np.sqrt(self.iteration + 1) / stepconst
        else:
            raise ValueError(f"Invalid stepsize_strategy:{self.stepsize_strategy}")
        # start computing
        dk = np.zeros_like(xk)
        while True:
            self.time_so_far = time.time() - start

            signal, gradfxk = self.check_termination(xk)
            self.grad_error = utils.l2_norm(dk - gradfxk)

            if self.config.save_seq:
                F_seq.append(self.Fxk)
                nz_seq.append(self.nz)
                grad_error_seq.append(self.grad_error)
                time_seq.append(self.time_so_far)
            if self.config.save_xseq:
                x_seq.append(csr_matrix(xk))
            if signal == "terminate":
                self.print_epoch()
                break
            else:
                self.num_data_pass += 1

            # print current epoch information
            if self.config.print_level > 0:
                if self.num_epochs % self.config.print_head_every == 0:
                    self.print_header()
                self.print_epoch()

            # new epoch
            self.num_epochs += 1
            batchidx = self.shuffleidx()
            if self.solve_mode == "inexact":
                epoch_inner_its = 0
                epoch_total_bak = 0            
            for i in range(self.num_batches):
                start_idx = i * self.config.batchsize
                end_idx = min(start_idx + self.config.batchsize, self.n)
                minibatch_idx = batchidx[start_idx:end_idx]
                _ = self.f.evaluate_function_value(xk, bias=0, idx=minibatch_idx)
                gradfxk_minibacth = self.f.gradient(xk, idx=minibatch_idx)
                dk = (self.iteration / (self.iteration + 1)) * dk + (1 / (self.iteration + 1)) * gradfxk_minibacth
                if self.solve_mode == "exact":
                    xkp1, _, _ = self.r.compute_proximal_gradient_update(np.zeros((self.p, 1)), self.alphak, dk)
                elif self.solve_mode == "inexact":
                    xkp1, ykp1 = self.r.compute_inexact_proximal_gradient_update(
                        np.zeros((self.p, 1)), self.alphak, dk, self.yk, self.stepsize_init, ipg_kwargs={'iteration':self.iteration + 1, 'xref':self.best_sol_so_far})
                    self.yk = ykp1
                    self.stepsize_init = self.r.stepsize
                    epoch_inner_its += self.r.inner_its
                    epoch_total_bak += self.r.total_bak                    
                    if self.r.flag != 'maxiter':
                        self.best_sol_so_far = xkp1
                    # sanity check
                    if self.r.gap < 0:
                        self.status = -3 # something wrong maye be stepsize too large
                        break
                    if self.config.ipg_save_log:
                        if i % 40 == 0:
                            self.r.print_header(filename=self.ipg_log_filename)
                        self.r.print_iteration(epoch = self.num_epochs, batch=i+1, filename=self.ipg_log_filename)
                else:
                    raise ValueError(f"Invalid solve_mode:{self.solve_mode}")
                self.iteration += 1
                xk = deepcopy(xkp1)
                # adjust stepsize
                if self.stepsize_strategy == "increasing":
                    self.alphak = np.sqrt(self.iteration + 1) / stepconst
            if self.solve_mode == "inexact":
                self.kwargs['total_bak_seq'].append(epoch_total_bak)
                self.kwargs['inner_its_seq'].append(epoch_inner_its)
        # return solutions
        self.xend = xk
        self.Fend = self.Fxk

        self.print_exit()
        return self.collect_info(xk, F_seq, nz_seq, grad_error_seq, x_seq, time_seq, **self.kwargs)

    def print_header(self):
        if self.config.compute_optim:
            header = " Epoch.   Obj.    alphak      #z   #nz   |egradf| |   optim     #pz    #pnz |"
        else:
            header = " Epoch.   Obj.    alphak      #z   #nz   |egradf|"
        header += "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(header)
        else:
            print(header)

    def print_epoch(self):
        if self.config.compute_optim:
            contents = f" {self.num_epochs:5d} {self.Fxk:.3e} {self.alphak:.3e} {self.nz:5d} {self.nnz:5d}  {self.grad_error:.3e} | {self.optim:.3e} {self.pz:5d}  {self.pnz:5d}  |"
        else:
            contents = f" {self.num_epochs:5d} {self.Fxk:.3e} {self.alphak:.3e} {self.nz:5d} {self.nnz:5d}  {self.grad_error:.3e} "                
        contents += "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)
