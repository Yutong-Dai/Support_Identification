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


class SPStorm(StoBaseSolver):
    def __init__(self, f, r, config):
        self.stepsize_strategy = config.spstorm_stepsize
        self.solver = "SPStorm"
        super().__init__(f, r, config)

    def solve(self, x_init=None, alpha_init=None, Lg=None, **kwargs):
        # process argument
        if x_init is None:
            xk = np.zeros((self.p, 1))
        else:
            xk = x_init
        xkm1 = deepcopy(xk)
        zk = deepcopy(xk)

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
        self.best_sol_so_far = xk

        if self.stepsize_strategy == "const":
            self.alphak = alpha_init
        else:
            raise ValueError(f"Invalid stepsize_strategy:{self.stepsize_strategy}")
        self.betak = 1.0
        if self.config.spstorm_zeta == 'dynanmic':
            self.zeta = 1.0
        elif isinstance(self.config.spstorm_zeta, float):
            self.zeta = self.config.spstorm_zeta
        else:
            raise ValueError(f"Invalid zeta parameter:{self.config.spstorm_zeta}")

        # start computing
        dk = 1e17
        while True:
            self.time_so_far = time.time() - start
            signal, gradfxk = self.check_termination(xk)
            # check sparsity at zk
            self.nnz, self.nz = self.r._get_group_structure(zk)

            self.grad_error = utils.l2_norm(dk - gradfxk)

            if self.config.save_seq:
                F_seq.append(self.Fxk)
                nz_seq.append(self.nz)
                grad_error_seq.append(self.grad_error)
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
            for i in range(self.num_batches):
                start_idx = i * self.config.batchsize
                end_idx = min(start_idx + self.config.batchsize, self.n)
                minibatch_idx = batchidx[start_idx:end_idx]
                _ = self.f.evaluate_function_value(xk, bias=0, idx=minibatch_idx)
                vk = self.f.gradient(xk, idx=minibatch_idx)
                if self.iteration == 0:
                    dk = vk
                else:
                    _ = self.f.evaluate_function_value(xkm1, bias=0, idx=minibatch_idx)
                    uk = self.f.gradient(xkm1, idx=minibatch_idx)
                    if self.stepsize_strategy == "const":
                        if self.config.spstorm_betak == -1.0:
                            self.betak = 1.0 / (self.iteration + 1)
                        else:
                            self.betak = self.config.spstorm_betak
                    else:
                        raise ValueError(f"Unknown stepsize strategy:{self.stepsize_strategy}")

                    dk = vk + (1 - self.betak) * (dkm1 - uk)
                # new iterate
                if self.solve_mode == "exact":
                    zk, _, _ = self.r.compute_proximal_gradient_update(xk, self.alphak, dk)
                elif self.solve_mode == "inexact":
                    zk, ykp1 = self.r.compute_inexact_proximal_gradient_update(
                        xk, self.alphak, dk, self.yk, self.stepsize_init, ipg_kwargs={'iteration':self.num_epochs, 'xref':self.best_sol_so_far})
                    self.yk = ykp1
                    self.stepsize_init = self.r.stepsize
                    if self.r.flag != 'maxiter':
                        self.best_sol_so_far = zk
                    if self.config.ipg_save_log:
                        if i % 40 == 0:
                            self.r.print_header(filename=self.ipg_log_filename)
                        self.r.print_iteration(epoch = self.num_epochs, batch=i+1, filename=self.ipg_log_filename)
                            
                if self.config.spstorm_zeta == 'dynanmic' and self.iteration > 0:
                    self.zeta = self.iteration
                xkp1 = xk + self.zeta * self.betak * (zk - xk)
                self.iteration += 1
                dkm1 = deepcopy(dk)
                xkm1 = deepcopy(xk)
                xk = deepcopy(xkp1)
                

        # return solutions
        self.xend = xk
        self.Fend = self.Fxk
        self.print_exit()
        return self.collect_info(xk, F_seq, nz_seq, grad_error_seq, x_seq)

    def print_header(self):
        if self.config.compute_optim:
            header = " Epoch.   Obj.    alphak     betak      zeta      #z   #nz   |egradf| |   optim     #pz    #pnz |"
        else:
            header = " Epoch.   Obj.    alphak     betak      zeta      #z   #nz   |egradf|"
        header += "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(header)
        else:
            print(header)

    def print_epoch(self):
        if self.config.compute_optim:
            contents = f" {self.num_epochs:5d} {self.Fxk:.3e} {self.alphak:.3e} {self.betak:.3e} {self.zeta:.3e} {self.nz:5d} {self.nnz:5d}  {self.grad_error:.3e} | {self.optim:.3e} {self.pz:5d}  {self.pnz:5d}  |"
        else:
            contents = f" {self.num_epochs:5d} {self.Fxk:.3e} {self.alphak:.3e} {self.betak:.3e} {self.zeta:.3e} {self.nz:5d} {self.nnz:5d}  {self.grad_error:.3e} "
        contents += "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)


