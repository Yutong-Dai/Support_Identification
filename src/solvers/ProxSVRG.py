import numpy as np
import datetime
import time
import os
import sys
from copy import deepcopy
from scipy.sparse import csr_matrix
sys.path.append("../../")
import src.utils as utils
from src.solvers.BaseSolver import StoBaseSolver


class ProxSVRG(StoBaseSolver):
    def __init__(self, f, r, config):
        self.stepsize_strategy = config.proxsvrg_stepsize
        self.version = "0.1 (2022-09-01)"
        self.solver = "ProxSVRG"
        super().__init__(f, r, config)

    def solve(self, x_init=None, alpha_init=None):
        # process argument
        if x_init is None:
            xk = np.zeros((self.p, 1))
        else:
            xk = x_init
        if not alpha_init:
            self.alphak = 1.0
        else:
            self.alphak = alpha_init

        if self.n % self.config.batchsize == 0:
            self.num_batches = self.n // self.config.batchsize
        else:
            self.num_batches = self.n // self.config.batchsize + 1
        # collect stats
        start = time.time()

        F_seq = []
        nz_seq = []
        grad_error_seq = []
        x_seq = []
        vk = 1e17
        while True:
            self.time_so_far = time.time() - start

            signal, gradfxk = self.check_termination(xk)
            self.grad_error = utils.l2_norm(vk - gradfxk)

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
                # evaluate the full gradient inside the check_termination, which is reuqired in ProxSVRG
                self.num_data_pass += 1

            # print current epoch information
            if self.config.print_level > 0:
                if self.num_epochs % self.config.print_head_every == 0:
                    self.print_header()
                self.print_epoch()

            # new epoch
            self.num_epochs += 1
            x_minibatch = deepcopy(xk)
            # end of the epoch returning the average iterate
            if self.config.proxsvrg_epoch_iterate == 'average':
                x_running = np.zeros_like(xk)
            # time_one_epoch_start = time.time()
            for m in range(self.config.proxsvrg_inner_repeat):
                batchidx = self.shuffleidx()
                for i in range(self.num_batches):
                    # time_one_minibatch_start = time.time()
                    start_idx = i * self.config.batchsize
                    end_idx = min(start_idx + self.config.batchsize, self.n)
                    minibatch_idx = batchidx[start_idx:end_idx]
                    _ = self.f.evaluate_function_value(x_minibatch, bias=0, idx=minibatch_idx)
                    gradf_minibacth = self.f.gradient(x_minibatch, idx=minibatch_idx)
                    # old grad
                    _ = self.f.evaluate_function_value(xk, bias=0, idx=minibatch_idx)
                    gradf_minibacth_old = self.f.gradient(xk, idx=minibatch_idx)
                    vk = gradf_minibacth - gradf_minibacth_old + gradfxk
                    x_minibatch, _, _ = self.r.compute_proximal_gradient_update(x_minibatch, self.alphak, vk)
                    if self.config.proxsvrg_epoch_iterate == 'average':
                        x_running += x_minibatch
                    # time_one_minibatch = time.time() - time_one_minibatch_start
                    # print(f"{i+1}/{self.num_batches}:{time_one_minibatch:.1f} secs")
                self.num_data_pass += 1
                # evaluate the function value after a full pass over the data
                if self.config.proxsvrg_epoch_iterate == 'average':
                    x_full_pass = x_running / ((m + 1) * self.num_batches)
                else:
                    x_full_pass = x_minibatch
                F_full_pass = self.f.func(x_full_pass) + self.r.func(x_full_pass)
                if self.config.save_seq:
                    F_seq.append(F_full_pass)
                    _, nz_full_pass = self.r._get_group_structure(x_full_pass)
                    nz_seq.append(nz_full_pass)
                if self.config.save_xseq:
                    x_seq.append(csr_matrix(x_full_pass))

                    # form the stochastic gradient estimate at the x_full_pass
                    gradfxk_full_pass = self.f.gradient(x_full_pass, idx=None)
                    minibatch_idx = batchidx[0:min(self.config.batchsize, self.n)]
                    _ = self.f.evaluate_function_value(x_full_pass, bias=0, idx=minibatch_idx)
                    gradf_minibacth = self.f.gradient(x_full_pass, idx=minibatch_idx)
                    # old grad
                    _ = self.f.evaluate_function_value(xk, bias=0, idx=minibatch_idx)
                    gradf_minibacth_old = self.f.gradient(xk, idx=minibatch_idx)
                    vk_full_pass = gradf_minibacth - gradf_minibacth_old + gradfxk

                    grad_error_seq.append(utils.l2_norm(vk_full_pass - gradfxk_full_pass))
            # time_one_epoch= time.time() - time_one_epoch_start
            # print(f"one epoch:{time_one_epoch:.1f} secs")
            # move to the new major iterate
            xk = deepcopy(x_full_pass)
            vk = deepcopy(vk_full_pass)
        # return solutions
        self.xend = xk
        self.Fend = self.Fxk
        self.print_exit()
        return self.collect_info(xk, F_seq, nz_seq, grad_error_seq, x_seq)

    def print_header(self):
        header = " Epoch.   Obj.    alphak      #z   #nz   |egradf| |   optim     #pz    #pnz |"
        header += "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(header)
        else:
            print(header)

    def print_epoch(self):
        contents = f" {self.num_epochs:5d} {self.Fxk:.3e} {self.alphak:.3e} {self.nz:5d} {self.nnz:5d}  {self.grad_error:.3e} | {self.optim:.3e} {self.pz:5d}  {self.pnz:5d}  |"
        contents += "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)
