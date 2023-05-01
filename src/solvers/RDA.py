'''
# File: RDA.py
# Project: solvers
# Created Date: 2022-09-05 11:23
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2023-05-01 7:39
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
from copy import deepcopy
from scipy.sparse import csr_matrix
import sys
sys.path.append("../../")
import src.utils as utils
from src.solvers.BaseSolver import StoBaseSolver


class RDA(StoBaseSolver):
    def __init__(self, f, r, config):
        self.stepsize_strategy = config.rda_stepsize
        self.version = "0.1 (2022-09-01)"
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

        if self.stepsize_strategy == "const":
            self.alphak = alpha_init
        elif self.stepsize_strategy == "increasing":
            self.alphak = np.sqrt(self.iteration + 1) / stepconst
        else:
            raise ValueError(f"Invalid stepsize_strategy:{self.stepsize_strategy}")
        self.compute_id_quantity = False

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
                gradfxk_minibacth = self.f.gradient(xk, idx=minibatch_idx)
                dk = (self.iteration / (self.iteration + 1)) * dk + (1 / (self.iteration + 1)) * gradfxk_minibacth

                xkp1, _, _ = self.r.compute_proximal_gradient_update(np.zeros((self.p, 1)), self.alphak, dk)
                self.iteration += 1
                xk = deepcopy(xkp1)
                # adjust stepsize
                if self.stepsize_strategy == "increasing":
                    self.alphak = np.sqrt(self.iteration + 1) / stepconst

        # return solutions
        self.xend = xk
        self.Fend = self.Fxk

        self.print_exit()
        return self.collect_info(xk, F_seq, nz_seq, grad_error_seq, x_seq)

    def print_header(self):
        header = " Epoch.   Obj.    alphak      #z   #nz   |egradf| |   optim     #pz    #pnz |"
        if self.compute_id_quantity:
            header += "  id_left    id_right  maxq<delta |"
        header += "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(header)
        else:
            print(header)

    def print_epoch(self):
        contents = f" {self.num_epochs:5d} {self.Fxk:.3e} {self.alphak:.3e} {self.nz:5d} {self.nnz:5d}  {self.grad_error:.3e} | {self.optim:.3e} {self.pz:5d}  {self.pnz:5d}  |"
        if self.compute_id_quantity:
            contents += f" {self.id_left:.3e}  {self.id_rifgt:.3e}    {str(self.check):5s}    |"
        contents += "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)
