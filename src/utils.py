'''
# File: utils.py
# Project: ipg
# Created Date: 2021-08-23 11:31
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2022-09-27 9:34
# Modified By: Yutong Dai yutongdai95@gmail.com
# 
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
'''

import random
import numpy as np
import scipy
from scipy.io import loadmat
from sklearn.datasets import load_svmlight_file
from scipy.sparse import issparse
from numba import jit
import warnings
import os
from numba_progress import ProgressBar


def mkdirs(dirpath):
    if not os.path.exists(dirpath):
        # multi-threading
        os.makedirs(dirpath, exist_ok=True)


def is_finished(tag):
    if os.path.exists(tag):
        print(f"Task:{tag} is finished. Exiting...")
        return True


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def l2_norm(x):
    return np.sqrt(np.dot(x.T, x))[0][0]


def linf_norm(x):
    return np.max(np.abs(x))


def set_up_xy(datasetName, fileType='txt', dbDir='../db', to_dense=False):
    filepath = dbDir + "/{}.{}".format(datasetName, fileType)
    if fileType != 'mat':
        if fileType != 'npy':
            data = load_svmlight_file(filepath)
            X, y = data[0], data[1].reshape(-1, 1)
        else:
            data = np.load(filepath, allow_pickle=True).item()
            X, y = data['X'], data['y'].reshape(-1, 1)
        if to_dense:
            print("  Begin converting {}...".format(datasetName))
            X = X.toarray()
            print("  Finish converting!")
        return X, y
    else:
        data_dict = loadmat(filepath)
        try:
            return data_dict['A'], data_dict['b']
        except KeyError:
            print("Invalid matlab data file path... I cannot find X and y.")


def gen_group(p, K):
    group = K * np.ones(p)
    size = int(np.floor(p / K))
    starts = []
    ends = []
    group_frequency = []
    for i in range(K):
        start_ = i * size
        end_ = start_ + size
        starts.append(start_)
        ends.append(end_)
        group_frequency.append(end_ - start_)
    if ends[-1] < p:
        ends[-1] = p
        group_frequency[-1] = ends[-1] - starts[-1]
    group = {'starts': np.array(starts), 'ends': np.array(ends), 'group_frequency': np.array(group_frequency)}
    return group


def lam_max(X, y, group, loss='logit'):
    """
    Reference: Yi Yang and Hui Zou. A fast unified algorithm for solving group-lasso penalize learning problems. Page 22.
    """
    # beta_1 = np.log(np.sum(y == 1) / np.sum(y == -1))
    beta_1 = np.zeros((X.shape[1], 1))
    lam_max = -1
    group_frequency = group['group_frequency']
    K = len(group_frequency)
    if loss == 'logit':
        ys = y / (1 + np.exp(y * (X @ beta_1)))
        nabla_L = X.T @ ys / X.shape[0]
    elif loss == 'ls':
        ys = y - X @ beta_1
        nabla_L = (ys.T @ X).T / X.shape[0]
    else:
        raise ValueError("Invalid loss!")
    if issparse(nabla_L):
        nabla_L = nabla_L.toarray()
        print('  nabla_L to dense...', flush=True)
    with ProgressBar(total=K) as progress_proxy:
        lam_max = lam_max_jit(group['starts'], group['ends'], group_frequency, nabla_L, progress_proxy)
    return lam_max


@jit(nopython=True, cache=True)
def lam_max_jit(starts, ends, group_frequency, nabla_L, progress_proxy):
    K = len(group_frequency)
    lam_max = -1
    for i in range(K):
        sub_grp = nabla_L[starts[i]:ends[i]]
        temp = np.sqrt(np.dot(sub_grp.T, sub_grp))[0][0] / np.sqrt(group_frequency[i])
        if temp > lam_max:
            lam_max = temp
        progress_proxy.update(1)
    return lam_max

# def estimate_lipschitz(A, loss='logit'):
#     m, n = A.shape
#     if loss == 'ls':
#         hess = A.T @ A / m
#     elif loss == 'logit':
#         # acyually this is an upper bound on hess
#         hess = A.T @ A / (4 * m)
#     hess = hess.toarray()
#     L = np.max(np.linalg.eigvalsh(hess))
#     return L


def estimate_lipschitz(A, loss='logit'):
    m, n = A.shape
    if loss == 'ls':
        hess = A.T @ A / m
    elif loss == 'logit':
        # acyually this is an upper bound on hess
        hess = A.T @ A / (4 * m)
    # compute the largest eigenval
    eigenval, eigenvec = scipy.sparse.linalg.eigsh(hess, k=1)
    L = eigenval[0]
    return L


class GenOverlapGroup:
    def __init__(self, dim, grp_size=None, overlap_ratio=None):
        self.dim = dim
        self.grp_size = grp_size
        self.overlap_ratio = overlap_ratio

    def get_group(self):
        if self.grp_size is not None and self.overlap_ratio is not None:
            if self.grp_size >= self.dim:
                raise ValueError(
                    "grp_size is too large that each group has all variables.")
            overlap = int(self.grp_size * self.overlap_ratio)
            if overlap < 1:
                msg = "current config of grp_size and overlap_ratio cannot produce overlapping groups.\n"
                msg += "overlap_ratio is adjusted to have at least one overlap."
                warnings.warn(msg)
                overlap = 1
            groups = []
            starts = []
            ends = []
            start = 0
            end = self.grp_size - 1
            while True:
                starts.append(start)
                ends.append(end)
                groups.append([*range(start, end + 1)])
                # update
                start = end - (overlap - 1)
                end = min(start + self.grp_size - 1, self.dim - 1)
                if end == ends[-1]:
                    break
            return groups, starts, ends
        else:
            raise ValueError("check your inputs!")


def calculate_id_params(x, gradfx, weights, starts, ends):
    K = len(starts)
    Delta = np.inf
    delta = np.inf
    zg = []
    for g in range(K):
        start, end = starts[g], ends[g]
        xg = x[start:end]
        xg_norm = l2_norm(xg)
        if xg_norm > 0:
            Delta = min(Delta, xg_norm)
            # print(f'xg_norm:{xg_norm:.3e} | Delta:{Delta:.3e}')
        else:
            gradfxg_norm = l2_norm(gradfx[start:end])
            delta = min(weights[g] - gradfxg_norm, delta)
            zg.append(g)
            # print(f'xg_norm:{xg_norm:.3e} | delta:{delta:.3e}')
    return delta, Delta, zg


fileTypeDict = {}
fileTypeDict['a9a'] = 'npy'
fileTypeDict['australian'] = 'npy'
fileTypeDict['breast_cancer'] = 'npy'
fileTypeDict['cod_rna'] = 'npy'
fileTypeDict['colon_cancer'] = 'npy'
fileTypeDict['covtype'] = 'npy'
fileTypeDict['diabetes'] = 'npy'
fileTypeDict['duke'] = 'npy'
fileTypeDict['fourclass'] = 'npy'
fileTypeDict['german_numer'] = 'npy'
fileTypeDict['gisette'] = 'npy'
fileTypeDict['heart'] = 'npy'
fileTypeDict['ijcnn1'] = 'npy'
fileTypeDict['ionosphere'] = 'npy'
fileTypeDict['leu'] = 'npy'
fileTypeDict['liver_disorders'] = 'npy'
fileTypeDict['madelon'] = 'npy'
fileTypeDict['mushrooms'] = 'npy'
fileTypeDict['phishing'] = 'npy'
fileTypeDict['rcv1'] = 'npy'
fileTypeDict['skin_nonskin'] = 'npy'
fileTypeDict['sonar'] = 'npy'
fileTypeDict['splice'] = 'npy'
fileTypeDict['svmguide1'] = 'npy'
fileTypeDict['svmguide3'] = 'npy'
fileTypeDict['w8a'] = 'npy'
fileTypeDict['HIGGS'] = 'npy'
fileTypeDict['news20'] = 'npy'
fileTypeDict['real-sim'] = 'npy'
fileTypeDict['SUSY'] = 'npy'
fileTypeDict['url_combined'] = 'npy'
fileTypeDict['avazu-app.tr'] = 'npy'
fileTypeDict['kdda'] = 'npy'



# fileTypeDict = {}
# fileTypeDict['a9a'] = 'txt'
# fileTypeDict['australian'] = 'txt'
# fileTypeDict['breast_cancer'] = 'txt'
# fileTypeDict['cod_rna'] = 'txt'
# fileTypeDict['colon_cancer'] = 'bz2'
# fileTypeDict['covtype'] = 'txt'
# fileTypeDict['diabetes'] = 'txt'
# fileTypeDict['duke'] = 'bz2'
# fileTypeDict['fourclass'] = 'txt'
# fileTypeDict['german_numer'] = 'txt'
# fileTypeDict['gisette'] = 'bz2'
# fileTypeDict['heart'] = 'txt'
# fileTypeDict['ijcnn1'] = 'txt'
# fileTypeDict['ionosphere'] = 'txt'
# fileTypeDict['leu'] = 'bz2'
# fileTypeDict['liver_disorders'] = 'txt'
# fileTypeDict['madelon'] = 'txt'
# fileTypeDict['mushrooms'] = 'txt'
# fileTypeDict['phishing'] = 'txt'
# fileTypeDict['rcv1'] = 'txt'
# fileTypeDict['skin_nonskin'] = 'txt'
# fileTypeDict['sonar'] = 'txt'
# fileTypeDict['splice'] = 'txt'
# fileTypeDict['svmguide1'] = 'txt'
# fileTypeDict['svmguide3'] = 'txt'
# fileTypeDict['w8a'] = 'txt'
# fileTypeDict['HIGGS'] = 'txt'
# fileTypeDict['news20'] = 'txt'
# fileTypeDict['real-sim'] = 'txt'
# fileTypeDict['SUSY'] = 'txt'
# fileTypeDict['url_combined'] = 'npy'
# fileTypeDict['avazu-app.tr'] = 'npy'
# fileTypeDict['kdda'] = 'npy'


# fileTypeDict['bodyfat_scale'] = 'txt'
# fileTypeDict['bodyfat_scale_expanded7'] = 'mat'
# fileTypeDict['bodyfat_scale_expanded2'] = 'mat'
# fileTypeDict['bodyfat_scale_expanded1'] = 'mat'
# fileTypeDict['bodyfat_scale_expanded5'] = 'mat'
# fileTypeDict['YearPredictionMSD.t_expanded1'] = 'mat'
# fileTypeDict['space_ga_scale_expanded1'] = 'mat'
# fileTypeDict['space_ga_scale_expanded5'] = 'mat'
# fileTypeDict['YearPredictionMSD.t'] = 'bz2'

# # regression
# fileTypeDict['abalone_scale'] = 'txt'
# fileTypeDict['bodyfat_scale'] = 'txt'
# fileTypeDict['cadata'] = 'txt'
# fileTypeDict['cpusmall_scale'] = 'txt'
# fileTypeDict['eunite2001'] = 'txt'
# fileTypeDict['housing_scale'] = 'txt'
# fileTypeDict['mg_scale'] = 'txt'
# fileTypeDict['mpg_scale'] = 'txt'
# fileTypeDict['pyrim_scale'] = 'txt'
# fileTypeDict['space_ga_scale'] = 'txt'
# fileTypeDict['E2006.train'] = 'txt'
# fileTypeDict['log1p.E2006.train'] = 'txt'
# fileTypeDict['YearPredictionMSD'] = 'txt'
# fileTypeDict['blogData_train'] = 'txt'
# fileTypeDict['UJIIndoorLoc'] = 'txt'
# fileTypeDict['driftData'] = 'txt'
# fileTypeDict['virusShare'] = 'txt'
# fileTypeDict['triazines_scale'] = 'txt'


# fileTypeDict['bodyfat_scale_expanded7'] = 'mat'
# fileTypeDict['pyrim_scale_expanded5'] = 'mat'
# fileTypeDict['triazines_scale_expanded4'] = 'mat'
# fileTypeDict['housing_scale_expanded7'] = 'mat'
# gammas = {}
# gammas['bodyfat_scale_expanded7'] = [1e-4, 1e-5, 1e-6]
# gammas['pyrim_scale_expanded5'] = [1e-2, 1e-3, 1e-4]
# gammas['triazines_scale_expanded4'] = [1e-2, 1e-3, 1e-4]
# gammas['housing_scale_expanded7'] = [1e-2, 1e-3, 1e-4]
# groups = {}
# groups['bodyfat_scale_expanded7'] = 388
# groups['pyrim_scale_expanded5'] = 671
# groups['triazines_scale_expanded4'] = 2118
# groups['housing_scale_expanded7'] = 258
