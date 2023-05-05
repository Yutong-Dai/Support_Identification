import spams
import numpy as np
import sys
sys.path.append('../')
from src.funcs.regularizer import TreeOG, NatOG
from scipy.sparse import csc_matrix
from regularizer import get_config

g0 = [0, 1, 2, 3, 4, 5, 6, 7]
g1 = [2, 3, 4]
g2 = [5, 6, 7]
groups = [g0, g1, g2]
weights = np.array([1.0, 20.0, 1.0])
penalty = 0.1

# pointer to the first variable of each group
own_variables = np.array([0, 2, 5], dtype=np.int32)
# number of "root" variables in each group
N_own_variables = np.array([2, 3, 3], dtype=np.int32)
# (variables that are in a group, but not in its descendants).
# for instance root(g1)={0,1}, root(g2)={2 3 4}, root(g3)={5 6 7 8 9}
# weights for each group, they should be non-zero to use fenchel duality
_groups_bool = np.asfortranarray([[0, 0, 0],
                                  [1, 0, 0],
                                  [1, 0, 0]], dtype=bool)
# first group should always be the root of the tree
# non-zero entriees mean inclusion relation ship, here g2 is a children of g1,
# g3 is a children of g1
_groups_bool = csc_matrix(_groups_bool, dtype=bool)
tree = {'eta_g': weights * penalty, 'groups': _groups_bool, 'own_variables': own_variables, 'N_own_variables': N_own_variables}

rtree = TreeOG(groups, tree, penalty, weights=weights)

p = len(g0)
np.random.seed(0)
xk = np.random.randn(p, 1)
dk = np.random.randn(p, 1)
alphak = 1.0

spams_sol, spams_nnz, spams_nz = rtree.compute_exact_proximal_gradient_update(xk, alphak, dk, compute_structure=True)
print("=== spams ===")
print('spams_sol', spams_sol.T, '\nspams_nnz', spams_nnz, 'spams_nz', spams_nz)


config = get_config()
rnatog = NatOG(groups, penalty=penalty, config=config, weights=weights)
mysol, _ = rnatog.compute_inexact_proximal_gradient_update(xk, alphak, dk, 
    y_init=None, stepsize_init=None, ipg_kwargs={'iteration':1, 'xref':xk})
rnatog.print_header(filename='log.txt')
rnatog.print_iteration(epoch = 1, batch=1, filename='log.txt')
mynnz, mynz = rnatog._get_group_structure(mysol)
print("=== my ===")
print('mysol', mysol.T, '\nmynnz', mynnz, 'mynz', mynz)