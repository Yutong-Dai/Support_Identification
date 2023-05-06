import spams
import numpy as np
import sys
sys.path.append('../')
from src.funcs.regularizer import TreeOG, NatOG
from scipy.sparse import csc_matrix
from test_NatOG import get_config
from src.utils import gen_tree

config = get_config()
############################## test case 1 ####################################
print("=== test case 1 ===")
weights = np.array([1.0, 20.0, 1.0])
penalty = 0.1
groups, tree, dot = gen_tree(nodes_list=[[0,1], [2,3,4], [5,6,7]], nodes_relation_dict={0:[1,2]}, penalty=penalty, weights=weights)
for i,g in enumerate(groups):
    print(f"G_{i}:", g)
dot.attr(bgcolor='transparent', ratio="compress")
dot.render(filename='case1', directory='./graphviz', cleanup=True, format='pdf')
print("tree structure is visualized in ./graphviz/case1.pdf")
rtree = TreeOG(groups, tree, penalty, weights=weights)
rnatog = NatOG(groups, penalty=penalty, config=config, weights=weights)
p = len(groups[0])
np.random.seed(0)
xk = np.random.randn(p, 1)
dk = np.random.randn(p, 1)
alphak = 1.0

spams_sol, spams_nnz, spams_nz = rtree.compute_exact_proximal_gradient_update(xk, alphak, dk, compute_structure=True)
print("=== spams sol ===")
print('spams_sol', spams_sol.T, '\nspams_nnz', spams_nnz, 'spams_nz', spams_nz)


mysol, _ = rnatog.compute_inexact_proximal_gradient_update(xk, alphak, dk, y_init=None, stepsize_init=None, ipg_kwargs={'iteration':1, 'xref':xk})
rnatog.print_header(filename='log.txt')
rnatog.print_iteration(epoch = 1, batch=1, filename='log.txt')
mynnz, mynz = rnatog._get_group_structure(mysol)
print("=== my sol===")
print('mysol', mysol.T, '\nmynnz', mynnz, 'mynz', mynz)


############################## test case 2 ####################################
print("=== test case 2 ===")
weights = np.array([1.0, 1.0, 1.0, 2000.0, 1.0, 1.0, 200.0, 1.0])
penalty = 0.1
groups, tree, dot = gen_tree(nodes_list=[[], [0,1,2], [3,4], [5], [], [6,7], [8], [9]], 
                            nodes_relation_dict={0:[1,4], 1:[2,3], 4:[5,6], 6:[7]}, 
                            penalty=penalty, weights=weights)
for i,g in enumerate(groups):
    print(f"G_{i}:", g)
dot.attr(bgcolor='transparent', ratio="compress")
dot.render(filename='case2', directory='./graphviz', cleanup=True, format='pdf')
print("tree structure is visualized in ./graphviz/case2.pdf")
rtree = TreeOG(groups, tree, penalty, weights=weights)
rnatog = NatOG(groups, penalty=penalty, config=config, weights=weights)
p = len(groups[0])
np.random.seed(0)
xk = np.random.randn(p, 1)
dk = np.random.randn(p, 1)
alphak = 1.0

spams_sol, spams_nnz, spams_nz = rtree.compute_exact_proximal_gradient_update(xk, alphak, dk, compute_structure=True)
print("=== spams sol ===")
print('spams_sol', spams_sol.T, '\nspams_nnz', spams_nnz, 'spams_nz', spams_nz)


mysol, _ = rnatog.compute_inexact_proximal_gradient_update(xk, alphak, dk, y_init=None, stepsize_init=None, ipg_kwargs={'iteration':1, 'xref':xk})
rnatog.print_header(filename='log.txt')
rnatog.print_iteration(epoch = 1, batch=1, filename='log.txt')
mynnz, mynz = rnatog._get_group_structure(mysol)
print("=== my sol===")
print('mysol', mysol.T, '\nmynnz', mynnz, 'mynz', mynz)

############################## test case 3 ####################################
print("=== test case 3 ===")
weights = np.array([1.0, 1.0, 2000.0, 200.0, 1.0, 1.0, 1.0, 1.0])
penalty = 0.1
groups, tree, dot = gen_tree(nodes_list=[[0], [1], [2], [3], [4], [5], [6], [7]], 
                            nodes_relation_dict={0:[1,2,3], 1:[4,5], 2:[6], 6:[7]},
                            penalty=penalty, weights=weights)
for i,g in enumerate(groups):
    print(f"G_{i}:", g)
dot.attr(bgcolor='transparent', ratio="compress")
dot.render(filename='case3', directory='./graphviz', cleanup=True, format='pdf')
print("tree structure is visualized in ./graphviz/case3.pdf")
rtree = TreeOG(groups, tree, penalty, weights=weights)
rnatog = NatOG(groups, penalty=penalty, config=config, weights=weights)
p = len(groups[0])
np.random.seed(0)
xk = np.random.randn(p, 1)
dk = np.random.randn(p, 1)
alphak = 1.0

spams_sol, spams_nnz, spams_nz = rtree.compute_exact_proximal_gradient_update(xk, alphak, dk, compute_structure=True)
print("=== spams sol ===")
print('spams_sol', spams_sol.T, '\nspams_nnz', spams_nnz, 'spams_nz', spams_nz)


mysol, _ = rnatog.compute_inexact_proximal_gradient_update(xk, alphak, dk, y_init=None, stepsize_init=None, ipg_kwargs={'iteration':1, 'xref':xk})
rnatog.print_header(filename='log.txt')
rnatog.print_iteration(epoch = 1, batch=1, filename='log.txt')
mynnz, mynz = rnatog._get_group_structure(mysol)
print("=== my sol===")
print('mysol', mysol.T, '\nmynnz', mynnz, 'mynz', mynz)
