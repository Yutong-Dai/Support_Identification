import argparse
import sys
sys.path.append("../")
from src.funcs.regularizer import  NatOG
from src.utils import gen_chain_group
import numpy as np
def get_config():
    parser = argparse.ArgumentParser()
    # IPG solver configurations
    parser.add_argument("--ipg_save_log", default=True, type=lambda x: (str(x).lower()
                        in ['true', '1', 'yes']), help="whether save the log of the ipg solver.")
    parser.add_argument("--exact_pg_computation", default=False, type=lambda x: (str(x).lower()
                        in ['true', '1', 'yes']), help="Mimic to the exact computation of the proximal operator.") 
    parser.add_argument("--exact_pg_computation_tol", type=float, default=1e-15, help="Deisred error tolerance.")                                                                      

    parser.add_argument("--ipg_do_linesearch", default=True, type=lambda x: (str(x).lower()
                        in ['true', '1', 'yes']), help="Whether do linesearch in the projected gradient ascent.")
    parser.add_argument("--ipg_linesearch_eta", type=float, default=1e-4, help="eta of the linesearch.")
    parser.add_argument("--ipg_linesearch_xi", type=float, default=0.8, help="xi of the linesearch.")
    parser.add_argument("--ipg_linesearch_beta", type=float, default=1.2, help="beta of the linesearch.")
    parser.add_argument("--ipg_linesearch_limits", type=int, default=100, help="max attempts of the linesearch.")
    
    parser.add_argument("--ipg_strategy", type=str, default="diminishing", choices=["diminishing"], 
        help="Strategy to inexactly evaluate the proximal operator.\ndiminishing: c * np.log(k+1) / k**delta")
    parser.add_argument("--ipg_diminishing_c", type=float, default=1, help="c of c * np.log(k+1) / k**delta")
    parser.add_argument("--ipg_diminishing_delta", type=float, default=2, help="delta of c * np.log(k+1) / k**delta")
    config = parser.parse_args()
    return config

if __name__ == '__main__':
    group = gen_chain_group(10, 5, 0.4)
    print(group)
    assert group['groups'] == [[0, 1, 2, 3, 4], [3, 4, 5, 6, 7], [6, 7, 8, 9]]
    r = NatOG(groups, penalty=1.0, config=config, weights=np.array([1,100,1]))

    groups = [[0,1,2], [1,2,3], [0,2,4]]
    print('using:', groups)
    config = get_config()
    r = NatOG(groups, penalty=1.0, config=config, weights=np.array([1,100,1]))
    print(isinstance(r, NatOG))
    print(r.A.todense())
    print(r.xdim)
    np.random.seed(0)
    x0 = np.random.randn(r.xdim, 1)
    print(r.func(x0))
    xtrial, ytrial = r.compute_inexact_proximal_gradient_update(xk=x0, alphak=1.0, dk=np.random.randn(r.xdim, 1), 
                                                y_init=None, stepsize_init=None, 
                                                ipg_kwargs={'iteration':10})
    print(f'target duality gap:{r.targap:.3f} | subsolver status:{r.flag} | aoptim:{r.aoptim:.3f}')
    print('init x             :', x0.T)
    print('approxmate solution:', xtrial.T)