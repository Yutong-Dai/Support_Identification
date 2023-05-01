import argparse
from scipy.io import savemat, loadmat
import numpy as np
import os
import sys
sys.path.append("../")
import src.utils as utils
from src.funcs.lossfunction import LogisticLoss
from src.funcs.regularizer import GL1
from src.solvers.SPStorm import SPStorm
from src.solvers.PStorm import PStorm
from src.solvers.ProxSVRG import ProxSVRG
from src.solvers.ProxSAGA import ProxSAGA
from src.solvers.RDA import RDA
from src.solvers.FaRSAGroup.params import params as farsa_config
from src.solvers.FaRSAGroup.solve import solve as faras_solve
from copy import deepcopy
from scipy.sparse import csr_matrix
import time


def main(config):
    # Setup the problem
    fileType = utils.fileTypeDict[config.dataset]
    utils.mkdirs(f'./{config.purpose}/{config.solver}_{config.loss}')
    config.data_dir = os.path.expanduser(config.data_dir)
    tag = f'./{config.purpose}/{config.solver}_{config.loss}/{config.dataset}_frac:{config.frac}_lam_shrink:{config.lam_shrink}'
    if utils.is_finished(tag + '_stats.npy'):
        exit()
    print("Working on: {}...".format(config.dataset))
    X, y = utils.set_up_xy(config.dataset, fileType, dbDir=config.data_dir)
    if config.loss == 'logit':
        f = LogisticLoss(X, y, config.dataset)
    p = X.shape[1]
    num_of_groups = max(int(p * config.frac), 1)
    group = utils.gen_group(p, num_of_groups)
    lammax_path = f'{config.data_dir}/lammax/lammax-{config.dataset}-{config.frac}.mat'
    Lip_path = f'{config.data_dir}/Lip/Lip-{config.dataset}.mat'
    if os.path.exists(lammax_path):
        lammax = loadmat(lammax_path)["lammax"][0][0]
        print(f"loading lammax from: {lammax_path}")
    else:
        print(f"Compute lammax ...")
        lammax = utils.lam_max(X, y, group, config.loss)
        savemat(lammax_path, {"lammax": lammax})
        print(f"save lammax to: {lammax_path}")
    if os.path.exists(Lip_path):
        L = loadmat(Lip_path)["L"][0][0]
        print(f"loading Lipschitz constant from: {Lip_path}")
    else:
        print(f"Compute Lip ...")
        L = utils.estimate_lipschitz(X, config.loss)
        savemat(Lip_path, {"L": L})
        print(f"save Lipschitz constant to: {Lip_path}")
    Lambda = lammax * config.lam_shrink
    start = time.time()
    r = GL1(penalty=Lambda, groups=group)
    print(f"Spend {time.time()-start:.1f} seconds to initialize r.", flush=True)

    if config.solver == 'FaRSAGroup':
        farsa_config['tag'] = tag
        farsa_config['save_log'] = True
        info = faras_solve(f, r, X_initial=None, proxStepsize=None, method='gradient',
                           update_proxStepsize='single', params=farsa_config,
                           print_group=False, print_second_level=False,
                           kappa_1=1e-1, kappa_2=1e-2, print_time=True)

        stats = {'nnz': info['nnz'], 'F': info['F'], 'Xsol': csr_matrix(info['X']), 'zeroGroup': sorted(info['zeroGroup']),
                 'iteration': info['iteration'], 'time': info['time']}
        np.save(f"{tag}_stats.npy", stats)
        if info['status'] != 0:
            print(f"Exit without 0 code! Check log files at {tag}!")
    else:
        tag_base = deepcopy(tag)
        for run in range(config.runs):
            tag = deepcopy(tag_base)
            print(f"Process: {run+1} / {config.runs}")
            actual_seed = run + config.seed
            tag += f"_seed:{actual_seed}"

            utils.setup_seed(actual_seed)
            alpha_init = 0.1 / L

            if config.solver == 'ProxSVRG':
                tag += f'_proxsvrg_inner_repeat:{config.proxsvrg_inner_repeat}'
                config.tag = tag
                solver = ProxSVRG(f, r, config)
                info = solver.solve(x_init=None, alpha_init=alpha_init)

            elif config.solver == 'ProxSAGA':
                config.tag = tag
                solver = ProxSAGA(f, r, config)
                info = solver.solve(x_init=None, alpha_init=alpha_init)

            elif config.solver == 'SPStorm':
                tag += f'_spstorm_stepsize:{config.spstorm_stepsize}'
                tag += f'_spstorm_betak:{config.spstorm_betak}'
                tag += f'_spstorm_interpolate:{config.spstorm_interpolate}'
                tag += f'_spstorm_zeta:{config.spstorm_zeta}'
                config.tag = tag
                solver = SPStorm(f, r, config)
                info = solver.solve(x_init=None, alpha_init=alpha_init, Lg=L)

            elif config.solver == 'PStorm':
                tag += f'_pstorm_stepsize:{config.pstorm_stepsize}'
                tag += f'_pstorm_betak:{config.pstorm_betak}'
                config.tag = tag
                solver = PStorm(f, r, config)
                info = solver.solve(x_init=None, alpha_init=alpha_init, Lg=L)

            elif config.solver == 'RDA':
                tag += f'_rda_stepconst:{config.rda_stepconst}'
                config.tag = tag
                solver = RDA(f, r, config)
                info = solver.solve(x_init=None, alpha_init=alpha_init, stepconst=config.rda_stepconst)

            elif config.solver == 'ProxSG':
                tag += f'_proxsg_stepconst:{config.proxsg_stepconst}'
                config.tag = tag
                solver = ProxSG(f, r, config)
                info = solver.solve(x_init=None, alpha_init=alpha_init, stepconst=config.proxsg_stepconst)
            else:
                raise ValueError(f"Unrecognized solver:{config.solver}")

            np.save(f"{tag}_stats.npy", info)


def float_or_str(value):
    try:
        return float(value)
    except:
        return value


def get_config():
    parser = argparse.ArgumentParser()
    # Problem setup
    parser.add_argument("--purpose", type=str, default="test", help="The purpose for this experiment.")
    parser.add_argument("--data_dir", type=str, default="~/db", help="Directory for datasets.")
    parser.add_argument("--dataset", type=str, default="a9a", help='Dataset Name.')
    parser.add_argument("--loss", type=str, default="logit", choices=["logit", "ls"], help='Name of the loss funciton.')
    parser.add_argument("--lam_shrink", type=float, default=0.1, help="The lambda used is calculated as lambda=lambda_max * lam_shrink")
    parser.add_argument("--frac", type=float, default=0.3, help="num_of_groups = max(int(p * config.frac), 1)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="add a quadratic penalty on the loss function to make it stronglt convex; set to 0 to disable.")

    # solver shared arguments
    parser.add_argument("--solver", type=str, choices=["FaRSAGroup", "SPStorm", "PStorm",
                                                       "ProxSVRG", "ProxSAGA", "RDA"],
                        help="Solver used.",
                        default="SPStorm")
    parser.add_argument("--accuracy", type=float, default=1e-4, help="Temrminated the algorithm if reach the desired accuracy.")
    parser.add_argument("--max_iters", type=int, default=100, help="Temrminated the algorithm if reach max number of iterations.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Temrminated the algorithm if reach max number of epochs.")
    parser.add_argument("--max_time", type=float, default=3600.0, help="Temrminated the algorithm if reach max time.")
    parser.add_argument("--print_level", type=int, default=2, choices=[0, 1, 2], help="0: print nothing; 1: print basics; 2:print everthing.")
    parser.add_argument("--print_head_every", type=int, default=50, help="print header every print_every iteration.")
    parser.add_argument('--shuffle', default=True, type=lambda x: (str(x).lower()
                        in ['true', '1', 'yes']), help='Whether shuffle the dataset after a full pass.')
    parser.add_argument("--batchsize", type=int, default=256, help="Number of samples used to form the stochastic gradient estimate.")
    parser.add_argument("--save_seq", default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help="Whether save intermediate results.")
    parser.add_argument("--save_xseq", default=True, type=lambda x: (str(x).lower()
                        in ['true', '1', 'yes']), help="Whether save intermediate iterates sequence.")
    parser.add_argument('--seed', default=2022, type=int, help='Global random seed.')
    parser.add_argument('--runs', default=1, type=int, help='(For stochastic algorithms) Total numbers of repeated runs.')
    parser.add_argument("--optim_scaled", default=True, type=lambda x: (str(x).lower()
                        in ['true', '1', 'yes']), help="Whether scale the optimality measure by the stepsize.")
    parser.add_argument("--save_log", default=True, type=lambda x: (str(x).lower()
                        in ['true', '1', 'yes']), help="Whether saved detailed outputs to a log file.")
    # IPG solver configurations
    parser.add_argument("--exact_pg_computation", default=False, type=lambda x: (str(x).lower()
                        in ['true', '1', 'yes']), help="Mimic to the exact computation of the proximal operator.") 
    parser.add_argument("--exact_pg_computation_tol", type=float, default=1e-15, help="Deisred error tolerance.")                                                                      
    parser.add_argument("--ipg_strategy", type=str, default="diminishing", choices=["diminishing"], 
        help="Strategy to inexactly evaluate the proximal operator.\ndiminishing: c * np.log(k+1) / k**delta")
    parser.add_argument("--ipg_do_linesearch", default=True, type=lambda x: (str(x).lower()
                        in ['true', '1', 'yes']), help="Whether do linesearch in the projected gradient ascent.")
    parser.add_argument("--ipg_linesearch_eta", type=float, default=1e-4, help="eta of the linesearch.")
    parser.add_argument("--ipg_linesearch_xi", type=float, default=0.8, help="xi of the linesearch.")
    parser.add_argument("--ipg_linesearch_beta", type=float, default=1.2, help="beta of the linesearch.")
    parser.add_argument("--ipg_linesearch_limits", type=int, default=100, help="max attempts of the linesearch.")
    parser.add_argument("--ipg_diminishing_c", type=float, default=1, help="c of c * np.log(k+1) / k**delta")
    parser.add_argument("--ipg_diminishing_delta", type=float, default=2, help="delta of c * np.log(k+1) / k**delta")

    # ProxSVRG
    parser.add_argument('--proxsvrg_inner_repeat', default=1, type=int,
                        help='number of full pass over the data in one epoch')
    parser.add_argument("--proxsvrg_stepsize", type=str, default='const', choices=['diminishing', 'const'], help="strategy to adjust stepsize.")
    parser.add_argument("--proxsvrg_epoch_iterate", type=str, default='last',
                        choices=['last', 'average'], help="end of the epoch returns the averaged iterate as the major iterate")

    # ProxSAGA
    parser.add_argument("--proxsaga_stepsize", type=str, default='const', choices=['diminishing', 'const'], help="strategy to adjust stepsize.")

    # SPStorm
    parser.add_argument("--spstorm_stepsize", type=str, default='const', choices=['diminishing', 'const'], help="strategy to adjust stepsize.")
    parser.add_argument("--spstorm_betak", type=float, default=-1.0,
                        help="if spstorm_betak is a positive float, then the momentum parameter is set to spstorm_betak; \
                            if spstorm_betak is set to -1.0, then use the diminishing rule to update the momentum paramter.")
    parser.add_argument("--spstorm_interpolate", type=lambda x: (str(x).lower()
                                                                 in ['true', '1', 'yes']), default=False,
                        help="Take an interpolation step x_{k+1}=x_k+zeta*betak*(y_k-x_k).")
    parser.add_argument("--spstorm_zeta", type=float_or_str, default=1.0, help="Used when spstorm_interpolate is True. Can be a string or a float")

    # PStorm
    parser.add_argument("--pstorm_stepsize", type=str, default='const', choices=['diminishing', 'const'], help="strategy to adjust stepsize.")
    parser.add_argument("--pstorm_betak", type=float, default=-1.0,
                        help="if pstorm_betak is a positive float, then the momentum parameter is set to pstorm_betak; \
                            if pstorm_betak is set to -1.0, then use the diminishing rule to update the momentum paramter.")
    # RDA
    parser.add_argument("--rda_stepsize", type=str, default='increasing', choices=['increasing', 'const'], help="strategy to adjust stepsize.")
    parser.add_argument("--rda_stepconst", type=float, default=1.0, help="alphak = sqrt(k)/rda_stepconst")

    # ProxSG
    parser.add_argument("--proxsg_stepsize", type=str, default='diminishing', choices=['diminishing', 'const'], help="strategy to adjust stepsize.")
    parser.add_argument("--proxsg_stepconst", type=float, default=1.0, help="alphak = proxsg_stepconst / k")
    # Parse the arguments
    config = parser.parse_args()
    return config


if __name__ == "__main__":

    main(get_config())
