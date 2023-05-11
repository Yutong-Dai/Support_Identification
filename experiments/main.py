import argparse
from scipy.io import savemat, loadmat
import numpy as np
import os
import sys
sys.path.append("../")
import src.utils as utils
from src.funcs.lossfunction import LogisticLoss, LeastSquares
from src.funcs.regularizer import GL1, NatOG, TreeOG
from src.solvers.SPStorm import SPStorm
from src.solvers.PStorm import PStorm
from src.solvers.ProxSVRG import ProxSVRG
from src.solvers.ProxSAGA import ProxSAGA
from src.solvers.RDA import RDA
from src.solvers.FaRSAGroup.params import params as farsa_config
from src.solvers.FaRSAGroup.solve import solve as faras_solve
from src.solvers.ProxGD import ProxGD
from copy import deepcopy
from scipy.sparse import csr_matrix
import time
import random

def main(config):
    # Setup the problem
    try:
        fileType = utils.fileTypeDict[config.dataset]
    except KeyError:
        fileType = 'npy'
    utils.mkdirs(f'./{config.purpose}/{config.solver}_{config.loss}_{config.regularizer}')
    config.data_dir = os.path.expanduser(config.data_dir)
    tag = f'./{config.purpose}/{config.solver}_{config.loss}_{config.regularizer}/{config.dataset}_lam_shrink:{config.lam_shrink}'
    if config.regularizer == 'GL1':
        tag += f"_frac:{config.frac}"
    elif config.regularizer == 'NatOG':
        if config.overlap_task == 'chain':
            tag += f"_{config.overlap_task}_{config.chain_grpsize}_{config.chain_overlap_ratio}_{config.ipg_strategy}"
        elif config.overlap_task == 'btree':
            tag += f"_{config.overlap_task}_{config.btree_lammax}_{config.btree_depth}_{config.btree_manual_weights}"
            if config.btree_manual_weights:
                tag += f"_{config.btree_remove}_{config.btree_manual_penalty}"
            tag += f"_{config.ipg_strategy}"
        else:
            raise ValueError("Unknown overlap task.")
    elif config.regularizer == 'TreeOG':
        tag += f"_{config.overlap_task}_{config.btree_lammax}_{config.btree_depth}_{config.btree_manual_weights}_{config.btree_remove}_{config.btree_manual_penalty}"        
    else:
        raise ValueError("Unknown regularizer type.")

    # if utils.is_finished(tag + '_stats.npy'):
    #     exit()

    print("Working on: {}...".format(config.dataset))
    X, y = utils.set_up_xy(config.dataset, fileType, dbDir=config.data_dir)
    if config.loss == 'logit':
        f = LogisticLoss(X, y, config.dataset, config.weight_decay)
    elif config.loss == 'ls':
        f = LeastSquares(X, y, config.dataset, config.weight_decay)
    else:
        raise ValueError("Unknown loss function.")
    
    # Setup the regularizer
    p = X.shape[1]
    weights = None
    if config.regularizer == 'GL1':
        num_of_groups = max(int(p * config.frac), 1)
        group = utils.gen_group(p, num_of_groups)
    elif config.regularizer in ['NatOG', 'TreeOG']:
        if config.overlap_task == 'chain':
            group = utils.gen_chain_group(dim=p, grp_size=config.chain_grpsize, overlap_ratio=config.chain_overlap_ratio)
        elif config.overlap_task == 'btree':
            pbak, nodes_list, nodes_relation_dict = utils.construct_complete_tree(config.btree_depth)
            assert p == pbak
            if config.btree_manual_weights:
                weights = np.ones(p)
                random.seed(config.btree_remove)
                nodes_idx = random.sample(range(1,p-1), k=int(p * config.btree_remove))
                dependent_group_lst = [utils.get_dependent_group(idx, nodes_relation_dict) for idx in nodes_idx]
                removed_groups = []
                for g in dependent_group_lst:
                    removed_groups += g
                removed_groups = list(set(removed_groups))
                print("remove num nodes:", len(nodes_idx), "remove num groups:", len(removed_groups))
                for i in nodes_idx:
                    if config.btree_manual_penalty != -1.0:
                        weights[i] = config.btree_manual_penalty
                    else:
                        weights[i] = utils.tree_manual_penalty_dict[config.btree_depth][config.btree_remove]
            group, tree, dot = utils.gen_tree_group(nodes_list, nodes_relation_dict, penalty=config.btree_lammax * config.lam_shrink, weights=weights)
            # if config.btree_manual_weights:
            #     # sqrt(|g|)
            #     weights = tree['eta_g'] / config.btree_lammax * config.lam_shrink
            #     random.seed(config.btree_remove)
            #     nodes_idx = random.sample(range(1,p-1), k=int(p * config.btree_remove))
            #     dependent_group_lst = [utils.get_dependent_group(idx, nodes_relation_dict) for idx in nodes_idx]
            #     removed_groups = []
            #     for g in dependent_group_lst:
            #         removed_groups += g
            #     removed_groups = list(set(removed_groups))
            #     print("remove num nodes:", len(nodes_idx), "remove num groups:", len(removed_groups))
            #     for i in nodes_idx:
            #         weights[i] *= config.btree_manual_penalty
            #     tree['eta_g'] = weights * config.btree_lammax * config.lam_shrink

    else:
        raise ValueError("Unknown regularizer type.")

    Lip_path = f'{config.data_dir}/Lip/Lip-{config.dataset}.mat'
    if os.path.exists(Lip_path):
        L = loadmat(Lip_path)["L"][0][0]
        print(f"loading Lipschitz constant from: {Lip_path}")
    else:
        print(f"Compute Lip ...")
        L = utils.estimate_lipschitz(X, config.loss)
        savemat(Lip_path, {"L": L})
        print(f"save Lipschitz constant to: {Lip_path}")

    if config.regularizer == 'GL1':
        lammax_path = f'{config.data_dir}/lammax/lammax-{config.dataset}-{config.frac}.mat'
    elif config.regularizer == 'NatOG':
        if config.overlap_task == 'chain':
            lammax_path = f'{config.data_dir}/lammax/lammax-{config.dataset}-chain_grpsize:{config.chain_grpsize}_chain_overlap_ratio:{config.chain_overlap_ratio}.mat'
        else:
            lammax_path = None
    elif config.regularizer == 'TreeOG':
        lammax_path = None
    else:
        raise ValueError("Unknown regularizer type.")
    
    if lammax_path is not None:
        if os.path.exists(lammax_path):
            lammax = loadmat(lammax_path)["lammax"][0][0]
            print(f"loading lammax from: {lammax_path}")
        else:
            print(f"Compute lammax ...")
            lammax = utils.lam_max(X, y, group, config.loss)
            savemat(lammax_path, {"lammax": lammax})
            print(f"save lammax to: {lammax_path}")
    else:
        if config.overlap_task == 'btree':
            lammax = config.btree_lammax
        else:
            raise ValueError("Unknown overlap task.")
        
    Lambda = lammax * config.lam_shrink
    start = time.time()
    if config.regularizer == 'GL1':
        r = GL1(penalty=Lambda, groups=group)
    elif config.regularizer == 'NatOG':
        r = NatOG(groups=group, penalty=Lambda, config=config, weights=weights)
    elif config.regularizer == 'TreeOG':
        if config.solver == 'ProxGD':
            r = TreeOG(group, tree, Lambda, weights=weights)
        else:
            raise ValueError(f"Solver{config.solver} is not supported.")
    else:
        raise ValueError("Unknown regularizer type.")

    print(f"Spend {time.time()-start:.1f} seconds to initialize r.", flush=True)

    if config.solver == 'FaRSAGroup':
        if utils.is_finished(tag + '_stats.npy'):
            exit()
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
    elif config.solver == 'ProxGD':
        tag += f'_proxgd_stepsize:{config.proxgd_stepsize}'
        if utils.is_finished(tag + '_stats.npy'):
            exit()        
        config.tag = tag
        solver = ProxGD(f, r, config)
        info = solver.solve(x_init=None, alpha_init=1.0)
        np.save(f"{tag}_stats.npy", info)
    else:
        tag_base = deepcopy(tag)
        for run in range(config.runs):
            tag = deepcopy(tag_base)
            print(f"Process: {run+1} / {config.runs}")
            actual_seed = run + config.seed
            tag += f"_seed:{actual_seed}"

            utils.setup_seed(actual_seed)
            

            if config.solver == 'ProxSVRG':
                alpha_init = config.proxsvrg_lipcoef / L
                tag += f'_proxsvrg_inner_repeat:{config.proxsvrg_inner_repeat}_proxsvrg_lipcoef:{config.proxsvrg_lipcoef}'
                if utils.is_finished(tag + '_stats.npy'):
                    exit()                
                config.tag = tag
                solver = ProxSVRG(f, r, config)
                info = solver.solve(x_init=None, alpha_init=alpha_init)

            elif config.solver == 'ProxSAGA':
                alpha_init = config.proxsaga_lipcoef / L
                tag += f'_proxsaga_lipcoef:{config.proxsaga_lipcoef}'
                if utils.is_finished(tag + '_stats.npy'):
                    exit()  
                config.tag = tag
                solver = ProxSAGA(f, r, config)
                info = solver.solve(x_init=None, alpha_init=alpha_init)

            elif config.solver == 'SPStorm':
                alpha_init = config.spstorm_lipcoef / L
                tag += f'_spstorm_betak:{config.spstorm_betak}'
                tag += f'_spstorm_zeta:{config.spstorm_zeta}'
                tag += f'_spstorm_lipcoef:{config.spstorm_lipcoef}'
                if utils.is_finished(tag + '_stats.npy'):
                    exit() 
                config.tag = tag
                solver = SPStorm(f, r, config)
                info = solver.solve(x_init=None, alpha_init=alpha_init, Lg=L)

            elif config.solver == 'PStorm':
                alpha_init = config.pstorm_lipcoef / L
                tag += f'_pstorm_stepsize:{config.pstorm_stepsize}'
                tag += f'_pstorm_betak:{config.pstorm_betak}'
                tag += f'_pstorm_lipcoef:{config.pstorm_lipcoef}'
                if utils.is_finished(tag + '_stats.npy'):
                    exit() 
                config.tag = tag
                solver = PStorm(f, r, config)
                info = solver.solve(x_init=None, alpha_init=alpha_init, Lg=L)

            elif config.solver == 'RDA':
                tag += f'_rda_stepconst:{config.rda_stepconst}'
                if utils.is_finished(tag + '_stats.npy'):
                    exit() 
                config.tag = tag
                solver = RDA(f, r, config)
                info = solver.solve(x_init=None, alpha_init=None, stepconst=config.rda_stepconst)

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
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="add a quadratic penalty on the loss function to make it stronglt convex; set to 0 to disable.")
    parser.add_argument("--regularizer", type=str, default='GL1', choices=["GL1", "NatOG", "TreeOG"], help='Name of the loss funciton.')
    parser.add_argument("--lam_shrink", type=float, default=0.1, help="The lambda used is calculated as lambda=lambda_max * lam_shrink")
    parser.add_argument("--frac", type=float, default=0.3, help="num_of_groups = max(int(p * config.frac), 1)")
    parser.add_argument("--overlap_task", type=str, default="btree", choices=["btree", "chain", "dgraph"], help="The overlap task for NatOG.")
    parser.add_argument("--chain_grpsize", type=int, default=10, help="number of variables in each group for NatOG")
    parser.add_argument("--chain_overlap_ratio", type=float, default=0.1, help="overlap ratio between groups for NatOG. Between 0 and 1.")
    parser.add_argument("--btree_depth", type=int, default=11, choices=[11,12,13,14,15], help="depth of the binary tree.")
    parser.add_argument("--btree_manual_weights",  type=lambda x: (str(x).lower()
                        in ['true', '1', 'yes']), default=True, help="manual weights for the binary tree.")
    parser.add_argument("--btree_remove", type=float, default=0.01, help="percent of nodes to remove in binary tree.")  
    parser.add_argument("--btree_manual_penalty", type=float, default=-1.0, 
                    help="manual penalty for the binary tree. Set -1.0 as default so I can use the loop uptable for my test cases.")                      
    parser.add_argument("--btree_lammax", type=float, default=1.0, help="max penalty lambda for the binary tree.")

    # solver shared arguments
    parser.add_argument("--solver", type=str, choices=["FaRSAGroup", "ProxGD", "SPStorm", "PStorm",
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
    parser.add_argument("--save_seq", default=True, type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                             help="Whether save intermediate function value, iterate sparsity, and gradient errors.")
    parser.add_argument("--save_xseq", default=True, type=lambda x: (str(x).lower()
                        in ['true', '1', 'yes']), help="Whether save intermediate iterates sequence.")
    parser.add_argument('--seed', default=2023, type=int, help='Global random seed.')
    parser.add_argument('--runs', default=1, type=int, help='(For stochastic algorithms) Total numbers of repeated runs.')
    parser.add_argument("--compute_optim", default=True, type=lambda x: (str(x).lower()
                        in ['true', '1', 'yes']), help="Whether compute the optimality measure.")
    parser.add_argument("--optim_scaled", default=False, type=lambda x: (str(x).lower()
                        in ['true', '1', 'yes']), help="Whether scale the optimality measure by the stepsize.")
    parser.add_argument("--save_log", default=True, type=lambda x: (str(x).lower()
                        in ['true', '1', 'yes']), help="Whether saved detailed outputs to a log file.")
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
    parser.add_argument("--ipg_strategy", type=str, default="diminishing", choices=["diminishing","linear_decay"], 
        help="Strategy to inexactly evaluate the proximal operator.\ndiminishing: c * np.log(k+1) / k**delta\nlinear_decay is only for proxsvrg and saga")    
    parser.add_argument("--ipg_diminishing_c", type=float, default=1, help="c of c * np.log(k+1) / k**delta")
    parser.add_argument("--ipg_diminishing_delta", type=float, default=2, help="delta of c * np.log(k+1) / k**delta")
    parser.add_argument("--ipg_linear_decay_const", type=float, default=0.99, help="const of epsilontilde_k = const * epsilontilde_{k-1}. const in (0,1)")

    # ProxSVRG
    parser.add_argument('--proxsvrg_inner_repeat', default=1, type=int,
                        help='number of full pass over the data in one epoch')
    parser.add_argument("--proxsvrg_stepsize", type=str, default='const', choices=['diminishing', 'const'], help="strategy to adjust stepsize.")
    parser.add_argument("--proxsvrg_epoch_iterate", type=str, default='last',
                        choices=['last', 'average'], help="end of the epoch returns the averaged iterate as the major iterate")
    parser.add_argument("--proxsvrg_lipcoef", type=float, default=0.1, help="lipcoef/L")

    # ProxSAGA
    parser.add_argument("--proxsaga_stepsize", type=str, default='const', choices=['diminishing', 'const'], help="strategy to adjust stepsize.")
    parser.add_argument("--proxsaga_lipcoef", type=float, default=0.1, help="lipcoef/L")
    # SPStorm
    parser.add_argument("--spstorm_stepsize", type=str, default='const', choices=['diminishing', 'const'], help="strategy to adjust stepsize.")
    parser.add_argument("--spstorm_betak", type=float, default=-1.0,
                        help="if spstorm_betak is a positive float, then the momentum parameter is set to spstorm_betak; \
                            if spstorm_betak is set to -1.0, then use the diminishing rule to update the momentum paramter.")
    parser.add_argument("--spstorm_zeta", type=float_or_str, default='dynanmic', help="Can be a string or a float")
    parser.add_argument("--spstorm_lipcoef", type=float, default=0.1, help="lipcoef/L")
    # PStorm
    parser.add_argument("--pstorm_stepsize", type=str, default='diminishing', choices=['diminishing', 'const'], help="strategy to adjust stepsize.")
    parser.add_argument("--pstorm_betak", type=float, default=-1.0,
                        help="if pstorm_betak is a positive float, then the momentum parameter is set to pstorm_betak; \
                            if pstorm_betak is set to -1.0, then use the diminishing rule to update the momentum paramter.")
    parser.add_argument("--pstorm_lipcoef", type=float, default=0.1, help="lipcoef/L")                            
    # RDA
    parser.add_argument("--rda_stepsize", type=str, default='increasing', choices=['increasing', 'const'], help="strategy to adjust stepsize.")
    parser.add_argument("--rda_stepconst", type=float, default=1.0, help="alphak = sqrt(k)/rda_stepconst")


    # ProxGD
    parser.add_argument("--proxgd_method", type=str, default='ISTA', choices=['ISTA', 'FISTA'], help="Proximal gradient method.")
    parser.add_argument("--proxgd_stepsize", type=str, default='linesearch', choices=['linesearch', 'const'], help="strategy to adjust stepsize.")
    # global linesearch parameters
    parser.add_argument("--linesearch_eta", type=float, default=1e-4, help="eta of the linesearch.")
    parser.add_argument("--linesearch_xi", type=float, default=0.8, help="xi of the linesearch.")
    parser.add_argument("--linesearch_beta", type=float, default=1.2, help="beta of the linesearch.")
    parser.add_argument("--linesearch_limits", type=int, default=100, help="max attempts of the linesearch.")

    # Parse the arguments
    config = parser.parse_args()
    return config


if __name__ == "__main__":

    main(get_config())
