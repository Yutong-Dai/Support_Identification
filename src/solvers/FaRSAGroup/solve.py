import sys
import os
import time
import numpy as np
sys.path.append("../")
sys.path.append("../..")

import src.solvers.FaRSAGroup.utils as utils
from src.solvers.FaRSAGroup.params import *
from src.solvers.FaRSAGroup.Algorithm import AlgoBase, set_prox_stepsize
import src.solvers.FaRSAGroup.printUtils as printUtils


def solve(f, r, X_initial=None, proxStepsize=None, method='gradient',
          update_proxStepsize='const', params=params, print_group=True, print_second_level=False,
          kappa_1=1e-2, kappa_2=1e-2, fraction=0.8, print_time=False, cg_backtrack_strategy="cutfrac"):
    """
    Solve the group l2 regualrized problem.

    Args:
        f: loss function object
        r: regularizer object
        X_intial(np.array): starting point
        proxStepsize(np.float64/np.array/None): 
            If it is set to None, then one should call  `set_prox_stepsize` to initialize the proxStepsize
        method:  parameters for `set_prox_stepsize`.
        update_proxStepsize:
            'const': use constant params['zeta']
            'single': use only one proxStpesize for all groups; estimated by Lipschtiz constant
            'block': each group has its own proxStepsize; at each iteration, working groups are updated with the same Lipschtiz constant estimation
            'group': each group has its own proxStepsize; at each iteration, proxStepsize for each working groups are updated separately
            'groupv': varinat of group and only allows to decrease proxstepsize
        update_kappa:
            'chicg': update min(kappa, 1/2) as chicg suggests
            'hack': update kappa as hack suggests
        params:
            a dictionary containing parameters for linesearch methods.
        force_cg: force one to try newton step with the unit stepsize
    """
    # print out some key parameters
    if update_proxStepsize in ('single', 'const'):
        alpha_type = 'singlealpha'
    else:
        alpha_type = 'groupalpha'
    if proxStepsize is None:
        # proxStepsize = set_prox_stepsize(f, r, alpha_type, method)
        proxStepsize = np.min([set_prox_stepsize(f, r, alpha_type, method), 1])
        print('proxStepsize is [None]; Set up by {}'.format(method))
    else:
        print('proxStepsize is [{}]'.format(type(proxStepsize)))
    print("Update proxStepsize using {}".format(update_proxStepsize))
    print("Print time per iteration: {}".format(print_time))
    print("Termination tol:cg:{} | pg:{}".format(
        params['tol_cg'], params['tol_pg']))
    print("maxtime:{} | maxiter:{}".format(
        params['max_time'], params['max_iter']))
    print(
        f"kappa1_max:{params['kappa1_max']:3.3e} | kappa2_max:{params['kappa2_max']:3.3e}")
    print(
        f"kappa1_min:{params['kappa1_min']:3.3e} | kappa2_min:{params['kappa2_min']:3.3e}")
    print(
        f"kappa_increase:{params['kappa_increase']} | kappa_decrease:{params['kappa_decrease']}")
    print(f"kappaStrategy: dynamic | count freq:{params['kappa_freq_count']}")
    print(f"Equipped with early termination")
    # set up algorithms
    Ndata = f.n
    p = f.p
    K = r.K
    G_i_starts = r.starts
    G_i_ends = r.ends
    unique_groups = r.unique_groups
    problem_attribute = f.__str__()
    problem_attribute += "Regularizer:{:.>44}\n".format(r.__str__())
    problem_attribute += "Penalty Parameter:{:.>30}lambda={:3.4f}\n".format(
        '', r.penalty)
    problem_attribute += "Number of groups:{:.>32}\n".format(K)
    problem_attribute += "Update proxStepsize:{:.>35}\n".format(
        update_proxStepsize)
    algo = AlgoBase(f, r, proxStepsize, params, kappa_1, kappa_2)
    iteration_start = time.time()
    if X_initial is None:
        X = np.zeros([p, 1])
    else:
        X = X_initial
    fevals = gevals = HvProds = subitsT = 0
    info = {}
    if params['save_log']:
        outID = '{}'.format(params['tag'])
    else:
        outID = f.datasetName
    time_so_far = print_cost = 0
    if params['printlevel'] >= 1:
        utils.print_problem(problem_attribute, outID)
    # normX = utils.l2_norm(X)
    normX = np.sqrt(np.dot(X.T, X))[0][0]
    algo.fval = algo.f.evaluate_function_value(X)
    rval = algo.r.func(X)
    algo.F = algo.fval + rval
    fevals += 1
    gevals += 1
    iteration = 0
    time_update_proxStepszie = 0
    if Ndata < p:
        print('n<p, use fraction:{}'.format(fraction))
        F_seq_switch = [np.inf, np.inf]
        fraction_init = fraction
    else:
        print('n>=p, no fraction')
    prox_time = 0
    newton_time = 0
    pg_ls_time = 0
    cg_ls_time = 0
    pg_time = 0
    cg_time = 0
    consequtive_pg = 0
    if params["kappa_freq_count"]:
        kappa_increase, kappa_decrease = 0, 0
        kappa_1_max, kappa_1_min = kappa_1, kappa_1
        kappa_2_max, kappa_2_min = kappa_2, kappa_2

    while True:
        # print('=============================')
        # print("Iteration:{}".format(iteration))
        # if iteration == 9:
        #     print(1)
        prox_time_iter = time.time()
        algo.proximal_step(X)
        # print("||s||:{}".format(utils.l2_norm(algo.proximal)))
        prox_time_iter = time.time() - prox_time_iter
        prox_time += prox_time_iter
        # call set_cg first because the internal dependence issues.
        algo.set_cg()
        algo.set_pg()
        # print(f"kappa_1:{algo.kappa_1:2.3e} | kappa_2:{algo.kappa_2:2.3e}")
        gI_cg, nI_cg, gI_pg, nI_pg = len(algo.I_cg_group), np.sum(
            algo.I_cg_index), algo.K - len(algo.I_cg_group), np.sum(algo.I_pg_index)
        # if iteration in [95, 96]:  # debug madelon to remove
        #     # print("Iter:{} | old: {} | new: {}".format(iteration, chicg_last_iteration, algo.chi_cg))
        #     print(iteration)
        #     print(utils.get_classification(algo.zeroGroup, algo.nonzeroGroup, algo.zeroProxGroup, algo.nonzeroProxGroup))
        iteration_end = time.time() - iteration_start - print_cost
        time_so_far += iteration_end

        if params['printlevel'] == 2:
            if iteration % params['printevery'] == 0:
                # utils.print_header(outID, print_time)
                printUtils.print_header(outID, print_time)
            res = utils.get_classification(
                algo.zeroGroup, algo.nonZeroGroup, algo.zeroProxGroup, algo.nonZeroProxGroup)
            nn, nz, zn, zz = len(
                res['NZ-NZ']), len(res['NZ-Z']), len(res['Z-NZ']), len(res['Z-Z'])
            # utils.print_iteration(iteration, algo.fval, normX, algo.F, algo.proxStepsize, algo.chi_cg, algo.chi_pg,
            #                       gI_cg, nI_cg, gI_pg, nI_pg, nn, nz, zn, zz, outID)
            printUtils.print_iteration(iteration, algo.fval, normX, algo.F, algo.proxStepsize, algo.kappa_1, algo.chi_cg, algo.chi_pg,
                                       gI_cg, nI_cg, gI_pg, nI_pg, nn, nz, zn, zz, outID)
        if iteration == 0:
            chi_cg_0 = algo.chi_cg
            chi_pg_0 = algo.chi_pg
            chi_cg_termination = params['tol_cg'] * max(1, chi_cg_0)
            chi_pg_termination = params['tol_pg'] * max(1, chi_pg_0)
        if (algo.chi_cg <= chi_cg_termination) and (algo.chi_pg <= chi_pg_termination):
            info['status'] = 0
            break
        if iteration >= params['max_iter']:
            info['status'] = 1
            break
        if time_so_far > params['max_time']:
            info['status'] = 2
            break
        iteration_start = time.time()

        if algo.chi_pg <= params['Gamma'] * algo.chi_cg:
            # choose the working groups
            cg_time_iter = time.time()
            if Ndata < p:
                algo.select_cg_frac(fraction)
            else:
                algo.select_cg()
            y = algo.cg_step(X, cg_backtrack_strategy)
            cg_time_iter = time.time() - cg_time_iter
            cg_time += cg_time_iter
            cg_ls_time += algo.ls_time_iter
            newton_time += algo.newton_time_iter
            HvProds += algo.subits
            fevals += algo.cg_feval
            if algo.status in [-1, 4]:
                info['status'] = algo.status
                if algo.status == -1:
                    print("maxback cg:", algo.cg_backtrack)
                utils.print_cg_step(algo.typeofIteration, algo.nI_cgs, algo.gradF_Icgs_norm, algo.subprobFlag, algo.subits,
                                    algo.res, algo.res_target, algo.normd, algo.cg_type, algo.newZB, algo.dirder,
                                    algo.projection_attempts, algo.cg_backtrack, algo.cg_stepsize,
                                    prox_time_iter, algo.newton_time_iter, algo.ls_time_iter, cg_time_iter,
                                    outID, print_time)
                break
        else:
            pg_time_iter = time.time()
            algo.select_pg()
            y = algo.pg_step(X)
            pg_time_iter = time.time() - pg_time_iter
            pg_time += pg_time_iter
            pg_ls_time += algo.ls_time_iter
            fevals += algo.pg_feval
            if algo.pg_flag == False:
                if update_proxStepsize == 'const':
                    algo.proxStepsize *= algo.params['zeta']
            if algo.status == -1:
                info['status'] = algo.status
                print("maxback pg:", algo.pg_backtrack)
                utils.print_pg_step(algo.typeofIteration, algo.nI_pgs, algo.subits, algo.normd,
                                    algo.pg_backtrack, algo.pg_stepsize,
                                    prox_time_iter, algo.ls_time_iter, pg_time_iter,
                                    outID, print_time)
                break
        if update_proxStepsize != 'const':
            time_update_proxStepszie_begin = time.time()
            if algo.typeofIteration == 'cg':
                algo.get_proxStepsize(
                    X, y, algo.f_old, algo.fval, algo.d_use, algo.stepSize_use)
            else:
                algo.get_proxStepsize(
                    X, y, algo.f_old, algo.fval, algo.d_use, algo.stepSize_use)
            algo.proxStepsize = algo.newProxStepsize
            time_update_proxStepszie_end = time.time()
            time_update_proxStepszie += time_update_proxStepszie_end - \
                time_update_proxStepszie_begin

        subitsT += algo.subits
        temp = time.time()
        if params['printlevel'] == 2:
            if algo.typeofIteration == 'cg':
                utils.print_cg_step(algo.typeofIteration, algo.nI_cgs, algo.gradF_Icgs_norm, algo.subprobFlag, algo.subits,
                                    algo.res, algo.res_target, algo.normd, algo.cg_type, algo.newZB, algo.dirder,
                                    algo.projection_attempts, algo.cg_backtrack, algo.cg_stepsize,
                                    prox_time_iter, algo.newton_time_iter, algo.ls_time_iter, cg_time_iter,
                                    outID, print_time)
            else:
                utils.print_pg_step(algo.typeofIteration, algo.nI_pgs, algo.subits, algo.normd,
                                    algo.pg_backtrack, algo.pg_stepsize,
                                    prox_time_iter, algo.ls_time_iter, pg_time_iter,
                                    outID, print_time)
            if print_second_level:
                if len(algo.bar_I_cg_group) != 0:
                    utils.print_more(algo.bar_I_cg_index, algo.bar_I_cg_group, algo.bar_chi_cg, algo.I_cg_index, algo.chi_cg,
                                     algo.norm_gradF_bar_I_cg, algo.group_X_norm, algo.group_gradF_norm,
                                     algo.small_radius_lst, algo.outter_radius_lst, algo.inner_radius_lst,
                                     algo.kappa_1, algo.kappa_2, algo.kappa_3, outID)
                else:
                    utils.print_empty_bar_Icg(outID)
        print_cost = time.time() - temp
        X = y
        # this is need if update_proxStepsize is not const or single
        # algo.fval = algo.f.evaluate_function_value(X)
        # algo.F = algo.fval + algo.r.evaluate_function_value(X)
        # normX = utils.l2_norm(X)
        normX = np.sqrt(np.dot(X.T, X))[0][0]
        gevals += 1  # about to call gradient method in the next iteration
        iteration += 1
        if Ndata < p:
            if (iteration % 5 == 1) and (algo.typeofIteration == 'cg'):
                F_seq_switch.pop(0)
                F_seq_switch.append(algo.F)
                if np.abs(F_seq_switch[0] - F_seq_switch[1]) <= 1e-3:
                    fraction = 1
                else:
                    fraction = fraction_init
        if algo.typeofIteration == 'cg':
            consequtive_pg = 0
            # 6 - 1 = 5 total backtracks
            if algo.cg_backtrack + algo.projection_attempts > 6:
                # default: 1e3 make 10 also controlable # factor to increase and decrease (10, 1/10)
                algo.kappa_1 = min(
                    algo.kappa_1 * params['kappa_increase'], params['kappa1_max'])
                algo.kappa_2 = min(
                    algo.kappa_2 * params['kappa_increase'], params['kappa2_max'])
                if params["kappa_freq_count"]:
                    kappa_increase += 1
                    kappa_1_max = max(algo.kappa_1, kappa_1_max)
                    kappa_2_max = max(algo.kappa_2, kappa_2_max)
        else:
            consequtive_pg += 1
            if consequtive_pg > 5:
                # default: 1e-5
                algo.kappa_1 = max(
                    algo.kappa_1 * params['kappa_decrease'], params['kappa1_min'])
                algo.kappa_2 = max(
                    algo.kappa_2 * params['kappa_decrease'], params['kappa2_min'])
                if params["kappa_freq_count"]:
                    kappa_decrease += 1
                    kappa_1_min = min(kappa_1_min, algo.kappa_1)
                    kappa_2_min = min(kappa_2_min, algo.kappa_2)

    if params['printlevel'] == 2:
        utils.print_exit(info['status'], outID)
    nnz = utils.get_group_structure(
        X, K, unique_groups, G_i_starts, G_i_ends, epsilon=1e-8)
    info['n'] = algo.f.n
    info['p'] = algo.f.p
    info['Lambda'] = algo.r.penalty
    info['num_groups'] = algo.r.K
    info['nnz'] = nnz
    info['nz'] = algo.K - nnz
    info['F'] = algo.F
    info['normX'] = normX
    info['f'] = algo.fval
    info['chipg'] = algo.chi_pg
    info['chicg'] = algo.chi_cg
    info['fevals'] = fevals
    info['HvProds'] = HvProds
    info['time'] = time_so_far
    info['iteration'] = iteration
    info['num_pg_steps'] = algo.num_pg_steps
    info['num_cg0_stpes'] = algo.num_cg0_steps
    info['num_cgdesc_steps'] = algo.num_cgdesc_steps
    info['gevals'] = gevals
    info['subits'] = subitsT
    info['time_update_stepsize'] = time_update_proxStepszie
    info['X'] = X
    info['zeroGroup'] = algo.zeroGroup
    info['nonZeroGroup'] = algo.nonZeroGroup
    info['proxStepsize'] = algo.proxStepsize
    info['cg_time'] = cg_time
    info['newton_time'] = newton_time
    info['cg_ls_time'] = cg_ls_time
    info['pg_time'] = pg_time
    info['pg_ls_time'] = pg_ls_time
    info['prox_time'] = prox_time
    if params["kappa_freq_count"]:
        info['kappa_increase'] = kappa_increase
        info['kappa_decrease'] = kappa_decrease
        info['kappa1_max'] = kappa_1_max
        info['kappa2_max'] = kappa_2_max
        info['kappa1_min'] = kappa_1_min
        info['kappa2_min'] = kappa_2_min
    if params['printlevel'] == 2 and info['status'] != -1 and print_group:
        utils.print_group_sparsity(
            X, K, unique_groups, G_i_starts, G_i_ends, outID, epsilon=1e-8)
    if params['printlevel'] == 2:
        utils.print_result(info, outID)
    return info
