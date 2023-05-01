import numpy as np
from scipy.io import loadmat
from sklearn.datasets import load_svmlight_file
from scipy.sparse import issparse
from numba import jit


def l2_norm(x):
    return np.sqrt(np.dot(x.T, x))[0][0]


def get_first_intersetion(a, b, c):
    delta = b ** 2 - 4 * a * c
    if delta <= 0:
        tau = np.inf  # turn off the projection
    else:
        temp = np.sqrt(delta)
        root1 = (-b + temp) / (2 * a)
        root2 = (-b - temp) / (2 * a)
        tau = min(root1, root2)
        if tau < 0:
            tau = np.inf
    return tau


# def load_data(matfilepath):
#     data_dict = loadmat(matfilepath)
#     try:
#         return (data_dict['X'], data_dict['y'])
#     except KeyError:
#         print("Wrong matlab data file... I cannot find X and y.")


def set_up_xy(datasetName, fileType='txt', dbDir='../db', to_dense=False):
    filepath = dbDir + "/{}.{}".format(datasetName, fileType)
    if fileType != 'mat':
        data = load_svmlight_file(filepath)
        X, y = data[0], data[1].reshape(-1, 1)
        # if datasetName in ['gisette']:
        #     to_dense = True
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


def intersection(lst1, lst2):
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def lam_max(X, y, group, loss='logit'):
    """
    Reference: Yi Yang and Hui Zou. A fast unified algorithm for solving group-lasso penalize learning problems. Page 22.
    """
    # beta_1 = np.log(np.sum(y == 1) / np.sum(y == -1))
    beta_1 = np.zeros((X.shape[1], 1))
    lam_max = -1
    unique_groups, group_frequency = np.unique(group, return_counts=True)
    K = len(unique_groups)
    if loss == 'logit':
        ys = y / (1 + np.exp(y * (X@beta_1)))
        nabla_L = X.T@ys / X.shape[0]
    elif loss == 'ls':
        ys = y - X@beta_1
        nabla_L = (ys.T@X).T / X.shape[0]
    else:
        raise ValueError("Invalid loss!")
    for i in range(K):
        sub_grp = nabla_L[group == (i + 1)]
        temp = l2_norm(sub_grp) / np.sqrt(group_frequency[i])
        if temp > lam_max:
            lam_max = temp
    return lam_max

# def lam_max(X, y, group):
#     """
#     Reference: Yi Yang and Hui Zou. A fast unified algorithm for solving group-lasso penalize learning problems. Page 22.
#     """
#     # beta_1 = np.log(np.sum(y == 1) / np.sum(y == -1))
#     beta_1 = 0
#     ys = y / (1 + np.exp(beta_1 * y))
#     nabla_L = X.T@ys / X.shape[0]
#     # print(beta_1, norm(ys, 2)**2, nabla_L.T)
#     lam_max = -1
#     unique_groups, group_frequency = np.unique(group, return_counts=True)
#     K = len(unique_groups)
#     for i in range(K):
#         sub_grp = nabla_L[group == (i + 1)]
#         temp = l2_norm(sub_grp) / np.sqrt(group_frequency[i])
#         if temp > lam_max:
#             lam_max = temp
#     return lam_max


def lam_max_jit_prep(X, y, group):
    print("  I am in lam_max_jit_prep", flush=True)
    beta_1 = float(0)
    ys = y / (1 + np.exp(beta_1 * y))
    print('  Compute nabla_L')
    nabla_L = (ys.T@X).T / X.shape[0]
    if issparse(nabla_L):
        nabla_L = nabla_L.toarray()
        print('  nabla_L to dense...', flush=True)
    print('  Compute unique', flush=True)
    unique_groups, group_frequency = np.unique(group, return_counts=True)
    return unique_groups, group_frequency, nabla_L


@jit(nopython=True, cache=True)
def lam_max_jit(group, unique_groups, group_frequency, nabla_L):
    K = len(unique_groups)
    lam_max = -1
    print('  Going to enter the loop...')
    for i in range(K):
        sub_grp = nabla_L[group == (i + 1)]
        temp = np.sqrt(np.dot(sub_grp.T, sub_grp))[0][0] / np.sqrt(group_frequency[i])
        if temp > lam_max:
            lam_max = temp
        if i % 100000 == 0:
            print("Process")
            print(i)
    return lam_max


def get_classification(zeroGroup, nonZeroGroup, zeroProxGroup, nonZeroProxGroup):
    res = {}
    res['Z-Z'] = intersection(zeroGroup, zeroProxGroup)
    res['Z-NZ'] = intersection(zeroGroup, nonZeroProxGroup)
    res['NZ-Z'] = intersection(nonZeroGroup, zeroProxGroup)
    res['NZ-NZ'] = intersection(nonZeroGroup, nonZeroProxGroup)
    return res


def gen_group(p, K):
    dtype = type(K)
    if dtype == int:
        group = K * np.ones(p)
        size = int(np.floor(p / K))
        for i in range(K):
            start_ = i * size
            end_ = start_ + size
            group[start_:end_] = i + 1
    elif dtype == np.ndarray:
        portion = K
        group = np.ones(p)
        chunk_size = p * portion
        start_ = 0
        for i in range(len(portion)):
            end_ = start_ + int(chunk_size[i])
            group[start_:end_] = i + 1
            start_ = int(chunk_size[i]) + start_
    return group


def get_group_structure(X, K, unique_groups, G_i_starts, G_i_ends, epsilon=1e-8):
    nz = 0
    for i in range(K):
        start, end = G_i_starts[i], G_i_ends[i]
        X_Gi = X[start:end]
        if (np.sum(np.abs(X_Gi)) == 0):
            nz += 1
    nnz = K - nz
    return nnz


def get_partition(I_cg, I_pg, nonzeroGroup, nonzeroProxGroup, K):
    temp = intersection(nonzeroGroup, nonzeroProxGroup)
    gI_cg = len(temp)
    nI_cg = np.sum(I_cg)
    gI_pg = K - gI_cg
    nI_pg = np.sum(I_pg)
    return gI_cg, nI_cg, gI_pg, nI_pg


def print_problem(problem_attribute, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    dashedline = '+' + '-' * 198 + '+\n'
    with open(filename, "a") as logfile:
        logfile.write("Problem Summary\n")
        logfile.write("=" * 30 + '\n')
        logfile.write(problem_attribute)
        logfile.write(dashedline)


def print_header(outID=None, print_time=False):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    column_titles = ' {Iter:^5s} {f:^8s} {x:^8s} {F:^7s}    | {alpha_min:^6s} {alpha_max:^6s} {chicg:^8s} {chipg:^8s} {g_cg:^5s} {nI_cg:^5s} {g_pg:^5s} {nI_pg:^5s} {struct}| {itype:>2s}{nS:>5s}  {gradF:^8s} {flag:>4s} {its:>4s} {residual:^8s}  {target:^8s} {d:^7s} | {ctype:^4s} #newZB  {dirder:^8s}  {bak:^4s}  {s:^8s}|'.format(
        Iter='Iter', f='f', x='|x|', F='F', alpha_min='alphaMin', alpha_max='alphaMax', chicg='chi_cg', chipg='chi_pg', g_cg='#B_cg',
        nI_cg='|I_cg|', g_pg='#B_pg', nI_pg='|I_pg|', struct=' n-n  n-z  z-n  z-z ',
        itype='type', nS='nVar', gradF='|gradF|', flag='flag', its='its', residual=' Res ', target='tarRes', d='|d|',
        ctype='type', dirder='dirder', bak='bak', s='stepsize')
    if print_time:
        column_titles += '    prox       Newton      LS          PG         CG   |\n'
    else:
        column_titles += '\n'
    with open(filename, "a") as logfile:
        logfile.write(column_titles)


def print_iteration(iteration, fval, normX, F, alpha, chi_cg,
                    chi_pg, gI_cg, nI_cg, gI_pg, nI_pg, nn, nz, zn, zz, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'

    if type(alpha) == np.float64:
        alpha_min = alpha_max = alpha
    else:
        alpha_max = max(alpha)
        alpha_min = min(alpha)
    contents = "{it:5d} {fval:8.2e} {normX:8.2e} {F:8.5e} | {alpha_min:8.2e} {alpha_max:8.2e} {chi_cg:8.2e} {chi_pg:8.2e} {I1:>5d} {nI1:>5d}  {I2:>5d} {nI2:>5d} {nn:>5d}{nz:>5d}{zn:>5d}{zz:>5d} |".format(it=iteration, fval=fval, normX=normX, F=F,
                                                                                                                                                                                                            alpha_min=alpha_min,
                                                                                                                                                                                                            alpha_max=alpha_max,
                                                                                                                                                                                                            chi_cg=chi_cg, chi_pg=chi_pg,
                                                                                                                                                                                                            I1=gI_cg, nI1=nI_cg,
                                                                                                                                                                                                            I2=gI_pg, nI2=nI_pg,
                                                                                                                                                                                                            nn=nn, nz=nz, zn=zn, zz=zz)
    with open(filename, "a") as logfile:
        logfile.write(contents)


def print_cg_step(typeOfIteration, nS, gradF_norm, subprobFlag, subits, res, res_target, normd, cg_type, newzb, dirder, proj, j, stepsize,
                  prox_time=None, newton_time=None, ls_time=None, cg_time=None, outID=None, print_time=False):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = " {ctype:>3s} {nS:>5d} {gradF_norm:8.2e} {subpb:5s} {subits:>4d} {res:8.2e} {res_target:8.2e} {normd:8.2e} | {itype:4s}  {newzb:>4d}  {dirder: 8.2e} {proj:>2d}/{j:>2d}  {s:8.2e}| ".format(
        ctype=typeOfIteration, nS=nS, gradF_norm=gradF_norm, subpb=subprobFlag, subits=subits, res=res, res_target=res_target, normd=normd,
        itype=cg_type, newzb=newzb, dirder=dirder, proj=proj, j=j, s=stepsize)
    if print_time:
        contents += " {:3.2e}    {:3.2e}   {:3.2e}   --------   {:3.2e}|\n".format(prox_time, newton_time, ls_time, cg_time)
    else:
        contents += "\n"
    with open(filename, "a") as logfile:
        logfile.write(contents)


def print_pg_step(typeOfIteration, nS, subits, normd, j, stepsize,
                  prox_time=None, ls_time=None, pg_time=None, outID=None, print_time=False):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = " {ctype:>3s} {nS:>5d} -------- ----- {subits:>4d} -------- -------- {normd:8.2e} | desc  ----- ---------   /{j:>2d}  {s:8.2e}| ".format(
        ctype=typeOfIteration, nS=nS, subits=subits, normd=normd, j=j, s=stepsize)
    if print_time:
        contents += " {:3.2e}    --------   {:3.2e}   {:3.2e}   --------|\n".format(prox_time, ls_time, pg_time)
    else:
        contents += "\n"
    with open(filename, "a") as logfile:
        logfile.write(contents)


def print_more(bar_I_cg_index, bar_I_cg_group, bar_chicg, I_cg_index, chicg, norm_gradF_bar_I_cg,
               group_X_norm, group_gradF_norm, small_radius_lst, outter_radius_lst, inner_radius_lst,
               kappa1, kappa2, kappa3, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = '=' * 80 + '\n'
    contents += " kappa1      : {:8.2e} | kappa2 : {:8.2e} |    kappa3       : {:8.2e}\n".format(kappa1, kappa2, kappa3)
    contents += " bar_chicg   : {:8.2e} | chicg  : {:8.2e} | |gradF_bar_I_cg|: {}\n".format(bar_chicg, chicg, norm_gradF_bar_I_cg)
    contents += " |bar_I_cg| : {:4d} | |I_cg| : {:4d}\n".format(np.sum(bar_I_cg_index), np.sum(I_cg_index))
    contents += ' Group      |X|       |gradF|    R_small    R_outter    R_inner  |X|>R_outter\n'
    for i in bar_I_cg_group:
        norm_x = group_X_norm[i]
        norm_gradF = group_gradF_norm[i]
        small_radius = small_radius_lst[i]
        if (outter_radius_lst is not None) and (inner_radius_lst is not None):
            outter_radius = outter_radius_lst[i]
            inner_radius = inner_radius_lst[i]
        else:
            inner_radius = -1
            outter_radius = -1
        contents += " {G:>4s}    {norm_x:3.3e}   {norm_gradF:3.3e}  {small_radius:3.3e}  {outter_radius:3.3e}  {inner_radius:3.3e}     {outside}\n".format(
            G=str(i), norm_x=norm_x, norm_gradF=norm_gradF, small_radius=small_radius, outter_radius=outter_radius, inner_radius=inner_radius, outside=norm_x >= outter_radius)
    contents += '=' * 80 + '\n'
    with open(filename, "a") as logfile:
        logfile.write(contents)


def print_empty_bar_Icg(outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = '=' * 80 + '\n'
    contents += 'bar_I_cg is empty as a result of \n 1. {X !=0 and (X+S)!=0} is empty \n or \n 2. kappa1 is too big!\n'
    contents += '=' * 80 + '\n'
    with open(filename, "a") as logfile:
        logfile.write(contents)


def print_group_sparsity(X, K, unique_groups, G_i_starts, G_i_ends, outID=None, epsilon=1e-8):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = ""
    for i in range(K):
        start, end = G_i_starts[i], G_i_ends[i]
        X_Gi = X[start:end]
        normX_Gi = l2_norm(X_Gi)
        if (np.sum(np.abs(X_Gi)) == 0):
            contents += "Group: {:^4d}     sparse | Variable: {:^6d} - {:^6d} are set to zero | norm(X_Gi,2): {:8.3e}\n".format(i + 1, start + 1, end, normX_Gi)

        else:
            contents += "Group: {:^4d} not sparse | ----------------------------------------- | norm(X_Gi,2): {:8.3e}\n".format(i + 1, normX_Gi)
    with open(filename, "a") as logfile:
        logfile.write(contents)


def print_exit(status, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = '\n' + "=" * 30 + '\n'
    if status == -1:
        contents += 'Exit: Line Search Failed\n'
    elif status == 0:
        contents += 'Exit: Optimal Solution Found\n'
    elif status == 1:
        contents += 'Exit: Iteration limit reached\n'
    elif status == 2:
        contents += 'Exit: Time limit reached\n'
    elif status == 3:
        contents += 'Exit: Active set identified\n'
    elif status == 4:
        contents += 'Exit: Early stop as no further progress can be made.\n'
    with open(filename, "a") as logfile:
        logfile.write(contents)
        print(contents)


def print_result(info, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    contents = "\nFinal Results\n"
    contents += "=" * 30 + '\n'
    contents += 'Iterations:{:.>65}{:d}\n'.format("", info['iteration'])
    contents += 'PG iters:{:.>67}{:d}\n'.format("", info['num_pg_steps'])
    contents += 'CG desc iters:{:.>62}{:d}\n'.format("", info['num_cgdesc_steps'])
    contents += 'CG proj iters:{:.>62}{:d}\n'.format("", info['num_cg0_stpes'])
    contents += 'CPU seconds:{:.>64}{:.4f}\n'.format("", info['time'])
    contents += 'Proximal gradient time:{:.>53}{:3.2e}\n'.format("", info['prox_time'])
    contents += 'CG iteration time:{:.>58}{:3.2e}\n'.format("", info['cg_time'])
    contents += 'PG iteration time:{:.>58}{:3.2e}\n'.format("", info['pg_time'])
    contents += 'Newton-CG time:{:.>61}{:3.2e}\n'.format("", info['newton_time'])
    contents += 'CG linesearch time:{:.>57}{:3.2e}\n'.format("", info['cg_ls_time'])
    contents += 'PG linesearch time:{:.>57}{:3.2e}\n'.format("", info['pg_ls_time'])
    contents += 'Step seconds:{:.>63}{:.4f}\n'.format("", info['time_update_stepsize'])
    contents += 'number of sparse groups:{:.>52}{:d}\n'.format("", info['nz'])
    contents += 'Objective function:{:.>57}{:8.6e}\n'.format("", info['f'])
    contents += 'Objective function plus regularizer:{:.>40}{:8.6e}\n'.format("", info['F'])
    contents += 'Optimality error:{:.>59}{:8.6e}\n'.format("", max(info['chipg'], info['chicg']))
    contents += 'Function evaluations:{:.>55}{:d}\n'.format("", info['fevals'])
    contents += 'Gradient evaluations:{:.>55}{:d}\n'.format("", info['gevals'])
    contents += 'Hessian vector products:{:.>52}{:d}\n\n\n'.format("", info['HvProds'])
    with open(filename, "a") as logfile:
        logfile.write(contents)


def _perturb_block(X, start, end, radius, seed, inplace=True):
    # this actually modify the input X !!!!!
    np.random.seed(seed)
    vec = np.random.randn(end - start, 1)
    Xblock = X[start:end]
    d = vec - Xblock
    d_norm = np.sqrt(np.dot(d.T, d))[0][0]
    ans = Xblock + d * radius / d_norm
    if inplace:
        X[start:end] = ans
        return X
    else:
        Y = np.copy(X)
        Y[start:end] = ans
        return Y


def _perturb(X, num_perturb, starts, ends, radius, zeroBlock, seed_base, inplace=False):
    if num_perturb > len(zeroBlock):
        raise ValueError("the num_peturb has to be no larger than len(zeroBlock).")
    radius_per_block = np.sqrt(radius ** 2 / num_perturb)
    if inplace:
        Y = X
    else:
        Y = np.copy(X)
    for i in range(num_perturb):
        block_num = zeroBlock[i]
        start, end = starts[block_num], ends[block_num]
        Y = _perturb_block(X=Y, start=start, end=end, radius=radius_per_block, seed=i + seed_base, inplace=True)
    return Y


def perturb(X, num_perturb, epsilon, starts, ends, zeroBlock, seed_base=1, inplace=False):
    actual_radius = epsilon * np.sqrt(np.dot(X.T, X))[0][0]
    Y = _perturb(X, num_perturb, starts, ends, actual_radius, zeroBlock, seed_base, inplace)
    return Y


def estimate_lipschitz(A, loss='logit'):
    m, n = A.shape
    if loss == 'ls':
        hess = A.T @ A / m
    elif loss == 'logit':
        # acyually this is an upper bound on hess
        hess = A.T @ A / (4 * m)
    hess = hess.toarray()
    L = np.max(np.linalg.eigvalsh(hess))
    return L
