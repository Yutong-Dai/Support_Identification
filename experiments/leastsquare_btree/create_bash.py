import os
import glob
import numpy as np
import sys
sys.path.append("../../")
import src.utils as utils
from copy import deepcopy



def create(scriptdir, data_dir,
           purpose, loss, weight_decay, 
           depth_lst, remove_ratio_lst, 
           accuracy, max_time, max_epochs, max_iters,
           seed, runs,
           solver, **kwargs):        
    PROJ_DIR = '/home/yutong/Support_Identification'
    PYTHON = '/home/yutong/anaconda3/bin/python'  
    TASK_DIR = f'{PROJ_DIR}/{scriptdir}/leastsquare_btree'            
    ext = ''
    if 'ext' in kwargs:
        ext = '_' + kwargs['ext']
    task_name = f'{solver}_{loss}{ext}'
    contents = f'cd {PROJ_DIR}/{scriptdir}\n\n'


    # create command lines
    command = f'{PYTHON} main.py --solver {solver} --purpose {purpose} --loss {loss} --weight_decay {weight_decay}'
    if solver != "ProxGD":
        command += f' --regularizer NatOG'
    else:
        command += f' --regularizer TreeOG'
    
    # Problem configurations
    command += f' --overlap_task btree --btree_manual_weights True --btree_lammax 1.0 --lam_shrink 1e-2'
    # termination configurations
    command += f' --accuracy {accuracy} --max_time {max_time} --max_epochs {max_epochs} --max_iters {max_iters}'
    # shared solver configurations
    command += f' --compute_optim False --ipg_save_log False'
    # Solver configurations
    if solver != "ProxGD":
        command += f" --seed {seed} --runs {runs}"
    for depth in depth_lst:
        datasetname = f'ls_dep:{depth}_num:2x_sparse:0.1'
        for remove_ratio in remove_ratio_lst:
            task = command + f' --data_dir {data_dir} --dataset {datasetname} --btree_depth {depth} --btree_remove {remove_ratio}'
            task_hypers = ""
            for k, v in kwargs.items():
                if k != 'ext':
                    task_hypers += f" --{k} {v}"
            task += task_hypers + f" >> {TASK_DIR}/log/{task_name}.txt &"
            contents += task + '\n\n'
    filename = f'./{task_name}.sh'
    with open(filename, "w") as pbsfile:
        pbsfile.write(contents)


if __name__ == '__main__':
    # clean all existing bash files
    for f in glob.glob("*.sh"):
        os.remove(f)
    scriptdir = 'experiments'
    data_dir = '~/db'
    logdir = "./log"
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    loss = 'ls'
    weight_decay = 1e-5
    ########################## config here ####################
    max_time = 43200.0  # 12h
    seed = 2023
    runs = 1 # change to 3 for final run
    purpose = 'leastsquare_btree/details'
    # this_run = 'get_ground_truth'
    # this_run = 'parameter_tuning'
    # this_run = 'diminishing_error_for_proxsvrg_and_proxsaga'
    # this_run = 'final_run'
    # this_run = 'final_run_rda'

    task_hypers_template = {
            'ProxGD': {'proxgd_method': 'ISTA', 'proxgd_stepsize': 'linesearch'},
            'ProxSVRG': {'proxsvrg_inner_repeat': 1, 'proxsvrg_lipcoef': 1.0, 'ipg_strategy': 'linear_decay', 'ipg_linear_decay_const':0.99},
            'ProxSAGA': {'proxsaga_lipcoef': 1.0, 'ipg_strategy': 'linear_decay', 'ipg_linear_decay_const':0.99},
            'SPStorm': {'spstorm_betak': -1.0, 'spstorm_zeta': 'dynanmic', 'spstorm_lipcoef': 1.0, 'ipg_strategy': 'diminishing'},
            'PStorm': {'pstorm_stepsize': 'diminishing', 'pstorm_betak': -1.0, 'pstorm_lipcoef': 1.0, 'ipg_strategy': 'diminishing'},
            'RDA': {'rda_stepconst': 1.0, 'ipg_strategy': 'diminishing'}
        }


    if this_run == 'get_ground_truth':
        solver = 'ProxGD'
        accuracy = 1e-6 
        max_epochs = 1
        max_iters = 100000
        depth_lst = [11, 12, 13, 14, 15]
        remove_ratio_lst = [0.05, 0.10, 0.20]
        hypers = task_hypers_template[solver]
        create(scriptdir, data_dir,
               purpose, loss, weight_decay,
               depth_lst, remove_ratio_lst,
               accuracy, max_time, max_epochs, max_iters, 
               seed, runs, solver, **hypers)
    elif this_run == 'parameter_tuning':
        # tunig stepconst
        depth_lst = [11, 12, 13]
        remove_ratio_lst = [0.05, 0.10, 0.20]
        max_iters = 1
        max_epochs = 500
        accuracy = -1.0  # disable chi termination options
        for solver in ['ProxSVRG', 'ProxSAGA', 'RDA', 'PStorm', 'SPStorm']:
            for const in [1.0, 0.5, 0.1]:
                hypers = deepcopy(task_hypers_template[solver])
                if solver == 'RDA':
                    hypers['rda_stepconst'] = const
                    hypers['ext'] = f'stepconst:{const}'
                else:
                    solver_lower_case = solver.lower()
                    hypers[f'{solver_lower_case}_lipcoef'] = const
                    hypers['ext'] = f'lipcoef:{const}'
                create(scriptdir, data_dir,
                        purpose, loss, weight_decay,
                        depth_lst, remove_ratio_lst,
                        accuracy, max_time, max_epochs, max_iters,
                        seed, runs, solver, **hypers)
    elif this_run == 'diminishing_error_for_proxsvrg_and_proxsaga':
        depth_lst = [11, 12, 13]
        remove_ratio_lst = [0.05, 0.10, 0.20]
        max_iters = 1
        max_epochs = 500
        accuracy = -1.0  # disable chi termination options
        for solver in ['ProxSVRG', 'ProxSAGA']:
            for const in [1.0, 0.5, 0.1]:
                hypers = deepcopy(task_hypers_template[solver])
                solver_lower_case = solver.lower()
                hypers[f'{solver_lower_case}_lipcoef'] = const
                hypers[f'ipg_strategy'] = 'diminishing'
                hypers['ext'] = f'lipcoef:{const}_ipg_strategy:diminishing'
                create(scriptdir, data_dir,
                        purpose, loss, weight_decay,
                        depth_lst, remove_ratio_lst,
                        accuracy, max_time, max_epochs, max_iters,
                        seed, runs, solver, **hypers)
    elif this_run == 'final_run':
        depth_lst = [14, 15]
        remove_ratio_lst = [0.05, 0.10, 0.20]
        max_iters = 1
        max_epochs = 500
        accuracy = -1.0  # disable chi termination options
        for solver in ['ProxSVRG', 'ProxSAGA', 'PStorm', 'SPStorm']:
            for const in [1.0]:
                hypers = deepcopy(task_hypers_template[solver])
                if solver == 'RDA':
                    hypers['rda_stepconst'] = const
                    hypers['ext'] = f'stepconst:{const}'
                else:
                    solver_lower_case = solver.lower()
                    hypers[f'{solver_lower_case}_lipcoef'] = const
                    hypers['ext'] = f'lipcoef:{const}'

                if solver in ['ProxSVRG', 'ProxSAGA']:
                    hypers[f'ipg_strategy'] = 'diminishing'
                    del hypers['ipg_linear_decay_const']
                create(scriptdir, data_dir,
                        purpose, loss, weight_decay,
                        depth_lst, remove_ratio_lst,
                        accuracy, max_time, max_epochs, max_iters,
                        seed, runs, solver, **hypers)
    elif this_run == 'final_run_rda':
        # tunig stepconst
        depth_lst = [14, 15]
        remove_ratio_lst = [0.05, 0.10, 0.20]
        max_iters = 1
        max_epochs = 500
        accuracy = -1.0  # disable chi termination options
        for solver in ['RDA']:
            for const in [1.0, 10.0]:
                hypers = deepcopy(task_hypers_template[solver])
                if solver == 'RDA':
                    hypers['rda_stepconst'] = const
                    hypers['ext'] = f'stepconst:{const}'
                else:
                    solver_lower_case = solver.lower()
                    hypers[f'{solver_lower_case}_lipcoef'] = const
                    hypers['ext'] = f'lipcoef:{const}'
                create(scriptdir, data_dir,
                        purpose, loss, weight_decay,
                        depth_lst, remove_ratio_lst,
                        accuracy, max_time, max_epochs, max_iters,
                        seed, runs, solver, **hypers)             
    else:
        print("No bash is creared.")           


    

