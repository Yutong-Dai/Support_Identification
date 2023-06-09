import os
import glob
import numpy as np
import sys
sys.path.append("../../")
import src.utils as utils
from copy import deepcopy



def create(scriptdir, data_dir, datasets,
           purpose, loss, weight_decay, 
           chain_grpsize, chain_overlap_ratio, lam_shrink,
           accuracy, max_time, max_epochs,
           seed, runs,
           solver, **kwargs):        
    PROJ_DIR = '/home/yutong/Support_Identification'
    PYTHON = '/home/yutong/anaconda3/bin/python'  
    TASK_DIR = f'{PROJ_DIR}/{scriptdir}/logit_chain'            
    ext = ''
    if 'ext' in kwargs:
        ext = '_' + kwargs['ext']
    task_name = f'{solver}_{loss}{ext}'
    contents = f'cd {PROJ_DIR}/{scriptdir}\n\n'


    # create command lines
    command = f'{PYTHON} main.py --solver {solver} --purpose {purpose} --loss {loss} --weight_decay {weight_decay} --regularizer NatOG'    
    # Problem configurations
    command += f' --overlap_task chain --chain_grpsize {chain_grpsize} --chain_overlap_ratio {chain_overlap_ratio} --lam_shrink {lam_shrink}'
    # termination configurations
    command += f' --accuracy {accuracy} --max_time {max_time} --max_epochs {max_epochs}'
    # shared solver configurations
    command += f' --compute_optim False --ipg_save_log False'
    # Solver configurations
    command += f" --seed {seed} --runs {runs}"
    for datasetname in datasets:
        task = command + f' --data_dir {data_dir} --dataset {datasetname}'
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
    loss = 'logit'
    weight_decay = 1e-5
    ########################## config here ####################
    max_time = 43200.0  # 12h
    seed = 2023
    runs = 1 # change to 3 for final run
    purpose = 'logit_chain/details'
    this_run = None
    # this_run = 'parameter_tuning'
    # this_run = 'parameter_tuning_rda'
    this_run = 'final_run'
    
    task_hypers_template = {
            'ProxSVRG': {'proxsvrg_inner_repeat': 1, 'proxsvrg_lipcoef': 1.0, 'ipg_strategy': 'diminishing'},
            'ProxSAGA': {'proxsaga_lipcoef': 1.0, 'ipg_strategy': 'diminishing'},
            'SPStorm': {'spstorm_betak': -1.0, 'spstorm_zeta': 'dynanmic', 'spstorm_lipcoef': 1.0, 'ipg_strategy': 'diminishing'},
            'PStorm': {'pstorm_stepsize': 'diminishing', 'pstorm_betak': -1.0, 'pstorm_lipcoef': 1.0, 'ipg_strategy': 'diminishing'},
            'RDA': {'rda_stepconst': 1.0, 'ipg_strategy': 'diminishing'}
        }


    if this_run == 'parameter_tuning':
        datasets = ["a9a", "w8a"]
        accuracy = -1.0  # disable chi termination options 
        max_epochs = 500
        # for solver in ['ProxSVRG']:
        for solver in ['ProxSAGA', 'PStorm', 'SPStorm']:
            for lam_shrink in [0.1, 0.01]:
                for chain_grpsize in [10, 100]:
                    for chain_overlap_ratio in [0.1, 0.2, 0.3]:
                        for const in [0.1, 0.5, 1.0]:
                            hypers = deepcopy(task_hypers_template[solver])
                            solver_lower_case = solver.lower()
                            hypers[f'{solver_lower_case}_lipcoef'] = const
                            hypers['ext'] = f'lipcoef:{const}_chain_grpsize:{chain_grpsize}_chain_overlap_ratio:{chain_overlap_ratio}_lam_shrink:{lam_shrink}'
                            create(scriptdir, data_dir, datasets,
                                purpose, loss, weight_decay, 
                                chain_grpsize, chain_overlap_ratio, lam_shrink,
                                accuracy, max_time, max_epochs,
                                seed, runs,
                                solver, **hypers)
    elif this_run == 'parameter_tuning_rda':
        datasets = ["a9a", "w8a"]
        accuracy = -1.0  # disable chi termination options 
        max_epochs = 500
        for solver in ['RDA']:
            for lam_shrink in [0.1, 0.01]:
                for chain_grpsize in [10, 100]:
                    for chain_overlap_ratio in [0.1, 0.2, 0.3]:
                        for const in [0.001, 0.01, 0.1, 1.0, 10.0]:
                            hypers = deepcopy(task_hypers_template[solver])
                            solver_lower_case = solver.lower()
                            hypers['rda_stepconst'] = const
                            hypers['ext'] = f'lipcoef:{const}chain_grpsize:{chain_grpsize}_chain_overlap_ratio:{chain_overlap_ratio}_lam_shrink:{lam_shrink}'
                            create(scriptdir, data_dir, datasets,
                                purpose, loss, weight_decay, 
                                chain_grpsize, chain_overlap_ratio, lam_shrink,
                                accuracy, max_time, max_epochs,
                                seed, runs,
                                solver, **hypers)                                

    elif this_run == 'final_run':
        datasets = ['rcv1', 'real-sim', 'news20']
        max_epochs = 500
        accuracy = -1.0  # disable chi termination options
        task_hypers_template = {
            'ProxSVRG': {'proxsvrg_inner_repeat': 1, 'proxsvrg_lipcoef': 1.0, 'ipg_strategy': 'diminishing'},
            'ProxSAGA': {'proxsaga_lipcoef': 1.0, 'ipg_strategy': 'diminishing'},
            'SPStorm': {'spstorm_betak': -1.0, 'spstorm_zeta': 'dynanmic', 'spstorm_lipcoef': 1.0, 'ipg_strategy': 'diminishing'},
            'PStorm': {'pstorm_stepsize': 'diminishing', 'pstorm_betak': -1.0, 'pstorm_lipcoef': 1.0, 'ipg_strategy': 'diminishing'},
            'RDA': {'rda_stepconst': 0.01, 'ipg_strategy': 'diminishing'}
        }
        for solver in ['RDA']:
        # for solver in ['ProxSVRG', 'ProxSAGA', 'PStorm', 'SPStorm']:
            for lam_shrink in [0.1, 0.01]:
                for chain_grpsize in [10, 100]:
                    for chain_overlap_ratio in [0.1, 0.2, 0.3]:
                            hypers = task_hypers_template[solver]
                            hypers['ext'] = f'chain_grpsize:{chain_grpsize}_chain_overlap_ratio:{chain_overlap_ratio}_lam_shrink:{lam_shrink}'
                            create(scriptdir, data_dir, datasets,
                                purpose, loss, weight_decay, 
                                chain_grpsize, chain_overlap_ratio, lam_shrink,
                                accuracy, max_time, max_epochs,
                                seed, runs,
                                solver, **hypers)
    else:
        print("No bash is creared.")           


    

