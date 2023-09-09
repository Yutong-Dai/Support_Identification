'''
# File: create_bash.py
# Project: OMS_rebuttal
# Created Date: 2023-08-31 11:53
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2023-09-01 12:52
# Modified By: Yutong Dai yutongdai95@gmail.com
# 
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
'''


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
           accuracy, max_time, max_iters,
           solver, **kwargs):        
    PROJ_DIR = '/home/yutong/Support_Identification'
    PYTHON = '/home/yutong/anaconda3/bin/python'  
    TASK_DIR = f'{PROJ_DIR}/{scriptdir}/OMS_rebuttal'            
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
    command += f' --accuracy {accuracy} --max_time {max_time} --max_iters {max_iters}'
    # shared solver configurations
    command += f' --compute_optim True'
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
    max_time = 28800.0  # 8h
    max_iters = 100000
    seed = 2023
    runs = 1 # change to 3 for final run
    purpose = 'OMS_rebuttal/details'
    # this_run = None
    # this_run = 'parameter_tuning'
    # this_run = 'parameter_tuning_rda'
    this_run = 'final_run'

    accuracy = 1e-4

    task_hypers_template = {'InexactProxGD': {'ipg_strategy':'yd', 'prox_step_strategy':'heuristic', 'rebuttal':True}}

    if this_run == 'final_run':
        # datasets = ['a9a', 'colon_cancer', 'duke', 'gisette', 'leu', 'madelon', 'mushrooms', 'real-sim', 'rcv1', 'w8a']
        datasets = ['colon_cancer']
        for solver in ['InexactProxGD']:
            for lam_shrink in [0.1, 0.01]:
                for chain_grpsize in [10, 100]:
                    for chain_overlap_ratio in [0.1, 0.2, 0.3]:
                            hypers = task_hypers_template[solver]
                            hypers['ext'] = f'chain_grpsize:{chain_grpsize}_chain_overlap_ratio:{chain_overlap_ratio}_lam_shrink:{lam_shrink}'
                            create(scriptdir, data_dir, datasets,
                                purpose, loss, weight_decay, 
                                chain_grpsize, chain_overlap_ratio, lam_shrink,
                                accuracy, max_time, max_iters,
                                solver, **hypers)
    else:
        print("No bash is creared.")           


    

