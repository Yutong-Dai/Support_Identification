cd ../experiments
BASE="python main.py --solver ProxSVRG --purpose test_linear --loss ls --weight_decay 1e-05 --regularizer NatOG --overlap_task btree --btree_manual_weights True --btree_lammax 1.0 --lam_shrink 1e-2 --accuracy -1.0 --max_time 3600.0 --max_epochs 20 --max_iters 1 --compute_optim True --ipg_save_log True --seed 2023 --runs 1 --data_dir ~/db --dataset ls_dep:11_num:2x_sparse:0.1 --btree_depth 11 --btree_remove 0.05 --proxsvrg_inner_repeat 1 --proxsvrg_lipcoef 1.0"

${BASE} --ipg_strategy linear_decay &
${BASE} --ipg_strategy diminishing &
