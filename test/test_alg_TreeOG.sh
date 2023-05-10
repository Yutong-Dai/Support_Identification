# test spstorm
cd ../experiments
max_iters=1000
max_epochs=500
depth=12
BASE="python main.py --purpose exp --loss ls --lam_shrink 1e-2 --btree_depth ${depth} --btree_manual_penalty 2.3 --btree_manual_weights True --btree_remove 0.1 --overlap_task btree  --btree_lammax 1.0 --weight_decay 0.00001   --max_time 43200.0 --save_seq True --data_dir ~/db --dataset ls_dep:${depth}_num:2x_sparse:0.1"    

# ProxGD
# $BASE --accuracy 1e-6 --regularizer TreeOG  --max_iters ${max_iters} --solver ProxGD  --proxgd_method ISTA --proxgd_stepsize linesearch &


# SPSTORM
# $BASE --accuracy -1 --regularizer NatOG --max_epochs ${max_epochs} --compute_optim True --solver SPStorm &

# PSTORM
# $BASE --accuracy -1 --regularizer NatOG --max_epochs ${max_epochs} --compute_optim True  --solver PStorm &


# RDA
$BASE --accuracy -1 --regularizer NatOG --max_epochs ${max_epochs} --compute_optim True --solver RDA --rda_stepconst 1.0

# ProxSAGA
# $BASE --accuracy -1 --regularizer NatOG --max_epochs ${max_epochs} --compute_optim True --solver ProxSAGA &
# $BASE --solver ProxSAGA --ipg_strategy linear_decay --ipg_linear_decay_const 0.9

# ProxSVRG
# $BASE --accuracy -1 --regularizer NatOG --max_epochs ${max_epochs} --compute_optim True --solver ProxSVRG &
# $BASE --solver ProxSVRG --ipg_strategy linear_decay --ipg_linear_decay_const 0.9