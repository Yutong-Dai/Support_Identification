# test spstorm
cd ../experiments
compute_optim=True
# regularizer="GL1"
regularizer="NatOG"
loss="logit"
max_epochs=100
# check whether regularizer is GL1
if [ $regularizer = "GL1" ]; then
    BASE="python main.py --purpose exp --loss ${loss} --regularizer GL1 --lam_shrink 0.1 --frac 0.5 --weight_decay 0.0001 --accuracy -1.0 --max_epochs ${max_epochs} --max_time 43200.0 --save_seq False --compute_optim ${compute_optim} --data_dir ~/db --dataset a9a "

else
    BASE="python main.py --purpose exp --loss ${loss} --regularizer NatOG  --overlap_task chain --lam_shrink 0.1 --chain_grpsize 10 --chain_overlap_ratio 0.1 --weight_decay 0.0001 --accuracy -1.0 --max_epochs ${max_epochs} --max_time 43200.0 --save_seq True --compute_optim ${compute_optim} --data_dir ~/db --dataset a9a"    
fi

# SPSTORM
# $BASE --solver SPStorm --spstorm_lipcoef 1.0 &

# PSTORM
# $BASE --solver PStorm


# RDA
$BASE --solver RDA --rda_stepconst 0.01 &

# ProxSAGA
# $BASE --solver ProxSAGA
# $BASE --solver ProxSAGA --ipg_strategy linear_decay --ipg_linear_decay_const 0.9

# ProxSVRG
# $BASE --solver ProxSVRG
# $BASE --solver ProxSVRG --ipg_strategy linear_decay --ipg_linear_decay_const 0.9