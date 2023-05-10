cd ../experiments

BASE="python main.py --purpose find_btree_exp_hypers --loss ls --regularizer TreeOG --solver ProxGD --accuracy 1e-6  --max_iters 100000 --max_time 43200.0 --overlap_task btree  --btree_manual_weights True  --btree_lammax 1.0 --weight_decay 0.00001  --save_seq True --data_dir ~/db"    

# depth=11
# $BASE --dataset ls_dep:${depth}_num:2x_sparse:0.1 --lam_shrink 1e-2 --btree_depth ${depth} --btree_remove 0.05 --btree_manual_penalty 2.4 &
# $BASE --dataset ls_dep:${depth}_num:2x_sparse:0.1 --lam_shrink 1e-2 --btree_depth ${depth} --btree_remove 0.10 --btree_manual_penalty 2.5 &
# $BASE --dataset ls_dep:${depth}_num:2x_sparse:0.1 --lam_shrink 1e-2 --btree_depth ${depth} --btree_remove 0.20 --btree_manual_penalty 2.5 &

# depth=12
# $BASE --dataset ls_dep:${depth}_num:2x_sparse:0.1 --lam_shrink 1e-2 --btree_depth ${depth} --btree_remove 0.05 --btree_manual_penalty 2.9 &
# $BASE --dataset ls_dep:${depth}_num:2x_sparse:0.1 --lam_shrink 1e-2 --btree_depth ${depth} --btree_remove 0.10 --btree_manual_penalty 2.3 &
# $BASE --dataset ls_dep:${depth}_num:2x_sparse:0.1 --lam_shrink 1e-2 --btree_depth ${depth} --btree_remove 0.20 --btree_manual_penalty 2.4 &

# depth=13
# $BASE --dataset ls_dep:${depth}_num:2x_sparse:0.1 --lam_shrink 1e-2 --btree_depth ${depth} --btree_remove 0.05 --btree_manual_penalty 2.9 &
# $BASE --dataset ls_dep:${depth}_num:2x_sparse:0.1 --lam_shrink 1e-2 --btree_depth ${depth} --btree_remove 0.10 --btree_manual_penalty 2.5 &
# $BASE --dataset ls_dep:${depth}_num:2x_sparse:0.1 --lam_shrink 1e-2 --btree_depth ${depth} --btree_remove 0.20 --btree_manual_penalty 3.0 &

# depth=14
# $BASE --dataset ls_dep:${depth}_num:2x_sparse:0.1 --lam_shrink 1e-2 --btree_depth ${depth} --btree_remove 0.05 --btree_manual_penalty 2.9 &
# $BASE --dataset ls_dep:${depth}_num:2x_sparse:0.1 --lam_shrink 1e-2 --btree_depth ${depth} --btree_remove 0.10 --btree_manual_penalty 2.4 &
# $BASE --dataset ls_dep:${depth}_num:2x_sparse:0.1 --lam_shrink 1e-2 --btree_depth ${depth} --btree_remove 0.20 --btree_manual_penalty 2.7 &

# depth=15
# $BASE --dataset ls_dep:${depth}_num:2x_sparse:0.1 --lam_shrink 1e-2 --btree_depth ${depth} --btree_remove 0.05 --btree_manual_penalty 3.0 &
# $BASE --dataset ls_dep:${depth}_num:2x_sparse:0.1 --lam_shrink 1e-2 --btree_depth ${depth} --btree_remove 0.10 --btree_manual_penalty 2.3 &
# $BASE --dataset ls_dep:${depth}_num:2x_sparse:0.1 --lam_shrink 1e-2 --btree_depth ${depth} --btree_remove 0.20 --btree_manual_penalty 2.8 &

# depth=15
# for penalty in 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0
# do
#     $BASE --dataset ls_dep:${depth}_num:2x_sparse:0.1 --lam_shrink 1e-2 --btree_depth ${depth} --btree_remove 0.10 --btree_manual_penalty ${penalty} &
# done



#depth:ratio:penalty
# { 
# 11: {0.05:2.4, 0.1:2.5, 0.2:2.5}, 
# 12: {0.05:2.9, 0.1:2.3, 0.2:2.4}, 
# 13: {0.05:2.9, 0.1:2.5, 0.2:3.0}, 
# 14: {0.05:2.9, 0.1:2.4, 0.2:2.7},
# 15: {0.05:3.0, 0.1:2.3, 0.2:2.8}, 
# }