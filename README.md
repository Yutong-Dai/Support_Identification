# Support_Identification
A collection of algorithms for support identification.

# todo
- [ ] finalize least square datasets
- [ ] modify the leastsquare loss functions 

# Setup
0. (Optional) If you want to change where the datasets are stored, change the following files.
   * Set the `DATASETS_DIR` variable in `experiments/setup/download.sh` to the desired directory, e.g, `DATASETS_DIR="<your_dir>"`.
   * Add `--datasetdir` to the commands in the `experiments/setup/process.sh` script. For example, `$PYTHON -m memory_profiler process_data.py --datasetname a9a --datasetdir <your_dir> >> process_log.txt`
   * Change `ROOT` variable in `experiments/setup/compute_Lip.py` to `<your_dir>`.

1. Change the path to the `Python` interpretor of the choice. Specifically, change `PYTHON` variable in the `experiments/setup/process.sh`.
2. At the root of the project directory, run `cd experiments/setup && bash setup.sh`. A `process_log.txt` file will be created to show the data preprocessing results.

# Dataset criterion
features less than 50 or samples less than 10000

blogData n=60,021 p=281
driftData n=13910 p=128
log1p.E2006.tfidf n=16,087 p=4,272,227
UJIdoorLoc n=19,937 p=520
VirusShare n=107,888 p=482
YearPredictionMSD n=463,715 p=90
AmazonAcess n=30,000 p=20,000
BuzzSocialMedia n=140000, p=77
FacebookComment n=40949, p = 54

https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/log1p.E2006.train.bz2
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2
https://archive.ics.uci.edu/ml/machine-learning-databases/00304/BlogFeedback.zip
https://archive.ics.uci.edu/ml/machine-learning-databases/00310/UJIndoorLoc.zip
https://archive.ics.uci.edu/ml/machine-learning-databases/00270/driftdataset.zip
https://archive.ics.uci.edu/ml/machine-learning-databases/00413/dataset.zip
https://archive.ics.uci.edu/ml/machine-learning-databases/00216/amzn-anon-access-samples.tgz
https://archive.ics.uci.edu/ml/machine-learning-databases/00248/regression.tar.gz
https://archive.ics.uci.edu/ml/machine-learning-databases/00363/Dataset.zip



# Proposed Experiments

## Support Identification (Tree structure)
Using the simulated dataset $D=(A,b)$ and solve $\frac{1}{2}\|Ax-b\|^2 + \lambda r(x)$, where $A$ is full column rank and $r(x)$ encodes the overlapping groups with a tree structure. The prox-operator can be comuted exactly using an iterative solver, which converges in $|n_G|$ iterations. Then we can use any proximal-gradient type methods to compute the "ground truth".


## Arbitrarily overlapping (Chain Like)

Using the one that has the best solution quality in terms of 
1. the two solutions have the smallest objective value
2. among these two choose whichever has the spartest structure as the ground truth solution.

## Strongly Convex Case

$\text{logistic loss} + 10^{-5}\|x\|^2 + \lambda r(x)$. 

## Convex Case

$\frac{1}{2}\|Ax-b\|^2 + \lambda r(x)$: $A\in R^{m\times n}$ and $m\leq n$.

---

# Scope of this paper? Focus on support identificaiton or change the empahsis on the algortihm itself.


# TroubleShooting
1. How to install graphviz
> 1. pip install graphviz
> 2. sudo apt install graphviz