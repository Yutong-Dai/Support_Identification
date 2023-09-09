from memory_profiler import profile
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize
import argparse
import numpy as np
import scipy
import time
import os


@profile
def load(filepath, to_dense=False):
    print(filepath)
    data = load_svmlight_file(filepath)
    X, y = data[0], data[1].reshape(-1, 1)
    if to_dense:
        print('converting')
        X = X.toarray()
    return X, y


def cleanData(datasetName, X, y):
    print(f"Process:{datasetName}")
    X = normalize(X, axis=1)
    ylabel = np.unique(y)
    if not np.array_equal(ylabel, np.array([-1, 1])):
        a = ylabel[0]
        b = ylabel[1]
        y[y == a] = -1
        y[y == b] = 1
        print(f"{datasetName}: Modified y lables from", ylabel, "to {-1,1}!", flush=True)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    # Problem setup
    parser.add_argument("--datasetdir", type=str, default=os.path.expanduser("~/db"), help='directory of datasets')
    parser.add_argument("--datasetname", type=str, default="a9a", help='Dataset Name.')
    parser.add_argument("--ext", type=str, default=None, help='extension')
    parser.add_argument("--rename", type=str, default=None, help='rename')
    config = parser.parse_args()
    if config.rename is None:
        savefile = f"{config.datasetdir}/{config.datasetname}.npy"
    else:
        savefile = f"{config.datasetdir}/{config.rename}.npy"
    if os.path.exists(savefile):
        print(f"{savefile} exists! Existing...")
        exit()
    print(f'loading_{config.datasetname}', flush=True)
    if config.ext is not None:
        filename = f"{config.datasetdir}/{config.datasetname}.{config.ext}"
    else:
        filename = f"{config.datasetdir}/{config.datasetname}"
    start = time.time()
    X, y = load(filename)
    time_elapsed = time.time() - start
    print(f"Spent {time_elapsed:.1f} seconds loading data.", flush=True)
    print('cleaning')
    X, y = cleanData(config.datasetname, X, y)
    np.save(savefile, {'X': X, 'y': y})
    print(f"Success: File saved at {savefile}!", flush=True)
    print("=" * 20)


if __name__ == "__main__":
    main()
