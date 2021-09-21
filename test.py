from pmlb import fetch_data
from memory_profiler import memory_usage
from sklearn import svm
import lisbon
import numpy as np
import time
import sys

X, y = fetch_data("adult", return_X_y=True, local_cache_dir="./pmlb/")
# Make sure the input data is C-style row contigueous array (row orientated)
X = np.ascontiguousarray(X)
y = np.ascontiguousarray(y)


# Memory profiler doesn't run on MacOS
if sys.platform.startswith("darwin"):
    linearsvc = svm.LinearSVC(loss="hinge", max_iter=1000, random_state=0)
    t = time.time()
    linearsvc.fit(X, y)
    print(
        f"liblinear took {time.time() - t} seconds and {linearsvc.n_iter_} iterations"
    )
    print("last 10 coefficients: ", linearsvc.coef_[:, -10:])
    print("Intercept: ", linearsvc.intercept_)

    svm._base.liblinear = lisbon

    t = time.time()
    linearsvc.fit(X, y)
    print(f"lisbon took {time.time() - t} seconds and {linearsvc.n_iter_} iterations")
    print("last 10 coefficients: ", linearsvc.coef_[:, -10:])
    print("Intercept: ", linearsvc.intercept_)
else:
    linearsvc = svm.LinearSVC(loss="hinge", max_iter=1000, random_state=0)
    t = time.time()
    mem_usage = memory_usage((linearsvc.fit, [X, y]), interval=1)
    print(
        f"liblinear took {time.time() - t} seconds and {linearsvc.n_iter_} iterations"
    )
    print("last 10 coefficients: ", linearsvc.coef_[:, -10:])
    print("Intercept: ", linearsvc.intercept_)
    print("Max memory usage: ", max(mem_usage))

    svm._base.liblinear = lisbon

    t = time.time()
    mem_usage = memory_usage((linearsvc.fit, [X, y]), interval=1)
    print(f"lisbon took {time.time() - t} seconds and {linearsvc.n_iter_} iterations")
    print("last 10 coefficients: ", linearsvc.coef_[:, -10:])
    print("Intercept: ", linearsvc.intercept_)
    print("Max memory usage: ", max(mem_usage))
