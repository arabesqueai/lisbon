from pmlb import fetch_data
from memory_profiler import memory_usage
from sklearn import svm
import numpy as np
import time
import sys

X, y = fetch_data("adult", return_X_y=True, local_cache_dir="./pmlb/")
# Make sure the input data is C-style row contigueous array (row orientated)
X = np.ascontiguousarray(X)
y = np.ascontiguousarray(y)


def run():
    linearsvc = svm.LinearSVC(loss="hinge", max_iter=5000, random_state=0)
    t = time.time()
    if not sys.platform.startswith("darwin"):
        mem_usage = memory_usage((linearsvc.fit, [X, y]), interval=1)
    else:
        linearsvc.fit(X, y)
    time_taken = time.time() - t
    print(f"liblinear took {time_taken} seconds and {linearsvc.n_iter_} iterations")
    print("last 10 coefficients: ", linearsvc.coef_[:, -10:])
    print("Intercept: ", linearsvc.intercept_)
    if not sys.platform.startswith("darwin"):
        print("Max memory usage: ", max(mem_usage))
    return time_taken


liblinear_time = run()

import lisbon  # noqa

lisbon_time = run()

if lisbon_time > liblinear_time:
    print("Lisbon is slower than liblinear.")
    exit(1)
