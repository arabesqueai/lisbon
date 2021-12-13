from pmlb import fetch_data
from memory_profiler import memory_usage
from sklearn import svm
import numpy as np
import time
import sys
import os

X, y = fetch_data("adult", return_X_y=True, local_cache_dir="./pmlb/")
# Make sure the input data is C-style row contigueous array (row orientated)
X = np.ascontiguousarray(X)
y = np.ascontiguousarray(y)


def run():
    linearsvc = svm.LinearSVC(loss="hinge", max_iter=500, random_state=0)
    t = time.time()
    if not sys.platform.startswith("darwin"):
        mem_usage = memory_usage((linearsvc.fit, [X, y]), interval=1)
    else:
        linearsvc.fit(X, y)
    time_taken = time.time() - t
    print(f"Run took {time_taken} seconds and {linearsvc.n_iter_} iterations")
    print("last 10 coefficients: ", linearsvc.coef_[:, -10:])
    print("Intercept: ", linearsvc.intercept_)
    if not sys.platform.startswith("darwin"):
        print("Max memory usage: ", max(mem_usage))
    return linearsvc.coef_, linearsvc.intercept_, time_taken


liblinear_coef, liblinear_intercept, liblinear_time = run()

sys.path.remove(os.getcwd())  # so that python imports from installed site-package
import lisbon  # noqa

lisbon_coef, lisbon_intercept, lisbon_time = run()

if not np.allclose(liblinear_coef, lisbon_coef, rtol=0, atol=1e-15) or not np.allclose(
    liblinear_intercept, lisbon_intercept, rtol=0, atol=1e-15
):
    print("Lisbon result is different from liblinear. There might be an error.")
    exit(1)

if lisbon_time > liblinear_time:
    print("Lisbon is slower than liblinear.")
    exit(1)
