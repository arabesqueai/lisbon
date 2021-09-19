from sklearn import base, datasets
from sklearn import svm
import numpy as np
import lisbon
import time

iris = datasets.load_breast_cancer()
be = base.BaseEstimator()

X, y = be._validate_data(
    iris.data,
    iris.target,
    accept_sparse="csr",
    dtype=np.float64,
    order="C",
    accept_large_sparse=False,
)

# print(X.shape)
linearsvc = svm.LinearSVC(loss='hinge', max_iter=10000, random_state=0)
t = time.time()
linearsvc.fit(X, y)
print(f"liblinear took {time.time() - t} seconds")
print(linearsvc.coef_, linearsvc.intercept_)
svm._base.liblinear = lisbon
t = time.time()
linearsvc.fit(X, y)
print(f"lisbon took {time.time() - t} seconds")
print(linearsvc.coef_, linearsvc.intercept_)