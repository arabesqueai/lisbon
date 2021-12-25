try:
    from sklearn import svm as __svm
    from sklearn.utils.multiclass import type_of_target
except ModuleNotFoundError:
    print("Please install scikit-learn to use lisbon.")
    exit()

from lisbon import lisbon

__orig_fit = __svm.LinearSVC.fit


def __lift(self, X, y, sample_weight=None):
    # conditionally swap liblinear out for lisbon if the routine matches
    if (
        __svm._base._get_liblinear_solver_type(
            self.multi_class, self.penalty, self.loss, self.dual
        )
        == 3 and type_of_target(y) == "binary"
    ):
        __svm._base.liblinear = lisbon
    else:
        __svm._base.liblinear = __svm._liblinear
    fitted = __orig_fit(self, X, y, sample_weight)
    __svm._base.liblinear = __svm._liblinear
    return fitted


__svm.LinearSVC.fit = __lift


def unload():
    __svm.LinearSVC.fit = __orig_fit
