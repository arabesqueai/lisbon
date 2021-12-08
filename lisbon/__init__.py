try:
    from sklearn import svm
except ModuleNotFoundError:
    print("Please install scikit-learn to use lisbon.")
    exit()

from lisbon import lisbon

orig_fit = svm.LinearSVC.fit


def _lift(self, X, y, sample_weight=None):
    if (
        svm._base._get_liblinear_solver_type(
            self.multi_class, self.penalty, self.loss, self.dual
        )
        == 3
    ):
        svm._base.liblinear = lisbon
    return orig_fit(self, X, y, sample_weight)


svm.LinearSVC.fit = _lift
