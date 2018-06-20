###############################################################################
#                                                                             #
# Fitting the data and tagging outliers. Additionally adding methods for      #
# useful metrics like threshold and number of errors.                         #
#                                                                             #
###############################################################################

from scipy import stats


def outliers(clf, X, clf_name):
    if clf_name == "LOF":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

    return y_pred, scores_pred


def threshold(scores_pred, outliers_fraction):
    return stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)


def nErrors(y_pred, ground_truth):
    return (y_pred != ground_truth).sum()
