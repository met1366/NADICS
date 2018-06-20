###############################################################################
#                                                                             #
# Evaluating, selecting and extracting the relevant features from our dataset.#
#                                                                             #
# NOTE: For getting a list of the most important features we make use of      #
# 'ExtraTreesClassifier' of which we found out works best regarding our       #
# application. Sklearn also provides other classifiers to do so, so feel free #
# to modify the 'importances' function to your needs.                         #
# Also, the number of estimators might be chosen differently depending on the #
# dataset you use.                                                            #
#                                                                             #
###############################################################################

import os
import matplotlib as mpl
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

def importances(xTrain, yTrain):
    forest = ExtraTreesClassifier(n_estimators=250)
    forest.fit(xTrain, yTrain)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_
                  for tree in forest.estimators_],
                 axis=0)

    return importances, \
        np.argsort(importances)[::-1], \
        std


def printImportances(xTrain, xFeatures, importances, indices):
    for f in range(xTrain.shape[1]):
        print("%d. %s (%f)" % (f + 1, xFeatures[indices[f]],
                               importances[indices[f]]))
        if (importances[indices[f]] < 0.01):
            print("...")
            break


def saveImportances(xTrain, xFeatures, importances, indices, filename):
    fi = open(filename, "w")
    for f in range(xTrain.shape[1]):
        fi.write("%d,%s,%f\n" % (f + 1, xFeatures[indices[f]],
                                 importances[indices[f]]))


def plotImportances(xTrain, xFeatures, importances, indices, std, filename):
    ylabels = []
    for i in indices:
        ylabels.append(list(xFeatures)[i])

    fig = plt.figure(figsize=[8, 8])
    fig.subplots_adjust(left=0.25)

    plt.title("Feature Importances")
    plt.barh(range(xTrain.shape[1]), np.flipud(importances[indices]),
             color="b", xerr=np.flipud(std[indices]), align="center")
    plt.yticks(range(xTrain.shape[1]), np.flipud(ylabels))
    plt.ylim([-1, xTrain.shape[1]])

    plt.savefig(filename, format="svg")
