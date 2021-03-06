###############################################################################
#                                                                             #
# Learning includes the tasks of training, validation and prediction. In      #
# order to do so, we provide a queueing mechanism for queueing jobs which     #
# might be a certain model fitting or a following prediction regarding a      #
# trained model.                                                              #
# This also includes time dependant paremeters to stop or kill jobs at a      #
# given time.                                                                 #
#                                                                             #
# Using queues you are also able to specify multiple models as input which    #
# will be run subsequently and stopped after a certain waiting time is        #
# exceeded.                                                                   #
#                                                                             #
# NOTE: In order to determine good hyperparameters for certain models we      #
# included the function gridSearch which might be modified to your needs in   #
# terms of classifiers and parameters.                                        #
# In future versions this function will be moved to a more proper place.      #
#                                                                             #
###############################################################################


import Queue
import multiprocessing
from sklearn.model_selection import GridSearchCV


def fit(queue, clf, X, y):
    fit = clf.fit(X, y)
    queue.put(fit)


def predict(queue, clf, X):
    y_pred = (clf.predict(X))
    queue.put(y_pred)


def queueingJob(queue, process, timeWaiting):
    process.start()

    try:
        item = queue.get(True, timeWaiting)
    except Queue.Empty:
        kill(process)

    process.join()

    if process.is_alive():
        process.terminate()
        process.join()

    return item


def training(clf, xTrain, yTrain, timeWaiting):
    q = multiprocessing.Queue()
    process = multiprocessing.Process(target=fit,
                                      args=(q,
                                            clf,
                                            xTrain,
                                            yTrain))
    return queueingJob(q, process, timeWaiting)


def testing(clf, xTest, timeWaiting):
    q = multiprocessing.Queue()
    process = multiprocessing.Process(target=predict,
                                      args=(q,
                                            clf,
                                            xTest))
    return queueingJob(q, process, timeWaiting)


def saveResults(score, filename):
    fi = open(filename, "w")
    fi.write("%s\n" % (str(round(score, 3))))


def kill(process):
    process.terminate()
    process.join()


def gridSearch(clf, clf_arg):
    dt = [{"splitter": ["best", "random"],
           "max_depth": [2, 4, 8, 16, 32],
           "min_samples_split": [2, 4, 8, 16, 32],
           "min_samples_leaf": [1, 2, 4, 8, 16],
           "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4],
           "max_features": [None, "sqrt", "log2"],
           "max_leaf_nodes": [None, 2, 4, 8, 16],
           "min_impurity_decrease": [0.0, 0.1, 0.2, 0.3, 0.4],
           "class_weight": [None, "balanced"],
           "presort": [False, True]}]
    rf = [{"n_estimators": [10, 20, 40, 80, 160, 320],
           "max_depth": [2, 4, 8, 16, 32],
           "min_samples_split": [2, 4, 8, 16, 32],
           "min_samples_leaf": [1, 2, 4, 8, 16],
           "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4],
           "max_leaf_nodes": [None, 2, 4, 8, 16],
           "min_impurity_decrease": [0.0, 0.1, 0.2, 0.3, 0.4],
           "n_jobs": [-1],
           "warm_start": [False, True],
           "class_weight": [None, "balanced"]}]
    knn = [{"n_neighbors": [5, 10, 20, 40, 80, 160],
            "weights": ["uniform", "distance"],
            "algorithm": ["ball_tree", "kd_tree", "brute"],
            "leaf_size": [30, 60, 120, 240, 480],
            "p": [2, 4, 8, 16, 32],
            "n_jobs": [-1]}]

    gridSearchParams = {"DecisionTree": dt,
                        "RandomForest": rf,
                        "KNeighbors": knn}

    return GridSearchCV(clf, gridSearchParams[clf_arg], cv=5)
