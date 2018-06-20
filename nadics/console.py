###############################################################################
#                                                                             #
# The current status of our programm should be told to the user during run-   #
# time. Some operations may take a while, so use this file to give hint to    #
# their status.                                                               #
#                                                                             #
# NOTE: Since we do not make use of e.g. a 'verbose' parameter or have a very #
# dynamic output, for the time being we just define our output in functions   #
# for each start of an operation or stop repestively. Thus, this file will be #
# modified soon.                                                              #
#                                                                             #
###############################################################################


import time
import sys
import colors as c
from os import path


def startConfig():
    sys.stdout.write("READING CONFIG FILE...")
    sys.stdout.flush()


def endConfig():
    sys.stdout.write("\t\t\t\tDONE.\n\n")
    sys.stdout.flush()


def endPreprocessed(isPreprocessed):
    sys.stdout.write("IS PREPROCESSED DATA IS STORED ON DISK?")
    sys.stdout.write(("\t\t%s.\n\n")
                     % format(c.colored("TRUE", c.Color.BLUE))) \
        if isPreprocessed \
        else sys.stdout.write(("\t\t%s.\n\n")
                              % format(c.colored("FALSE", c.Color.RED)))
    sys.stdout.flush()


def startLoading():
    sys.stdout.write("LOADING DATA FROM DISK...")
    sys.stdout.flush()
    return time.time()


def endLoading(start_time):
    time_elapsed = time.time() - start_time
    sys.stdout.write("\t\t\tDONE in\t %s seconds.\n\n"
                     % format(round(time_elapsed, 3)))
    sys.stdout.flush()


def startEncoding():
    sys.stdout.write("ENCODING DATA SET...")
    sys.stdout.flush()
    return time.time()


def endEncoding(start_time):
    time_elapsed = time.time() - start_time
    sys.stdout.write("\t\t\t\tDONE in\t %s seconds.\n"
                     % format(round(time_elapsed, 3)))
    sys.stdout.flush()


def startSavingDataset():
    sys.stdout.write("SAVING PREPARED DATA SET TO DISK...")
    sys.stdout.flush()
    return time.time()


def endSavingDataset(start_time):
    time_elapsed = time.time() - start_time
    sys.stdout.write("\t\tDONE in\t %s seconds.\n\n"
                     % format(round(time_elapsed, 3)))
    sys.stdout.flush()


def startNormalizing():
    sys.stdout.write("NORMALIZING DATA SETS...")
    sys.stdout.flush()
    return time.time()


def endNormalizing(start_time):
    time_elapsed = time.time() - start_time
    sys.stdout.write("\t\t\tDONE in\t %s seconds.\n\n"
                     % format(round(time_elapsed, 3)))
    sys.stdout.flush()


def reportDataset(path_train, path_test, xTrain, xTest, xFeatures):
    training_filename = path.splitext(path.basename(path_train))[0]
    testing_filename = path.splitext(path.basename(path_test))[0]

    count_rows_xTraining = xTrain.shape[0]
    count_rows_xTesting = xTest.shape[0]
    count_features = len(xFeatures)

    print("TRAINING SET:\t%s"
          % c.colored(str(training_filename), c.Color.BLUE))
    print("TESTING SET:\t%s\n"
          % c.colored(str(testing_filename), c.Color.BLUE))
    print("TRAINING SIZE:\t%s"
          % c.colored(str(count_rows_xTraining), c.Color.BLUE))
    print("TESTING SIZE:\t%s"
          % c.colored(str(count_rows_xTesting), c.Color.BLUE))
    print("FEATURES:\t%s\n"
          % c.colored(str(count_features), c.Color.BLUE))


def startPrepareImportances():
    sys.stdout.write("PREPARING FEATURE IMPORTANCES...")
    sys.stdout.flush()
    return time.time()


def endPrepareImportances(start_time):
    feature_evaluation = time.time() - start_time
    sys.stdout.write("\t\tDONE in\t %s seconds.\n"
                     % format(round(feature_evaluation, 3)))
    sys.stdout.flush()


def startPlotting():
    sys.stdout.write("SAVING PLOTTED FEATURE IMPORTANCES TO DISK...")
    sys.stdout.flush()
    return time.time()


def endPlotting(start_time):
    time_elapsed = time.time() - start_time
    sys.stdout.write("\tDONE in\t %s seconds.\n\n"
                     % format(round(time_elapsed, 3)))
    sys.stdout.flush()


def startSavingImportances():
    sys.stdout.write("SAVING FEATURE IMPORTANCES TO DISK...")
    sys.stdout.flush()
    return time.time()


def endSavingImportances(start_time):
    time_elapsed = time.time() - start_time
    sys.stdout.write("\t\tDONE in\t %s seconds.\n\n"
                     % format(round(time_elapsed, 3)))
    sys.stdout.flush()


def startClustering():
    sys.stdout.write("CLUSTERING THE DATA SET...")
    sys.stdout.flush()
    return time.time()


def endClustering(start_time, threshold, n_errors):
    time_elapsed = time.time() - start_time
    sys.stdout.write("\t\t\tDONE in\t %s seconds.\n\n"
                     % format(round(time_elapsed, 3)))
    sys.stdout.flush()

    print("THRESHOLD:\t%s"
          % c.colored(str(threshold), c.Color.BLUE))
    print("# ERRORS:\t%s\n"
          % c.colored(str(n_errors), c.Color.BLUE))


def startSampling():
    sys.stdout.write("SAMPLING THE DATA SET...")
    sys.stdout.flush()
    return time.time()


def endSampling(start_time):
    time_elapsed = time.time() - start_time
    sys.stdout.write("\t\t\tDONE in\t %s seconds.\n"
                     % format(round(time_elapsed, 3)))
    sys.stdout.flush()


def startClassification(clf):
    sys.stdout.write("\nClassification algorithm: %s\n\n"
                     % c.colored(clf, c.Color.BLUE))
    sys.stdout.flush()


def startBoosting():
    sys.stdout.write("APPLYING BOOSTING...")
    sys.stdout.flush()


def endBoosting():
    sys.stdout.write("\t\t\t\tDONE.\n\n")
    sys.stdout.flush()


def startTraining():
    sys.stdout.write("TRAINING THE MODEL...")
    sys.stdout.flush()
    return time.time()


def endTraining(start_time):
    time_training = time.time() - start_time
    sys.stdout.write("\t\t\t\tDONE in\t %s seconds.\n"
                     % format(round(time_training, 3)))
    sys.stdout.flush()
    return time_training


def startTesting():
    sys.stdout.write("PREDICTING...")
    sys.stdout.flush()
    return time.time()


def endTesting(start_time):
    time_prediction = time.time() - start_time
    sys.stdout.write("\t\t\t\t\tDONE in\t %s seconds.\n\n"
                     % format(round(time_prediction, 3)))
    sys.stdout.flush()
    return time_prediction


def startWritingResults():
    sys.stdout.write("SAVING RESULTS TO DISK...")
    sys.stdout.flush()
    return time.time()


def endWritingResults(start_time):
    time_elapsed = time.time() - start_time
    sys.stdout.write("\t\tDONE in\t %s seconds.\n\n"
                     % format(round(time_elapsed, 3)))
    sys.stdout.flush()
