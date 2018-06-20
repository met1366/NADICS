###############################################################################
#                                                                             #
# The main function includes our whole workflow in terms of machine learning. #
# Each step is visually seperated by several lines of comments and functions  #
# that might take a while to execute, should give a hint to the user about    #
# their status via the 'console' module.                                      #
#                                                                             #
# NOTE: In terms of debugging we will sanity checks later and some test       #
# functions to make sure the installation was successful.                     #
#                                                                             #
###############################################################################

from os import path
import ConfigParser
import models
import parser
import console
import report
import preprocessor as pre
import normalizer as norm
import selector as sel
import sampling as sam
import boosting as boo
import learning as lea
import clustering as clu
import version
import time
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)
# Turning off UndefinedMetricWarning
warnings.filterwarnings("ignore")


def intro():
    print("#####################################################")
    print("#                                                   #")
    print("#          NIDS MACHINE LEARNING ENGINE             #")
    print("#            ..:::: ICEFALL " + version.VERSION + " ::::..\
              #")
    print("#                                                   #")
    print("#####################################################\n")


def outro():
    print("#####################################################")
    print("#                                                   #")
    print("#     SHUTTING MACHINE LEARNING ENGINE DOWN...      #")
    print("#                                                   #")
    print("#####################################################\n")


def run_main():
    main()


def main():
    intro()

    #####################################################
    #                                                   #
    # ARGUMENT PARSING                                  #
    #                                                   #
    #####################################################

    args = parser.parseArgs()

    classification = args.classification
    clfs = args.classifiers
    clustering = models.clustering[args.clustering] if args.clustering else None
    sampling = models.samplings[args.sampling] if args.sampling else None
    boosting = models.boostings[args.boosting] if args.boosting else None
    training_size = args.training
    testing_size = args.testing
    encodingEnabled = args.encoding
    generationForced = args.overwrite
    printImportances = args.print_importances
    plotImportances = args.plot_importances
    writeImportances = args.write_importances
    writeResults = args.write_results
    gridSearch = args.grid_search
    timeWaiting = args.time

    #####################################################
    #                                                   #
    # READ INI FILE                                     #
    #                                                   #
    #####################################################

    config = ConfigParser.ConfigParser()
    config.read("config/defaults.ini")

    # NOTE: Further default values whith might be used
    # random_state = config.get("defaults", "random_state")
    # time_until_kill = config.get("defaults", "time_until_kill")
    # verbosity = config.get("defaults", "verbosity")
    floating_precision = int(config.get("defaults", "floating_precision"))

    path_data = config.get("data", "path_data")
    path_pre = config.get("data", "path_pre")
    path_layout = config.get("data", "path_layout")
    path_plots = config.get("data", "path_plots")
    path_importances = config.get("data", "path_importances")
    path_score = config.get("data", "path_score")

    training_path = path.join(
        path.realpath(path_data),
        config.get("data", "training"))
    training_filename = path.splitext(path.basename(training_path))[0]
    testing_path = path.join(
        path.realpath(path_data),
        config.get("data", "testing"))
    # NOTE: Might be useful for debugging purposes
    # testing_filename = path.splitext(path.basename(testing_path))[0]
    layout_path = path.join(
        path.realpath(path_layout),
        config.get("data", "layout"))
    xTraining_pre = path.join(
        path.realpath(path_pre),
        config.get("data", "xTraining_pre"))
    xTesting_pre = path.join(
        path.realpath(path_pre),
        config.get("data", "xTesting_pre"))
    yTraining_pre = path.join(
        path.realpath(path_pre),
        config.get("data", "yTraining_pre"))
    yTesting_pre = path.join(
        path.realpath(path_pre),
        config.get("data", "yTesting_pre"))

    plots_filetype = config.get("plots", "filetype")
    time_start = time.strftime("%Y%m%d-%H%M")

    if(encodingEnabled):
        encoding_prefix = config.get("encoding", "prefix")
    else:
        encoding_prefix = ""

    #####################################################
    #                                                   #
    # CHECKING THE GIVEN PATHS                          #
    #                                                   #
    #####################################################

    err = pre.checkIniPaths(training_path,
                            testing_path,
                            layout_path)
    if(err):
        return

    #####################################################
    #                                                   #
    # READ DATASET CONFIG FILE                          #
    #                                                   #
    #####################################################

    console.startConfig()
    fullHeader, featureHeader, labelHeader, stringFeatures = parser.parseData(
        layout_path,
        classification)
    console.endConfig()

    #####################################################
    #                                                   #
    # CHECK FOR STORED DATA                             #
    #                                                   #
    #####################################################

    isPreprocessed = pre.preprocessed(xTraining_pre,
                                      xTesting_pre,
                                      yTraining_pre,
                                      yTesting_pre)
    console.endPreprocessed(isPreprocessed)

    #####################################################
    #                                                   #
    # LOAD DATASETS                                     #
    #                                                   #
    #####################################################

    if (isPreprocessed and not generationForced):
        runtime_start = console.startLoading()
        xTraining, xTesting, yTraining, yTesting = pre.loadPreprocessed(
            xTraining_pre,
            xTesting_pre,
            yTraining_pre,
            yTesting_pre)
        console.endLoading(runtime_start)
    else:
        runtime_start = console.startEncoding()
        training, testing = pre.load(training_path,
                                     testing_path,
                                     fullHeader)

        #####################################################
        #                                                   #
        # ONE-HOT-ENCODING OF STRINGS                       #
        #                                                   #
        #####################################################

        if (encodingEnabled):
            training, training, newFeatures = pre.encode(training,
                                                         testing,
                                                         stringFeatures)
            featureHeader.extend(newFeatures)

        #####################################################
        #                                                   #
        # SAMPLING RATIO PROCESS                            #
        #                                                   #
        #####################################################

        training, testing = pre.samplingRatio(training,
                                              testing,
                                              training_size,
                                              testing_size)

        #####################################################
        #                                                   #
        # SELECTION OF FEATURE COLUMNS                      #
        #                                                   #
        #####################################################

        commonFeatures = pre.commonFeatures(encodingEnabled,
                                            testing,
                                            featureHeader)

        xTraining = training[commonFeatures].astype(float)
        xTesting = testing[commonFeatures].astype(float)
        yTraining = training[labelHeader]
        yTesting = testing[labelHeader]

        console.endEncoding(runtime_start)

        #####################################################
        #                                                   #
        # SAVING CREATED DATASET                            #
        #                                                   #
        #####################################################

        runtime_start = console.startSavingDataset()

        # Writing prepared files
        pre.saveToPickle(xTraining, xTraining_pre,
                         xTesting, xTesting_pre,
                         yTraining, yTraining_pre,
                         yTesting, yTesting_pre)

        console.endSavingDataset(runtime_start)

    xFeatures = list(xTraining)

    console.reportDataset(training_path,
                          testing_path,
                          xTraining,
                          xTesting,
                          xFeatures)

    #####################################################
    #                                                   #
    # NORMALIZATION OF DATA SETS                        #
    #                                                   #
    #####################################################

    runtime_start = console.startNormalizing()
    xTraining, xTesting = norm.normalize(xTraining, xTesting)
    console.endNormalizing(runtime_start)

    #####################################################
    #                                                   #
    # SHOW FEATURE IMPORTANCES WITH FORESTS OF TREES    #
    #                                                   #
    #####################################################

    if (printImportances or plotImportances or writeImportances):
        runtime_start = console.startPrepareImportances()
        importances, indices, std = sel.importances(xTraining, yTraining)
        console.endPrepareImportances(runtime_start)

    if (printImportances):
        sel.printImportances(xTraining, xFeatures, importances, indices)

    if (plotImportances):
        runtime_start = console.startPlotting()

        filestr = path_plots + \
            encoding_prefix + \
            training_filename + "_" + \
            time_start + \
            plots_filetype

        sel.plotImportances(xTraining,
                            xFeatures,
                            importances,
                            indices,
                            std,
                            filestr)
        console.endPlotting(runtime_start)

    if (writeImportances):
        runtime_start = console.startSavingImportances()

        filestr = path_importances + \
            encoding_prefix + \
            training_filename + "_" + \
            time_start + \
            plots_filetype

        sel.saveImportances(xTraining,
                            xFeatures,
                            importances,
                            indices,
                            filestr)
        console.endSavingImportances(runtime_start)

    #####################################################
    #                                                   #
    # CLUSTERING                                        #
    #                                                   #
    #####################################################

    if not (clustering is None):
        runtime_start = console.startClustering()
        y_pred, scores_pred = clu.outliers(clustering,
                                           xTraining,
                                           args.clustering)
        threshold = clu.threshold(scores_pred, clustering.contamination)
        n_errors = clu.nErrors(y_pred, yTraining.values)
        console.endClustering(runtime_start, threshold, n_errors)

    #####################################################
    #                                                   #
    # OVER- OR UNDERSAMPLING                            #
    #                                                   #
    #####################################################

    if not (sampling is None):
        runtime_start = console.startSampling()
        xTraining, yTraining = sam.sampling(sampling,
                                            xTraining,
                                            yTraining)
        console.endSampling(runtime_start)

    #####################################################
    #                                                   #
    # STARTING CLASSIFICATION                           #
    #                                                   #
    #####################################################

    results = []

    for clf_arg in clfs:
        console.startClassification(clf_arg)
        clf = models.classifiers[clf_arg]

        #####################################################
        #                                                   #
        # APPLYING BOOSTER                                  #
        #                                                   #
        #####################################################

        if not (boosting is None):
            console.startBoosting()
            clf = boo.boosting(boosting, clf)
            console.endBoosting()

        #####################################################
        #                                                   #
        # TRAINING                                          #
        #                                                   #
        #####################################################

        if (gridSearch):
            clf = lea.gridSearch(clf)

        runtime_start = console.startTraining()
        clf = lea.training(clf,
                           xTraining,
                           yTraining,
                           timeWaiting)
        time_training = console.endTraining(runtime_start)

        if (gridSearch):
            report.printGridSearch(clf)

        #####################################################
        #                                                   #
        # PREDICTION                                        #
        #                                                   #
        #####################################################

        runtime_start = console.startTesting()
        yPrediction = lea.testing(clf, xTesting, timeWaiting)
        time_testing = console.endTesting(runtime_start)

        #####################################################
        #                                                   #
        # REPORT                                            #
        #                                                   #
        #####################################################

        result = report.printAll(clf_arg,
                                 time_training,
                                 time_testing,
                                 yPrediction,
                                 yTesting)
        results.append(result)

        if (writeResults):
            runtime_start = console.startWritingResults()
            filestr = path_score + \
                encoding_prefix + \
                clf_arg + "_" + \
                "train_" + str(training_size) + "_" + \
                "test_" + str(testing_size) + "_" + \
                training_filename + "_" + \
                time_start

            report.save(filestr, result)
            console.endWritingResults(runtime_start)
            return

    #####################################################
    #                                                   #
    # CLEANING UP                                       #
    #                                                   #
    #####################################################

    if results:
        report.printSummary(clfs, results, floating_precision)
    outro()


if __name__ == "__main__":
    run_main()
