###############################################################################
#                                                                             #
# Printing short or full reports of the resutls. Feel free to edit the output #
# to your needs.                                                              #
#                                                                             #
###############################################################################

import colors as c
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support


def printAll(clf, time_training, time_testing, yPrediction, groundTruth):
    accuracyScore = accuracy_score(y_pred=yPrediction, y_true=groundTruth)
    precision, recall, f1Score, support = \
        precision_recall_fscore_support(y_pred=yPrediction,
                                        y_true=groundTruth,
                                        average="weighted")
    report = classification_report(y_pred=yPrediction,
                                   y_true=groundTruth)

    clf_result = [clf,
                  time_training,
                  time_testing,
                  accuracyScore,
                  precision,
                  f1Score,
                  recall,
                  support]

    report = classification_report(y_pred=yPrediction,
                                   y_true=groundTruth)
    printAccuracy(accuracyScore)
    printReport(report)

    return clf_result


def printAccuracy(accuracyScore):
    print ("Accuracy Score: %s"
           % c.colored(str(round(accuracyScore, 3)), c.Color.BLUE))


def printReport(report):
    print ("Classification report: \n" + report)


def maxStrLen(*strLists):
    count = 0
    for strList in strLists:
        for string in strList:
            maxLength = len(string)
            count = maxLength if maxLength > count else count
    return count


def printSummary(classifiers, results, floating_precision):
    fields = ["",
              "Time training [s]",
              "Time prediction [s]",
              "Accuracy score",
              "Weighted precision",
              "Weighted f1 score",
              "Weighted recall",
              "Weighted support"]
    delimiter = "|"
    minCellLen = maxStrLen(fields)
    lineSeperator = "+-" + minCellLen * "-" + "-+"

    # Adding the named headers to the first column
    for i in range(0, len(fields)):
        header = fields[i]
        extra_spaces = minCellLen - len(header)
        fields[i] = "%s %s%s %s" % (delimiter,
                                    header,
                                    extra_spaces * " ",
                                    delimiter)

    # Adding missing spaces for a convenient layout to the first column
    minCellLen = maxStrLen(classifiers)
    for clf in classifiers:
        lineSeperator += "-" + minCellLen * "-" + "-+"
        extra_spaces = minCellLen - len(clf)
        fields[0] += " %s%s %s" % (clf, extra_spaces * " ", delimiter)

    # Extracting the best and worst results of all algorithms into an array of
    # [[best_0, worst_0],...,[best_n, worst_n]]
    best_worst_results = []
    for i in range(1, len(fields) - 1):
        row = ([clf_result[i] for clf_result in results])
        if (i is 1) or (i is 2):
            best_worst_results.append([min(row), max(row)])
        else:
            best_worst_results.append([max(row), min(row)])

    for clf_result in results:
        for i in range(1, len(fields)):
            value = clf_result[i]
            # Standard color for printing
            color = c.Color.BLACK
            # Do not change the order or "if" into "else if" since the order
            # prevents two equal values shown as "worst" instead of "best"
            # in this way both values will be shown as "best"
            if not (i is len(fields) - 1):
                if value is best_worst_results[i - 1][1]:
                    color = c.Color.RED
                if value is best_worst_results[i - 1][0]:
                    color = c.Color.GREEN
                value = round(value, floating_precision)
            extra_spaces = minCellLen - len(str(value))
            formatted_value = c.colored(str(value), color)
            fields[i] += " %s%s %s" % (formatted_value,
                                       extra_spaces * " ",
                                       delimiter)

    # Print the whole table with the results of all classifiers
    # +---------------------...------------+
    # |         | CLF_0   | ...  | CLF_K   |
    # +---------------------...------------+
    # | FIELD_0 | RSLT_00 | ...  | RSLT_K0 |
    # | ...     | ...     | ...  | ...     |
    # | FIELD_K | RSLT_0L | ...  | RSLT_KL |
    # +---------------------...------------+
    print lineSeperator
    for i in range(0, len(fields)):
        print fields[i]
        if i is 0:
            print lineSeperator
    print lineSeperator + "\n"


def printGridSearch(clf):
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means,
                                 stds,
                                 clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


def save(filename, result):
    # result_string = ''.join(str(e) for e in result)

    fi = open(filename, "w")
    fi.write("%s\n" % result)
