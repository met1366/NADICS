import argparse
import models
import ConfigParser

def parseArgs():
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter)
    parser.add_argument("classification",
                        metavar = "C",
                        choices = models.classifications,
                        default = "2",
                        help = "Binary- or Multi-Classificaion."
                        )
    parser.add_argument("classifiers",
                        metavar = "M",
                        choices = (models.classifiers.keys() +
                                   models.novelty_outliers.keys()),
                        nargs = "+",
                        help = "Models to choose for training."
                        )
    parser.add_argument("--sampling", "-s",
                        metavar = "SAMPLER",
                        choices = models.samplings.keys(),
                        help = "Over- or undersampling method to choose."
                        )
    parser.add_argument("--boosting", "-b",
                        metavar = "BOOSTER",
                        choices = models.boostings.keys(),
                        help = "Boosting method to choose."
                        )
    parser.add_argument("--training", "-x",
                        metavar = "SIZE",
                        type = float,
                        default = 0.30,
                        help = "Partial training size between in (0,1]."
                        )
    parser.add_argument("--testing", "-y",
                        metavar = "SIZE",
                        type = float,
                        default = 0.30,
                        help = "Partial testing size between in (0,1]."
                        )
    parser.add_argument("--encoding", "-e",
                        action = "store_true",
                        help = "Use One-Hot-Encoding for string features."
                        )
    parser.add_argument("--overwrite", "-w",
                        action = "store_true",
                        help = "Overwrite prepared data stored on disk."
                        )
    parser.add_argument("--print_importances", "-i",
                        action = "store_true",
                        help = "Print the importances on the console."
                        )
    parser.add_argument("--plot_importances", "-p",
                        action = "store_true",
                        help = "Plot the importances using Extra-Trees."
                        )
    parser.add_argument("--write_importances", "-v",
                        action = "store_true",
                        help = "Write the importances to disk using Extra-Trees."
                        )
    parser.add_argument("--write_results", "-r",
                        action = "store_true",
                        help = "Write the results to disk."
                        )
    parser.add_argument("--grid_search", "-g",
                        action = "store_true",
                        help = "Run grid search on the specified models."
                        )
    parser.add_argument("--time", "-t",
                        metavar = "SECONDS",
                        type = int,
                        default = None,
                        help = "Number of seconds to wait until termination."
                        )
    return parser.parse_args()

def parseData(path_config, classification):
    config = ConfigParser.ConfigParser(allow_no_value=True)
    config.read(path_config)

    fullHeader = config.get("DataPreparation", "fullHeader").splitlines()
    featureHeader = config.get("DataPreparation", "featureHeader").splitlines()
    if (classification == "2"):
        labelHeader = config.get("DataPreparation", "binaryLabels")
    elif (classification == "n"):
        labelHeader = config.get("DataPreparation", "multiLabels")

    stringFeatures = config.get("DataPreparation",
                                "stringFeatures").splitlines()

    return fullHeader, \
           featureHeader, \
           labelHeader, \
           stringFeatures
