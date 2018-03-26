from os import path         # Comfortable OS path manipulation
import sys                  # Console printing
import pandas

def checkIniPaths(train, test, config):
    if (not (path.isfile(train))):
        return 1
    if (not (path.isfile(test))):
        return 2
    if (not (path.isfile(config))):
        return 3

    return 0

def preprocessed(prep_train, prep_test, prep_label_train, prep_label_test):
    isPreprocessed = True if (path.exists(prep_train)
                              and path.exists(prep_test)
                              and path.exists(prep_label_train)
                              and path.exists(prep_label_test)) \
        else False
    return isPreprocessed

def oneHotEncode(dataset, featuresToEncode):
    newFeatures = []
    for feature in featuresToEncode:
        columnToEncode = dataset[feature]
        # This is the actual One-hot-Encoding method using sklearn
        dummies = pandas.get_dummies(columnToEncode)
        insertionIndex = dataset.columns.get_loc(feature)
        encodedFeatures = list(dummies)
        # For convenience in reading and debugging we want to insert features
        # in the same order we create the binarized data set
        for newFeature in reversed(encodedFeatures):
            dataset.insert(insertionIndex,
                           newFeature,
                           dummies[newFeature],
                           allow_duplicates=False)
        newFeatures.extend(encodedFeatures)
        dataset.drop(feature, axis=1, inplace=True)
    return dataset, newFeatures

def encode(data_train, data_test, features):
    # Encoding all features containing strings via One-Hot-Encoding
    dataset_training, training_features = oneHotEncode(data_train,
                                                 features)
    dataset_testing, _ = oneHotEncode(data_test,
                                      features)
    return dataset_training, \
           dataset_testing, \
           training_features

def commonFeatures(encodingEnabled, dataset, features):
    if (encodingEnabled):
        # Selecting features which are in the encoded and sampled training
        # dataset as well as in the testing one
        commonFeatures = filter(lambda x: x in features,
                                list(dataset))
    else:
        commonFeatures = features
    
    return commonFeatures

def samplingRatio(data_train, data_test, size_train, size_test):
    return data_train.sample(frac=size_train), \
           data_test.sample(frac=size_test)

def load(path_training, path_testing, header):
    # Reading and cutting (plus shiffling) the whole training dataset to a
    # given percentage
    dataset_training = pandas.read_csv(path_training, header=None,
                                       names=header, low_memory=False)
    dataset_testing = pandas.read_csv(path_testing, header=None,
                                      names=header, low_memory=False)

    return dataset_training, dataset_testing

def loadPreprocessed(prep_train, prep_test, prep_label_train, prep_label_test):
    return pandas.read_pickle(prep_train), \
           pandas.read_pickle(prep_test), \
           pandas.read_pickle(prep_label_train), \
           pandas.read_pickle(prep_label_test)

def saveToPickle(xTrain, path_xTrain,
                 xTest, path_xTest,
                 yTrain, path_yTrain,
                 yTest, path_yTest):
    xTrain.to_pickle(path_xTrain)
    xTest.to_pickle(path_xTest)
    yTrain.to_pickle(path_yTrain)
    yTest.to_pickle(path_yTest)
