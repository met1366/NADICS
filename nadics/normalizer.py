from sklearn.preprocessing import MinMaxScaler

def normalize(xTrain, xTest):
    # Normalize the training dataset and scale the testing set accordingly to
    # the testing one
    # INFO: Do not use fit MinMaxScaler for the testing data set again, use the
    # one based on the training set!
    min_max_scaler = MinMaxScaler()
    
    return min_max_scaler.fit_transform(xTrain), \
        min_max_scaler.transform(xTest)
