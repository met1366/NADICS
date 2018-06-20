###############################################################################
#                                                                             #
# Given a certain boosting we apply it to our current model.                  #
#                                                                             #
###############################################################################


def boosting(model, clf):
    return model.set_params(base_estimator=clf)
