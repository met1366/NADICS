def boosting(model, clf):
    return model.set_params(base_estimator=clf) 
