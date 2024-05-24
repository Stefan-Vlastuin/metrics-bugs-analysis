from logistic_regression.log_reg import apply_logistic_regression


def multivariate(x, y):
    # TODO: use k-best features to select features
    return apply_logistic_regression(x, y)
