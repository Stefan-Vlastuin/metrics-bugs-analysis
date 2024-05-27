import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

from logistic_regression.log_reg import apply_logistic_regression


def multivariate(x, y):
    # In the current implementation, impurity is the same as using a field variable
    # Therefore, we delete the impurity metrics, to prevent problems with multicollinearity
    # TODO: remove when the implementations of these metrics have been improved
    if 'MethodImpure' in x.columns:
        del x['MethodImpure']
    if 'MethodRatioImpure' in x.columns:
        del x['MethodRatioImpure']

    selector = SelectKBest(f_regression, k=7)   # Number of baseline metrics; can substitute them for new metrics
    x_new = selector.fit_transform(x, y)

    selected_features_indices = selector.get_support(indices=True)
    selected_features = x.columns[selected_features_indices]

    x_new = pd.DataFrame(x_new, columns=selected_features)

    return selected_features, apply_logistic_regression(x_new, y)
