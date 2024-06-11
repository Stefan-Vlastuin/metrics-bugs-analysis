import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

from logistic_regression.log_reg import apply_logistic_regression


def multivariate(x, y):
    # In the current implementation, impurity is the same as using a field variable
    # Therefore, we delete the impurity metrics, to prevent problems with multicollinearity

    # Remove redundant features here
    if 'MethodImpure' in x.columns:
        del x['MethodImpure']
    if 'MethodRatioImpure' in x.columns:
        del x['MethodRatioImpure']
    if 'LambdaFieldVariable' in x.columns:
        del x['LambdaFieldVariable']
    if 'LambdaSideEffect' in x.columns:
        del x['LambdaSideEffect']

    model = LogisticRegression(class_weight="balanced", max_iter=5000)
    sfs = SequentialFeatureSelector(model, direction="forward", n_features_to_select="auto", tol=1e-4)
    x_new = sfs.fit_transform(x, y)
    selected_features = x.columns[sfs.get_support()]

    x_new = pd.DataFrame(x_new, columns=selected_features)

    return selected_features, apply_logistic_regression(x_new, y)
