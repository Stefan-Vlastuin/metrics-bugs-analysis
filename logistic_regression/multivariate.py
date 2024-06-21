import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

from logistic_regression.log_reg import apply_logistic_regression


def multivariate(x, y):
    # In the current implementation, impurity is the same as using a field variable
    # Therefore, we delete the impurity metrics, to prevent problems with multicollinearity
    if 'MethodImpure' in x.columns:
        del x['MethodImpure']
    if 'MethodRatioImpure' in x.columns:
        del x['MethodRatioImpure']

    # Remove other redundant features here
    # if 'LambdaFieldVariable' in x.columns:
    #     del x['LambdaFieldVariable']
    # if 'LambdaSideEffect' in x.columns:
    #     del x['LambdaSideEffect']
    # if 'LambdaComplexity' in x.columns:
    #     del x['LambdaComplexity']
    # if 'Complexity' in x.columns:
    #     del x['Complexity']
    # if 'LambdaCount' in x.columns:
    #     del x['LambdaCount']
    # if 'LambdaLines' in x.columns:
    #     del x['LambdaLines']
    # if 'MaxStreamOperations' in x.columns:
    #     del x['MaxStreamOperations']

    model = LogisticRegression(class_weight="balanced", max_iter=10000)
    sfs = SequentialFeatureSelector(model, direction="backward", n_features_to_select="auto", tol=-1e-4, n_jobs=-1)
    x_new = sfs.fit_transform(x, y)
    selected_features = x.columns[sfs.get_support()]

    x_new = pd.DataFrame(x_new, columns=selected_features)

    return selected_features, apply_logistic_regression(x_new, y)
