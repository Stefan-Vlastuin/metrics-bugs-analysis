import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

from read_data.read_data import get_dataframe


def univariate_all(df):
    x = df.iloc[:, 1:-1]  # Features (leave out file names and target variable)
    x = x.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['hasBug']  # Target variable

    # Do univariate logistic regression for each feature separately
    for feature in x.columns:
        if enough_values(df[feature]):
            coefficient, odds_ratio, p_value, pr_squared, accuracy, precision, recall, f1 = univariate(x[[feature]], y)
            print(feature, coefficient, odds_ratio, p_value, pr_squared, accuracy, precision, recall, f1)
        else:
            print(feature, "has not enough non-zero values")


def enough_values(v):
    return v.where(lambda x: x > 0).count() >= 10


def univariate(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # TODO: use k-fold cross validation
    # TODO: balance the dataset

    # Add constants (for intercept)
    x_train_const = sm.add_constant(x_train)
    x_test_const = sm.add_constant(x_test)

    model = sm.Logit(y_train, x_train_const)
    model_sm = model.fit(disp=0, maxiter=100)

    # Model properties
    p_value = model_sm.pvalues.iloc[1]
    pr_squared = model_sm.prsquared  # Use pseudo R-squared because that fits better with logistic regression
    coefficient = model_sm.params.iloc[1]
    odds_ratio = np.exp(coefficient)

    # Prediction metrics
    y_pred_prob = model_sm.predict(x_test_const)
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)

    return coefficient, odds_ratio, p_value, pr_squared, accuracy, precision, recall, f1


data = get_dataframe()
univariate_all(data)
