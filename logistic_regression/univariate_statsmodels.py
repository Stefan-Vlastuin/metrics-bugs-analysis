import pandas as pd

from logistic_regression.log_reg import enough_values, apply_logistic_regression
from read_data.read_data import get_dataframe


def univariate(df):
    x = df.iloc[:, 1:-1]  # Features (leave out file names and target variable)
    x = x.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['hasBug']  # Target variable

    # Do univariate logistic regression for each feature separately
    for feature in x.columns:
        if enough_values(df[feature]):
            coefficients, odds_ratios, p_values, pr_squared, accuracy, precision, recall, f1, confusion_matrix = apply_logistic_regression(x[[feature]], y)
            print(feature, coefficients[feature], odds_ratios[feature], p_values[feature], pr_squared, accuracy, precision, recall, f1)
        else:
            print(feature, "has not enough non-zero values")


data = get_dataframe()
univariate(data)
