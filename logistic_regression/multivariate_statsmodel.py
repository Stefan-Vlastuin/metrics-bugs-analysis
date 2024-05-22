import pandas as pd

from logistic_regression.log_reg import enough_values, apply_logistic_regression
from read_data.read_data import get_dataframe


def multivariate(df):
    x = df.iloc[:, 1:-1]  # Features (leave out file names and target variable)
    x = x.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['hasBug']  # Target variable

    x = x.loc[:, x.apply(enough_values)]  # Filter out features with few values

    coefficients, odds_ratios, p_values, pr_squared, accuracy, precision, recall, f1, confusion_matrix = apply_logistic_regression(x, y)
    print(accuracy, precision, recall, f1)
    print(confusion_matrix)


data = get_dataframe()
multivariate(data)
