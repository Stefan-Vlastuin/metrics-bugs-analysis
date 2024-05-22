import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

from read_data.read_data import read_data, prepare_data


def univariate_all(df):
    x = df.iloc[:, 1:-1]  # Features (leave out file names and target variable)
    y = df['hasBug']  # Target variable

    # Do univariate logistic regression for each feature separately
    for feature in x.columns:
        accuracy, precision, recall, f1, r_squared = univariate(x[[feature]], y)
        print(feature, accuracy, precision, recall, f1, r_squared)


def log_likelihood(y_true, y_pred_prob):
    epsilon = 1e-15
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
    return np.sum(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))


def univariate(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    model = LogisticRegression(max_iter=5000, class_weight='balanced')
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    # y_pred_prob = model.predict_proba(x_test)[:, 1]
    # # Calculate McFadden's R-squared
    # ll_model = log_likelihood(y_test, y_pred_prob)
    # print("ll_model", ll_model)
    # y_null_prob = np.full_like(y_test, y_train.mean())
    # ll_null = log_likelihood(y_test, y_null_prob)
    # print("ll_null", ll_null)
    # r_squared = 1 - (ll_model / ll_null)

    print(y_train)

    # Calculate McFadden's R-squared
    x_train = x_train.apply(pd.to_numeric, errors='coerce')
    x_train = x_train.fillna(0)
    y_train = y_train.apply(pd.to_numeric, errors='coerce')
    y_train = y_train.fillna(0)
    x_train_const = sm.add_constant(x_train)
    model_sm = sm.Logit(y_train, x_train_const).fit(disp=0)
    llf = model_sm.llf
    llnull = model_sm.llnull
    r_squared = 1 - (llf / llnull)

    accuracy = accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1, r_squared


data = prepare_data(read_data())
univariate_all(pd.DataFrame(data))
