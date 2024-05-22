import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from read_data.read_data import read_data, prepare_data
from stats.likelihood_ratio_test import likelihood_ratio_test


def univariate_all(df):
    x = df.iloc[:, 1:-1]  # Features (leave out file names and target variable)
    y = df['hasBug']  # Target variable

    # Do univariate logistic regression for each feature separately
    for feature in x.columns:
        accuracy, precision, recall, f1, p_value = univariate(x[[feature]], y)
        print(feature, accuracy, precision, recall, f1, p_value)


def univariate(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    model = LogisticRegression(max_iter=5000, class_weight='balanced')
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    p_value = likelihood_ratio_test(x_train, x_test, y_train, y_test)
    return accuracy, precision, recall, f1, p_value


data = prepare_data(read_data())
univariate_all(pd.DataFrame(data))
