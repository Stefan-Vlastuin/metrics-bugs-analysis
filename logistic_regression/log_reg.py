import numpy as np
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def apply_logistic_regression(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # TODO: use k-fold cross validation

    # Oversample classes with bugs
    smote = SMOTE(random_state=0)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    # Add constants (for intercept)
    x_train_const = sm.add_constant(x_train)
    x_test_const = sm.add_constant(x_test)

    model = sm.Logit(y_train, x_train_const)
    model_sm = model.fit(disp=0, maxiter=100)

    # Model properties
    p_values = model_sm.pvalues
    pr_squared = model_sm.prsquared  # Use pseudo R-squared because that fits better with logistic regression
    coefficients = model_sm.params
    odds_ratios = np.exp(coefficients)

    # Prediction metrics
    y_pred_prob = model_sm.predict(x_test_const)
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    return coefficients, odds_ratios, p_values, pr_squared, accuracy, precision, recall, f1, confusion_matrix


def enough_values(v):
    return v.where(lambda x: x > 0).count() >= 10
