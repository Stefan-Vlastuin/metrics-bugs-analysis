import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.stats import chi2
from sklearn.model_selection import train_test_split

from read_data.read_data import prepare_data, read_data, get_dataframe


def likelihood_ratio_test2(features_alternate, labels, lr_model, features_null=None):
    """
    Compute the likelihood ratio test for a model trained on the set of features in
    `features_alternate` vs a null model.  If `features_null` is not defined, then
    the null model simply uses the intercept (class probabilities).  Note that
    `features_null` must be a subset of `features_alternative` -- it can not contain
    features that are not in `features_alternate`.
    Returns the p-value, which can be used to accept or reject the null hypothesis.
    """
    labels = np.array(labels)
    features_alternate = np.array(features_alternate)

    if features_null:
        features_null = np.array(features_null)

        if features_null.shape[1] >= features_alternate.shape[1]:
            return -1

        lr_model.fit(features_null, labels)
        null_prob = lr_model.predict_proba(features_null)[:, 1]
        df = features_alternate.shape[1] - features_null.shape[1]
    else:
        null_prob = sum(labels) / float(labels.shape[0]) * \
                    np.ones(labels.shape)
        df = features_alternate.shape[1]

    lr_model.fit(features_alternate, labels)
    alt_prob = lr_model.predict_proba(features_alternate)

    alt_log_likelihood = -log_loss(labels,
                                   alt_prob,
                                   normalize=False)
    null_log_likelihood = -log_loss(labels,
                                    null_prob,
                                    normalize=False)

    G = 2 * (alt_log_likelihood - null_log_likelihood)
    p_value = chi2.sf(G, df)

    return p_value


def likelihood_ratio_test3(x, y):
    full_model = LogisticRegression(class_weight='balanced')
    full_model.fit(x, y)
    log_likelihood_full = -log_loss(y, full_model.predict_proba(x)[:, 1], normalize=False)

    y_pred = full_model.predict(x)
    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)
    f1 = metrics.f1_score(y, y_pred)
    print(accuracy, precision, recall, f1)

    reduced_model = LogisticRegression(class_weight='balanced')
    if x.shape[1] == 1:
        # Only one feature, reduced model is the only-intercept model
        x_train_reduced = np.ones((x.shape[0], 1))
    else:
        # TODO: when doing multivariate regression, we need to be able to choose which feature to ignore
        x_train_reduced = x[['Cohesion']]

    reduced_model.fit(x_train_reduced, y)
    #  log_likelihood_reduced = -log_loss(y_test, reduced_model.predict_proba(x_test_reduced)[:, 1])
    null_prob = sum(y) / float(y.shape[0]) * np.ones(y.shape)
    log_likelihood_reduced = -log_loss(y, null_prob, normalize=False)

    lr_statistic = 2 * (log_likelihood_full - log_likelihood_reduced)
    p_value = 1 - chi2.cdf(lr_statistic, 1)  # Degrees of freedom is 1, since we always ignore 1 metric

    return p_value


def likelihood_ratio_test(x_train, x_test, y_train, y_test):
    print(x_train)
    print(y_train)

    full_model = LogisticRegression(class_weight='balanced')
    full_model.fit(x_train, y_train)
    log_likelihood_full = -log_loss(y_test, full_model.predict_proba(x_test)[:, 1])

    y_pred = full_model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    print(accuracy, precision, recall, f1)

    reduced_model = LogisticRegression(class_weight='balanced')
    if x_train.shape[1] == 1:
        # Only one feature, reduced model is the only-intercept model
        x_train_reduced = np.ones((x_train.shape[0], 1))
        x_test_reduced = np.ones((x_test.shape[0], 1))
    else:
        # TODO: when doing multivariate regression, we need to be able to choose which feature to ignore
        print(x_train)
        x_train_reduced = x_train[['Cohesion']]
        #  x_train_reduced = x_train[:, :1]
        x_test_reduced = x_test[['Cohesion']]

    reduced_model.fit(x_train_reduced, y_train)
    log_likelihood_reduced = -log_loss(y_test, reduced_model.predict_proba(x_test_reduced)[:, 1])
    #  null_prob = sum(y_train) / float(y_train.shape[0]) * np.ones(y_test.shape)
    #  log_likelihood_reduced = -log_loss(y_test, null_prob)
    #print(null_prob)
    print(log_likelihood_reduced)

    # y_pred = reduced_model.predict(x_test)
    # print(y_pred)
    # accuracy = metrics.accuracy_score(y_test, y_pred)
    # precision = metrics.precision_score(y_test, y_pred)
    # recall = metrics.recall_score(y_test, y_pred)
    # f1 = metrics.f1_score(y_test, y_pred)
    # print(accuracy, precision, recall, f1)

    lr_statistic = 2 * (log_likelihood_full - log_likelihood_reduced)
    p_value = 1 - chi2.cdf(lr_statistic, 1)  # Degrees of freedom is 1, since we always ignore 1 metric

    return p_value


data = get_dataframe()
#  data = pd.DataFrame([['A', 100, 1], ['B', 80, 1], ['C', 10, 0], ['D', 5, 0], ['E', 200, 1], ['F', 10, 0], ['F', 10, 0], ['F', 10, 0], ['F', 10, 0], ['F', 10, 0], ['F', 10, 0], ['F', 10, 0], ['F', 10, 0], ['F', 10, 0], ['G', 100, 1], ['G', 100, 1], ['G', 100, 1], ['G', 100, 1]], columns=['FileName', 'Complexity', 'hasBug'])
x = data.iloc[:, 1:-1]
x = x[['Cohesion', 'Complexity']]
y = data['hasBug']
a, b, c, d = train_test_split(x, y, test_size=0.3, random_state=0)
p = likelihood_ratio_test(a, b, c, d)
print("p", p)


p2 = likelihood_ratio_test2(x, y, LogisticRegression(class_weight='balanced'))
print("test2", p2)
p3 = likelihood_ratio_test3(x, y)
print("test3", p3)
