import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.stats import chi2
from sklearn.model_selection import train_test_split

from read_data.read_data import get_dataframe


def likelihood_ratio_test(x_train, x_test, y_train, y_test):
    full_model = LogisticRegression(class_weight='balanced')
    full_model.fit(x_train, y_train)
    log_likelihood_full = -log_loss(y_test, full_model.predict_proba(x_test)[:, 1])

    reduced_model = LogisticRegression(class_weight='balanced')
    if x_train.shape[1] == 1:
        # Only one feature, reduced model is the only-intercept model
        x_train_reduced = np.ones((x_train.shape[0], 1))
        x_test_reduced = np.ones((x_test.shape[0], 1))
    else:
        # TODO: when doing multivariate regression, we need to be able to choose which features to ignore and keep
        x_train_reduced = x_train[['Cohesion']]
        x_test_reduced = x_test[['Cohesion']]

    reduced_model.fit(x_train_reduced, y_train)
    log_likelihood_reduced = -log_loss(y_test, reduced_model.predict_proba(x_test_reduced)[:, 1])

    lr_statistic = 2 * (log_likelihood_full - log_likelihood_reduced)
    p_value = 1 - chi2.cdf(lr_statistic, 1)  # Degrees of freedom is 1, since we always ignore 1 metric

    return p_value


data = get_dataframe()
x = data.iloc[:, 1:-1]
x = x[['Cohesion', 'Complexity']]
y = data['hasBug']
a, b, c, d = train_test_split(x, y, test_size=0.3, random_state=0)
p = likelihood_ratio_test(a, b, c, d)
print("P-value", p)
