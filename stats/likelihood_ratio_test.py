from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.stats import chi2


def likelihood_ratio_test(x_train, y_train, x_test, y_test):
    full_model = LogisticRegression()
    full_model.fit(x_train, y_train)
    log_likelihood_full = -log_loss(y_test, full_model.predict_proba(x_test)[:, 1])

    reduced_model = LogisticRegression()
    reduced_model.fit(x_train[:, :1], y_train)
    log_likelihood_reduced = -log_loss(y_test, reduced_model.predict_proba(x_test[:, :1])[:, 1])

    lr_statistic = 2 * (log_likelihood_full - log_likelihood_reduced)
    degrees_freedom = full_model.coef_.shape[1] - reduced_model.coef_.shape[1]
    p_value = 1 - chi2.cdf(lr_statistic, degrees_freedom)

    print("Likelihood Ratio Test Statistic:", lr_statistic)
    print("Degrees of Freedom:", degrees_freedom)
    print("P-value:", p_value)
