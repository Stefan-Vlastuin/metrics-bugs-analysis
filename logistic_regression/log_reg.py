import numpy as np
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

PROB_THRESHOLD = 0.5


def apply_logistic_regression(x, y):
    kfold = StratifiedKFold(n_splits=5)

    # Lists to store metrics across folds
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    confusion_matrices = []

    for train_index, test_index in kfold.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Oversample classes with bugs
        smote = SMOTE(random_state=0)
        x_train, y_train = smote.fit_resample(x_train, y_train)

        # Add constants (for intercept)
        x_train_const = sm.add_constant(x_train)
        x_test_const = sm.add_constant(x_test)

        model = sm.Logit(y_train, x_train_const)
        model_sm = model.fit(disp=0, maxiter=100)

        # Prediction metrics
        y_pred_prob = model_sm.predict(x_test_const)
        y_pred = (y_pred_prob > PROB_THRESHOLD).astype(int)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, zero_division=0)
        recall = metrics.recall_score(y_test, y_pred, zero_division=0)
        f1 = metrics.f1_score(y_test, y_pred, zero_division=0)
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        confusion_matrices.append(confusion_matrix)

    avg_accuracy = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)

    # Final model on all data
    smote_full = SMOTE(random_state=0)
    x_resampled_full, y_resampled_full = smote_full.fit_resample(x, y)
    x_const_full = sm.add_constant(x_resampled_full)
    model_full = sm.Logit(y_resampled_full, x_const_full)
    model_sm_full = model_full.fit(disp=0, maxiter=100)

    # Model properties
    p_values = model_sm_full.pvalues
    pr_squared = model_sm_full.prsquared  # Use pseudo R-squared because that fits better with logistic regression
    coefficients = model_sm_full.params
    odds_ratios = np.exp(coefficients)

    return {
        'coefficients': coefficients,
        'odds_ratios': odds_ratios,
        'p_values': p_values,
        'pr_squared': pr_squared,
        'accuracy': avg_accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'confusion_matrix': avg_confusion_matrix
    }
