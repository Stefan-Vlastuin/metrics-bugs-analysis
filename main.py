import sys

import pandas as pd

from logistic_regression.multivariate import multivariate
from logistic_regression.univariate import univariate
from read_data.read_data import get_dataframe
from stats.correlation import show_correlation
from stats.descriptive_stats import show_descriptive_stats


def format_small(x):
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def enough_values(v):
    return v.where(lambda x: x > 0).count() >= 10


def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <metrics_path> <bugs_path>")
        sys.exit(1)

    metrics_path = sys.argv[1]
    bugs_path = sys.argv[2]

    data = get_dataframe(metrics_path, bugs_path)

    # # Only use files using FP
    # data = data[data['UsesFP'] == 1]

    x = data.iloc[:, 1:-2]  # Features (leave out file names, target variable (hasBug) and usesFP)
    x = x.loc[:, x.apply(enough_values)]  # Filter out features with few values
    y = data['hasBug']  # Target variable

    total = y.size
    bugs = y.sum()
    percentage = bugs / total * 100
    print(bugs, " of ", total, " files (", percentage, "%) have bugs")
    print()
    using_fp = data[data['UsesFP'] == 1].shape[0]
    percentage = using_fp / total * 100
    print(using_fp, " of ", total, " files (", percentage, "%) use FP")
    print()

    show_descriptive_stats(x)
    show_correlation(x)

    print()
    print("UNIVARIATE LOGISTIC REGRESSION PER METRIC")
    uni_results = univariate(x, y)
    for result in uni_results:
        print("Feature", result['feature'])
        print("\tP-Value", result['p_value'])
        print("\tPseudo R-Squared", result['pr_squared'])
        print("\tOdds Ratio", result['odds_ratio'])
        print("\tAccuracy", result['accuracy'])
        print("\tPrecision", result['precision'])
        print("\tRecall", result['recall'])
        print("\tF1 Score", result['f1'])

    print()
    print("UNIVARIATE LOGISTIC REGRESSION PER METRIC (LATEX)")
    uni_results = pd.DataFrame(uni_results).set_index('feature')
    uni_results['odds_ratio'] = uni_results['odds_ratio'].apply(format_small)
    uni_results['p_value'] = uni_results['p_value'].apply(format_small)
    uni_results['pr_squared'] = uni_results['pr_squared'].apply(format_small)
    print(uni_results.to_latex(float_format="%.3f", columns=['odds_ratio', 'p_value', 'pr_squared', 'accuracy',
                                                             'precision', 'recall', 'f1']))

    print()
    print("MULTIVARIATE LOGISTIC REGRESSION (ONLY BASELINE)")
    x_baseline = x[['LOC', 'Complexity', 'Depth', 'Children', 'Response', 'Cohesion', 'Coupling']]
    selected_features, baseline_result = multivariate(x_baseline, y)
    print("Selected features:", selected_features)
    print("Accuracy", baseline_result['accuracy'])
    print("Precision", baseline_result['precision'])
    print("Recall", baseline_result['recall'])
    print("F1 Score", baseline_result['f1'])
    print(baseline_result['confusion_matrix'])

    print()
    print("MULTIVARIATE LOGISTIC REGRESSION (ONLY BASELINE, CHOSEN METRICS)")
    baseline_metrics = pd.DataFrame({
        'odds_ratios': baseline_result['odds_ratios'].apply(format_small),
        'p_values': baseline_result['p_values'].apply(format_small)
    })
    print(baseline_metrics)
    print()
    print("MULTIVARIATE LOGISTIC REGRESSION (ONLY BASELINE, CHOSEN METRICS, LATEX)")
    print(baseline_metrics.to_latex(float_format="%.3f"))

    print()
    print("MULTIVARIATE LOGISTIC REGRESSION")
    selected_features, multi_result = multivariate(x, y)
    print("Selected features:", selected_features)
    print("Accuracy", multi_result['accuracy'])
    print("Precision", multi_result['precision'])
    print("Recall", multi_result['recall'])
    print("F1 Score", multi_result['f1'])
    print(multi_result['confusion_matrix'])

    print()
    print("MULTIVARIATE LOGISTIC REGRESSION (CHOSEN METRICS)")
    multi_metrics = pd.DataFrame({
        'odds_ratios': multi_result['odds_ratios'].apply(format_small),
        'p_values': multi_result['p_values'].apply(format_small)
    })
    print(multi_metrics)
    print()
    print("MULTIVARIATE LOGISTIC REGRESSION (CHOSEN METRICS, LATEX)")
    print(multi_metrics.to_latex(float_format="%.3f"))

    comparison = pd.DataFrame([[baseline_result['accuracy'], multi_result['accuracy']],
                               [baseline_result['precision'], multi_result['precision']],
                               [baseline_result['recall'], multi_result['recall']],
                               [baseline_result['f1'], multi_result['f1']]],
                              index=['Accuracy', 'Precision', 'Recall', 'F1'], columns=['Baseline', 'New'])
    print("COMPARISON")
    print(comparison)
    print()
    print("COMPARISON (LATEX)")
    print(comparison.to_latex(float_format="%.3f"))


if __name__ == "__main__":
    main()
