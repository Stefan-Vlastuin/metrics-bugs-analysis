import sys

from logistic_regression.multivariate import multivariate
from logistic_regression.univariate import univariate
from read_data.read_data import get_dataframe
from stats.correlation import show_correlation
from stats.descriptive_stats import show_descriptive_stats


def enough_values(v):
    return v.where(lambda x: x > 0).count() >= 10


def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <metrics_path> <bugs_path>")
        sys.exit(1)

    metrics_path = sys.argv[1]
    bugs_path = sys.argv[2]

    data = get_dataframe(metrics_path, bugs_path)

    x = data.iloc[:, 1:-1]  # Features (leave out file names and target variable)
    x = x.loc[:, x.apply(enough_values)]  # Filter out features with few values
    y = data['hasBug']  # Target variable

    show_descriptive_stats(x)
    show_correlation(x)

    print()
    print("Univariate logistic regression per metric")
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
    print("Multivariate logistic regression (only baseline)")
    x_baseline = x[['LOC', 'Complexity', 'Depth', 'Children', 'Response', 'Cohesion', 'Coupling']]
    baseline_result = multivariate(x_baseline, y)
    print("Accuracy", baseline_result['accuracy'])
    print("Precision", baseline_result['precision'])
    print("Recall", baseline_result['recall'])
    print("F1 Score", baseline_result['f1'])
    print(baseline_result['confusion_matrix'])

    print()
    print("Multivariate logistic regression")
    multi_result = multivariate(x, y)
    print("Accuracy", multi_result['accuracy'])
    print("Precision", multi_result['precision'])
    print("Recall", multi_result['recall'])
    print("F1 Score", multi_result['f1'])
    print(multi_result['confusion_matrix'])


if __name__ == "__main__":
    main()
