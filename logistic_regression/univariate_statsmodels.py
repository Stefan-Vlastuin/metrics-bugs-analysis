from logistic_regression.log_reg import apply_logistic_regression


def univariate(x, y):
    results = []

    # Do univariate logistic regression for each feature separately
    for feature in x.columns:
        result = apply_logistic_regression(x[[feature]], y)
        results.append({
            'feature': feature,
            'coefficient': result['coefficients'][feature],
            'odds_ratio': result['odds_ratios'][feature],
            'p_value': result['p_values'][feature],
            'pr_squared': result['pr_squared'],
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1'],
            'confusion_matrix': result['confusion_matrix']
        })

    return results
