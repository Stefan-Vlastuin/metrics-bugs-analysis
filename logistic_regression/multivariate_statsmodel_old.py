import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from read_data.read_data import read_data, prepare_data

# Read and prepare data
data = prepare_data(read_data())
df = pd.DataFrame(data)

# Define features and target
X = df.iloc[:, 1:-1].apply(pd.to_numeric)  # Features (leave out file names and target variable)
y = pd.to_numeric(df['hasBug'])  # Target variable

# Check for multicollinearity using correlation matrix
correlation_matrix = X.corr()
print(correlation_matrix)

# Check for multicollinearity using VIF
X_const = sm.add_constant(X)
vif = pd.DataFrame()
vif["Feature"] = X_const.columns
vif["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
print(vif)

# Drop features with high VIF (example: if VIF > 10)
features_to_drop = vif[vif["VIF"] > 10]["Feature"].tolist()
features_to_drop.remove('const')  # Don't drop the intercept
X = X.drop(columns=features_to_drop)

# Drop NaN features
features_to_drop = vif[np.isnan(vif["VIF"])]["Feature"].tolist()
X = X.drop(columns=features_to_drop)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Add a constant (intercept) term to the predictors
X_train_const = sm.add_constant(X_train)

weights = pd.Series([1 if val == 0 else 1 / (y_train.value_counts()[1] / y_train.value_counts()[0]) for val in y_train])

# Fit the logistic regression model using statsmodels
model = sm.Logit(y_train, X_train_const, freq_weights=weights).fit(maxiter=5000)

# Get the summary of the model
print(model.summary())

# Extract p-values and coefficients
p_values = model.pvalues
coefficients = model.params
odds_ratios = np.exp(coefficients)

# Combine feature names, coefficients, odds ratios, and p-values
summary_df = pd.DataFrame({
    'Feature': X_train_const.columns,
    'Coefficient': coefficients,
    'Odds Ratio': odds_ratios,
    'p-value': p_values
})

print(summary_df)


# Add constant to test data
X_test_const = sm.add_constant(X_test)

# Predict probabilities
y_pred_proba = model.predict(X_test_const)

# Convert probabilities to binary predictions (0 or 1)
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
