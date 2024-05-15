import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from read_data.read_data import read_data, prepare_data

data = prepare_data(read_data())
df = pd.DataFrame(data)

# Splitting data into training and testing sets
X = df.iloc[:, 1:-1]  # Features (leave out file names and target variable)
# X = df[['LOC','Complexity','Depth','Children','Response','Cohesion','Coupling']] # Only baseline metrics
y = df['hasBug']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Creating a logistic regression model
# TODO: should we use class_weight='balanced' here?
logreg = LogisticRegression(max_iter=2000)

# Training the model
logreg.fit(X_train, y_train)

# Making predictions on the testing set
y_pred = logreg.predict(X_test)

# Evaluating the model
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
