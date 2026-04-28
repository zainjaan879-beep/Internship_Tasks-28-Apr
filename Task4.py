# Task 4: Loan Default Risk with Cost Optimization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

# Load YOUR dataset (correct file name)
df = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")

print("Dataset Preview")
print(df.head())

# Drop ID column
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# Separate columns
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

# Handle missing values (SAFE METHOD)
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Encode categorical variables
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ⚠️ TARGET FIX (important)
if "Loan_Status" in df.columns:
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]
else:
    print("Warning: Loan_Status not found. Creating dummy target for demo only.")
    X = df.copy()
    np.random.seed(42)
    y = np.random.randint(0, 2, size=len(df))

# FINAL NaN safety check
X = X.fillna(0)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = LogisticRegression(max_iter=5000, solver="saga")
model.fit(X_train, y_train)

# Predictions
y_probs = model.predict_proba(X_test)[:, 1]

print("ROC-AUC Score:", roc_auc_score(y_test, y_probs))

# Business cost setup
FP_COST = 500
FN_COST = 2000

# Threshold optimization
thresholds = np.arange(0.1, 0.9, 0.01)
costs = []

for t in thresholds:
    y_pred = (y_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cost = (fp * FP_COST) + (fn * FN_COST)
    costs.append(cost)

optimal_threshold = thresholds[np.argmin(costs)]
print("Optimal Threshold:", optimal_threshold)

# Final predictions
final_pred = (y_probs >= optimal_threshold).astype(int)

# Confusion matrix
cm = confusion_matrix(y_test, final_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()

# Feature importance
importance = pd.DataFrame({
    "Feature": df.drop(df.columns[-1], axis=1).columns,
    "Importance": model.coef_[0]
}).sort_values(by="Importance", ascending=False)

print("Top Features:")
print(importance.head(10))

# Cost graph
plt.figure()
plt.plot(thresholds, costs)
plt.xlabel("Threshold")
plt.ylabel("Cost")
plt.title("Cost vs Threshold Optimization")
plt.show()

print("Task Completed Successfully")