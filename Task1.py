# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score

import shap # type: ignore

# Load dataset
df = pd.read_csv("bank-full.csv", sep=';')

print("First rows:")
print(df.head())

print("\nInfo:")
print(df.info())

# EDA (Exploratory Data Analysis)

print("\nTarget distribution:")
print(df['y'].value_counts())

sns.countplot(x='y', data=df)
plt.title("Target Distribution")
plt.show()

sns.histplot(df['age'])
plt.title("Age Distribution")
plt.show()

sns.boxplot(x='y', y='balance', data=df)
plt.title("Balance vs Subscription")
plt.show()

# Preprocessing

df['y'] = df['y'].map({'yes': 1, 'no': 0})

df = pd.get_dummies(df, drop_first=True)

# Split data
X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions

y_pred_lr = lr.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test)

# Evaluation

print("\nLogistic Regression Results")
print(confusion_matrix(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))

print("\nRandom Forest Results")
print(confusion_matrix(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))

# ROC Curve (Random Forest)
y_prob = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# SHAP (Explainable AI)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values[1], X_test)

# Explain 5 individual predictions
for i in range(5):
    print(f"\nExplanation for sample {i}")
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][i],
        X_test.iloc[i],
        matplotlib=True
    )

# Final Conclusion

print("\nConclusion:")
print("Random Forest performed better than Logistic Regression.")
print("Important features include duration, balance, and age.")
print("Model can help bank target customers more effectively.")