
"""
xgboost_model.py
APS Failure Prediction using XGBoost
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from xgboost import XGBClassifier

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
df = pd.read_csv("../../data/processed/train_selected.csv")

X = df.drop("class", axis=1)
y = df["class"]

# Missing value imputation
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Handle class imbalance
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()
scale_pos_weight = negative / positive

print("Scale Pos Weight:", scale_pos_weight)

# ---------------------------------------------------
# Base Model
# ---------------------------------------------------
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight
)

param_grid = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [200, 300, 500],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

grid = GridSearchCV(
    estimator=XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    ),
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
y_pred = grid.predict(X_test)
y_prob = grid.predict_proba(X_test)[:, 1]

# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------
print("\n========== Evaluation ==========")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_prob))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix")
print(cm)

# ---------------------------------------------------
# Cross Validation
# ---------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    grid,
    X,
    y,
    cv=cv,
    scoring="f1"
)

print("\nCross Validation F1 Scores:", scores)
print("Mean F1:", scores.mean())

# ---------------------------------------------------
# Grid Search
# ---------------------------------------------------
param_grid = {
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 300]
}

grid = GridSearchCV(
    XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight
    ),
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest Parameters")
print(grid.best_params_)
print("Best CV F1:", grid.best_score_)

best_model = grid.best_estimator_

# ---------------------------------------------------
# Feature Importance
# ---------------------------------------------------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop 20 Features")
print(importance.head(20))

plt.figure(figsize=(10,7))
plt.barh(
    importance["Feature"].head(20),
    importance["Importance"].head(20)
)
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# ---------------------------------------------------
# Confusion Matrix Plot
# ---------------------------------------------------
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.savefig("confusion_matrix.png")
plt.close()

# ---------------------------------------------------
# ROC Curve
# ---------------------------------------------------
RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.savefig("roc_curve.png")
plt.close()

# ---------------------------------------------------
# Save Model
# ---------------------------------------------------
# joblib.dump(best_model, "xgboost_model.pkl")
# joblib.dump(imputer, "imputer.pkl")

# print("\nModel saved as xgboost_model.pkl")
# print("Imputer saved as imputer.pkl")

# ---------------------------------------------------
# Example Prediction
# ---------------------------------------------------
sample = X_test.iloc[[0]]
prediction = best_model.predict(sample)[0]
probability = best_model.predict_proba(sample)[0][1]

print("\nExample Prediction")
print("Predicted Class:", prediction)
print("Failure Probability:", probability)
