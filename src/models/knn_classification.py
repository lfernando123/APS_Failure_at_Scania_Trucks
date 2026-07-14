# ======================================================
# K-Nearest Neighbors (KNN) Classification
# ======================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt

# ======================================================
# Load Dataset
# ======================================================

df = pd.read_csv("../../data/processed/train_selected.csv")

# ======================================================
# Features and Target
# ======================================================

X = df.drop("class", axis=1)
y = df["class"]

# ======================================================
# Train-Test Split
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================================================
# Build Pipeline
# ======================================================

knn_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# ======================================================
# Hyperparameter Tuning
# ======================================================

param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"]
}

grid_search = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)

# ======================================================
# Best Model
# ======================================================

best_knn = grid_search.best_estimator_

# ======================================================
# Predictions
# ======================================================

y_pred = best_knn.predict(X_test)
y_prob = best_knn.predict_proba(X_test)[:,1]

# ======================================================
# Evaluation
# ======================================================

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nROC-AUC:", roc_auc_score(y_test, y_prob))

# ======================================================
# Confusion Matrix
# ======================================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - KNN")
plt.show()

# ======================================================
# ROC Curve
# ======================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.4f}")
plt.plot([0,1], [0,1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - KNN")
plt.legend()
plt.show()

# ======================================================
# Cross Validation
# ======================================================

cv_scores = cross_val_score(
    best_knn,
    X,
    y,
    cv=5,
    scoring="f1"
)

print("\nCross Validation F1 Scores:")
print(cv_scores)

print("Mean F1 Score:", cv_scores.mean())