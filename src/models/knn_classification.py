# ============================================================
# KNN Classifier with LIME & Permutation Importance
# APS Failure Detection Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.inspection import permutation_importance

from lime.lime_tabular import LimeTabularExplainer

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("../../data/processed/train_selected.csv")

X = df.drop("class", axis=1)
y = df["class"]

# ============================================================
# Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# ============================================================
# Missing Value Imputation
# ============================================================

imputer = SimpleImputer(strategy="median")

X_train_imp = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X.columns
)

X_test_imp = pd.DataFrame(
    imputer.transform(X_test),
    columns=X.columns
)

# ============================================================
# Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_imp),
    columns=X.columns
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_imp),
    columns=X.columns
)

# ============================================================
# Train KNN
# ============================================================

knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    metric="minkowski",
    p=2
)

knn_model.fit(X_train_scaled, y_train)

# ============================================================
# Predictions
# ============================================================

y_pred = knn_model.predict(X_test_scaled)
y_prob = knn_model.predict_proba(X_test_scaled)[:,1]

# ============================================================
# Evaluation
# ============================================================

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("Accuracy :", accuracy_score(y_test,y_pred))
print("Precision:", precision_score(y_test,y_pred))
print("Recall   :", recall_score(y_test,y_pred))
print("F1 Score :", f1_score(y_test,y_pred))
print("ROC AUC  :", roc_auc_score(y_test,y_prob))

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test,y_pred)

disp = ConfusionMatrixDisplay(cm)

disp.plot(cmap="Blues")

plt.title("KNN Confusion Matrix")

plt.show()

# ============================================================
# LIME Explainability
# ============================================================

print("\nGenerating LIME Explanation...")

explainer = LimeTabularExplainer(
    training_data=X_train_scaled.values,
    feature_names=X.columns.tolist(),
    class_names=["Negative","Positive"],
    mode="classification"
)

sample = 0

exp = explainer.explain_instance(
    X_test_scaled.iloc[sample].values,
    knn_model.predict_proba,
    num_features=10
)

# Show explanation
fig = exp.as_pyplot_figure()

plt.tight_layout()

plt.show()

# ============================================================
# Permutation Importance
# ============================================================

print("\nCalculating Permutation Importance...")

perm = permutation_importance(
    knn_model,
    X_test_scaled,
    y_test,
    scoring="f1",
    n_repeats=10,
    random_state=42
)

importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":perm.importances_mean
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Features")
print(importance.head(20))

# ============================================================
# Plot Feature Importance
# ============================================================

plt.figure(figsize=(8,10))

plt.barh(
    importance["Feature"].head(20),
    importance["Importance"].head(20)
)

plt.gca().invert_yaxis()

plt.xlabel("Permutation Importance")

plt.title("Top 20 Important Features")

plt.tight_layout()

plt.show()

print("\nFinished.")