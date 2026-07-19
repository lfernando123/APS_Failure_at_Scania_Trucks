# ============================================================
# Decision Tree Classifier with SHAP & Permutation Importance
# APS Failure Detection Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.inspection import permutation_importance

import shap

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
# Decision Tree Pipeline
# ============================================================

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", DecisionTreeClassifier(
        criterion="gini",
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ))
])

# ============================================================
# Train
# ============================================================

model.fit(X_train, y_train)

# ============================================================
# Predictions
# ============================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ============================================================
# Evaluation
# ============================================================

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_prob))

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm)

disp.plot(cmap="Blues")
plt.title("Decision Tree Confusion Matrix")
plt.show()

# ============================================================
# Plot Decision Tree
# ============================================================

X_train_processed = model.named_steps["imputer"].transform(X_train)

plt.figure(figsize=(22,12))

plot_tree(
    model.named_steps["classifier"],
    feature_names=X.columns,
    class_names=["Negative","Positive"],
    filled=True,
    rounded=True,
    fontsize=7,
    max_depth=3
)

plt.title("Decision Tree (First 3 Levels)")
plt.show()

# ============================================================
# Feature Importance
# ============================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.named_steps["classifier"].feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Features")
print(importance.head(20))

plt.figure(figsize=(8,10))

plt.barh(
    importance["Feature"].head(20),
    importance["Importance"].head(20)
)

plt.gca().invert_yaxis()

plt.xlabel("Importance")
plt.title("Decision Tree Feature Importance")

plt.tight_layout()
plt.show()

# ============================================================
# SHAP Explainability (SHAP 0.48.0)
# ============================================================

print("Calculating SHAP values...")

imputer = SimpleImputer(strategy="median")

X_train_imp = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X.columns
)

X_test_imp = pd.DataFrame(
    imputer.transform(X_test),
    columns=X.columns
)

dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=10,
    random_state=42
)

dt_model.fit(X_train_imp, y_train)

explainer = shap.TreeExplainer(dt_model)

shap_values = explainer.shap_values(X_test_imp)

shap.summary_plot(
    shap_values,
    X_test_imp,
    feature_names=X_test_imp.columns,
    plot_type="dot"
)

shap.summary_plot(
    shap_values,
    X_test_imp,
    feature_names=X_test_imp.columns,
    plot_type="bar"
)

# ============================================================
# Permutation Importance
# ============================================================

print("\nCalculating Permutation Importance...")

perm = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=10,
    random_state=42
)

perm_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm.importances_mean
})

perm_df = perm_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Permutation Features")
print(perm_df.head(20))

plt.figure(figsize=(8,10))

plt.barh(
    perm_df["Feature"].head(20),
    perm_df["Importance"].head(20)
)

plt.gca().invert_yaxis()

plt.xlabel("Permutation Importance")

plt.title("Top 20 Features (Permutation Importance)")

plt.tight_layout()

plt.show()

print("\nFinished.")