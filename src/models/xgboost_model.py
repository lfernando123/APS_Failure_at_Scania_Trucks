
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
df = pd.read_csv("../../data/processed/train_selected.csv")

X = df.drop("class", axis=1)
y = df["class"]

# ---------------------------------------------------
# Missing Value Imputation
# ---------------------------------------------------
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ---------------------------------------------------
# Train/Test Split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# ---------------------------------------------------
# Handle Class Imbalance
# ---------------------------------------------------
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()
scale_pos_weight = negative / positive

print(f"Scale Pos Weight: {scale_pos_weight:.2f}")

# ---------------------------------------------------
# Hyperparameter Tuning
# ---------------------------------------------------
param_grid = {
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 300],
    "subsample": [0.8],
    "colsample_bytree": [0.8]
}

grid = GridSearchCV(
    XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight
    ),
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters")
print(grid.best_params_)
print("Best CV F1:", grid.best_score_)

# ---------------------------------------------------
# Cross Validation
# ---------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(best_model, X, y, cv=cv, scoring="f1", n_jobs=-1)

print("\nCross Validation F1 Scores:", scores)
print("Mean F1:", scores.mean())

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]

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

# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()

# ---------------------------------------------------
# ROC Curve
# ---------------------------------------------------
RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.title("ROC Curve")
plt.show()

# ---------------------------------------------------
# Feature Importance
# ---------------------------------------------------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop 20 Features")
print(importance.head(20))

plt.figure(figsize=(10,7))
plt.barh(
    importance["Feature"].head(20),
    importance["Importance"].head(20)
)
plt.gca().invert_yaxis()
plt.title("XGBoost Feature Importance")
plt.show()

# ---------------------------------------------------
# SHAP Summary & Bar Plot
# ---------------------------------------------------
print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X.columns,
    show=False
)
plt.title("SHAP Summary")
plt.show()

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X.columns,
    plot_type="bar",
    show=False
)
plt.title("SHAP Bar Plot")
plt.show()

# ---------------------------------------------------
# SHAP Waterfall
# ---------------------------------------------------
explainer2 = shap.Explainer(best_model)
exp = explainer2(X_test.iloc[:1])

shap.plots.waterfall(exp[0], show=False)
plt.title("SHAP Waterfall Plot")
plt.show()

# ---------------------------------------------------
# Permutation Importance
# ---------------------------------------------------
perm = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm.importances_mean
}).sort_values("Importance", ascending=False)

print("\nTop 20 Permutation Importance")
print(perm_df.head(20))

plt.figure(figsize=(10,7))
plt.barh(
    perm_df["Feature"].head(20),
    perm_df["Importance"].head(20)
)
plt.title("Permutation Importance")
plt.show()

# ---------------------------------------------------
# Example Prediction
# ---------------------------------------------------
sample = X_test.iloc[[0]]
prediction = best_model.predict(sample)[0]
probability = best_model.predict_proba(sample)[0][1]

print("\nExample Prediction")
print("Predicted Class:", prediction)
print("Failure Probability:", probability)

print("\nFinished successfully.")
