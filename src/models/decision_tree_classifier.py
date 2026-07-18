# ==========================
# Decision Tree Classifier
# ==========================
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt

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

# Create the model
dt_model = DecisionTreeClassifier(
    criterion="gini",      # or "entropy"
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

param_grid = {
    "max_depth": [4, 6, 8, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    estimator=dt_model,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)


# Train
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]

# Evaluation
print("Best Parameters:")
print(grid.best_params_)
print(f"Best CV F1 Score: {grid.best_score_:.4f}")

print("Decision Tree Results")
print("-" * 40)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

disp.plot(cmap="Blues")
plt.title("Decision Tree Confusion Matrix")
plt.show()