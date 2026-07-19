import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------
# Reproducibility
# ---------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
df = pd.read_csv("../../data/processed/train_selected.csv")

X = df.drop("class", axis=1)
y = df["class"]

# ---------------------------------------------------
# Preprocessing
# ---------------------------------------------------
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=SEED
)

# ---------------------------------------------------
# Build MLP
# ---------------------------------------------------
def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

mlp = build_model(X_train.shape[1])

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = mlp.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------
y_prob = mlp.predict(X_test, verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)

print("\n========== Evaluation ==========")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_prob))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------
# Training Curves
# ---------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("mlp_loss.png", dpi=300)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("mlp_accuracy.png", dpi=300)
plt.close()

# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# ---------------------------------------------------
# ROC Curve
# ---------------------------------------------------
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.close()

# ---------------------------------------------------
# SHAP (Kernel Explainer)
# ---------------------------------------------------
print("\nCalculating SHAP values...")

background = X_train.sample(min(100, len(X_train)), random_state=SEED)

explainer = shap.KernelExplainer(
    mlp.predict,
    background
)

sample = X_test.iloc[:100]

shap_values = explainer.shap_values(sample)

shap.summary_plot(
    shap_values,
    sample,
    feature_names=X.columns,
    show=False
)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)
plt.close()

shap.summary_plot(
    shap_values,
    sample,
    feature_names=X.columns,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=300)
plt.close()

# ---------------------------------------------------
# Permutation Importance
# ---------------------------------------------------
class KerasWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (self.model.predict(X, verbose=0).flatten() >= 0.5).astype(int)

    def predict_proba(self, X):
        p = self.model.predict(X, verbose=0).flatten()
        return np.column_stack((1-p, p))

wrapper = KerasWrapper(mlp)

perm = permutation_importance(
    wrapper,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=10,
    random_state=SEED,
    n_jobs=1
)

perm_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm.importances_mean
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10,7))
plt.barh(
    perm_df["Feature"].head(20),
    perm_df["Importance"].head(20)
)
plt.gca().invert_yaxis()
plt.title("Permutation Importance")
plt.tight_layout()
plt.savefig("permutation_importance.png", dpi=300)
plt.close()

# ---------------------------------------------------
# Save model
# ---------------------------------------------------
mlp.save("mlp_model.keras")

print("\nFinished successfully.")
