
"""
APS Failure Prediction using TensorFlow
Manual Grid Search + 5-Fold Stratified Cross Validation
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("../../data/processed/train_selected.csv")

X = df.drop("class", axis=1)
y = df["class"]

# ----------------------------
# Preprocess
# ----------------------------
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ----------------------------
# Model factory
# ----------------------------
def build_model(input_dim, lr, n1, n2, dr):
    model = Sequential([
        tf.keras.Input(shape=(input_dim,)),
        Dense(n1, activation="relu"),
        BatchNormalization(),
        Dropout(dr),
        Dense(n2, activation="relu"),
        BatchNormalization(),
        Dropout(dr),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )
    return model

# ----------------------------
# Hyperparameter grid
# ----------------------------
grid = {
    "learning_rate":[0.001,0.0005],
    "neurons1":[128,256],
    "neurons2":[64,128],
    "dropout":[0.2,0.3],
    "batch_size":[64],
    "epochs":[30]
}

keys = list(grid.keys())
combinations = list(itertools.product(*grid.values()))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
best_auc = -1
best_params = None

print(f"Testing {len(combinations)} hyperparameter combinations...")

for idx, values in enumerate(combinations, start=1):
    params = dict(zip(keys, values))
    print(f"\nCombination {idx}/{len(combinations)}: {params}")

    fold_scores = []

    for fold, (tr, val) in enumerate(cv.split(X_train, y_train), start=1):
        print(f"  Fold {fold}/5")

        model = build_model(
            X_train.shape[1],
            params["learning_rate"],
            params["neurons1"],
            params["neurons2"],
            params["dropout"]
        )

        es = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        model.fit(
            X_train[tr], y_train.iloc[tr] if hasattr(y_train,"iloc") else y_train[tr],
            validation_data=(X_train[val], y_train.iloc[val] if hasattr(y_train,"iloc") else y_train[val]),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=0,
            callbacks=[es]
        )

        probs = model.predict(X_train[val], verbose=0).ravel()
        auc = roc_auc_score(y_train.iloc[val] if hasattr(y_train,"iloc") else y_train[val], probs)
        fold_scores.append(auc)

    mean_auc = np.mean(fold_scores)
    print("Mean ROC-AUC:", round(mean_auc,5))

    results.append({**params,"mean_auc":mean_auc})

    if mean_auc > best_auc:
        best_auc = mean_auc
        best_params = params

results_df = pd.DataFrame(results)
results_df.to_csv("DL_CV_Results.csv", index=False)

print("\nBest Parameters")
print(best_params)
print("Best CV ROC-AUC:", best_auc)

# ----------------------------
# Train final model
# ----------------------------
final_model = build_model(
    X_train.shape[1],
    best_params["learning_rate"],
    best_params["neurons1"],
    best_params["neurons2"],
    best_params["dropout"]
)

es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = final_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=best_params["epochs"],
    batch_size=best_params["batch_size"],
    callbacks=[es],
    verbose=1
)

y_prob = final_model.predict(X_test).ravel()
y_pred = (y_prob > 0.5).astype(int)

print(classification_report(y_test, y_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_prob))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

fpr,tpr,_=roc_curve(y_test,y_prob)
plt.figure()
plt.plot(fpr,tpr,label=f"AUC={roc_auc_score(y_test,y_prob):.4f}")
plt.plot([0,1],[0,1],"--")
plt.legend()
plt.title("ROC Curve")
plt.show()

final_model.save("APS_DNN_Final.keras")
print("Finished.")
