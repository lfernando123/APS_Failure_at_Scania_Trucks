# ==========================================
# APS Failure Prediction using Deep Learning
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# ---------------------------------------
# Load Dataset
# ---------------------------------------

df = pd.read_csv("../../data/processed/train_selected.csv")

# ---------------------------------------
# Separate X and y
# ---------------------------------------

X = df.drop("class", axis=1)
y = df["class"]

# ---------------------------------------
# Missing Value Imputation
# ---------------------------------------

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

# ---------------------------------------
# Train Test Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------
# Feature Scaling
# ---------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------
# Handle Class Imbalance
# ---------------------------------------

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)

# ---------------------------------------
# Build Deep Neural Network
# ---------------------------------------

model = Sequential()

model.add(Dense(256,
                activation='relu',
                input_shape=(X_train.shape[1],)))

model.add(BatchNormalization())
model.add(Dropout(0.30))

model.add(Dense(128,
                activation='relu'))

model.add(BatchNormalization())
model.add(Dropout(0.30))

model.add(Dense(64,
                activation='relu'))

model.add(Dropout(0.20))

model.add(Dense(32,
                activation='relu'))

model.add(Dense(1,
                activation='sigmoid'))

# ---------------------------------------
# Compile Model
# ---------------------------------------

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC()
    ]
)

model.summary()

# ---------------------------------------
# Early Stopping
# ---------------------------------------

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# ---------------------------------------
# Train Model
# ---------------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# ---------------------------------------
# Prediction
# ---------------------------------------

y_prob = model.predict(X_test)

y_pred = (y_prob > 0.5).astype(int)

# ---------------------------------------
# Evaluation
# ---------------------------------------

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)

print("ROC AUC Score:", auc)

# ---------------------------------------
# Confusion Matrix
# ---------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ---------------------------------------
# ROC Curve
# ---------------------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr,
         label=f"AUC = {auc:.4f}")

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.show()

# ---------------------------------------
# Training Curves
# ---------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.legend()

plt.title("Loss Curve")

plt.show()

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.legend()

plt.title("Accuracy Curve")

plt.show()

# ---------------------------------------
# Save Model
# ---------------------------------------

model.save("APS_DeepLearning_Model.keras")

print("\nModel Saved Successfully.")