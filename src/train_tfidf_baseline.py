import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "action_annotation_expanded_labeled.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "action_model_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("TF-IDF + LOGISTIC REGRESSION BASELINE")
print("=" * 60)

print("\nTotal samples:", len(df))

# Remove missing text or labels
df = df.dropna(subset=["text", "label"])

df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(str).str.strip().str.upper()

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 3. INPUT AND TARGET
# ============================================================

X = df["text"]
y = df["label"]


# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# 60% train, 20% validation, 20% test

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\n" + "=" * 60)
print("DATASET SPLIT")
print("=" * 60)

print("Training   :", len(X_train))
print("Validation :", len(X_val))
print("Testing    :", len(X_test))


# ============================================================
# 5. TF-IDF VECTORIZATION
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

# IMPORTANT:
# Fit only on training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform validation/test using the same vocabulary
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print("\n" + "=" * 60)
print("TF-IDF")
print("=" * 60)

print("Training shape   :", X_train_tfidf.shape)
print("Validation shape :", X_val_tfidf.shape)
print("Testing shape    :", X_test_tfidf.shape)


# ============================================================
# 6. LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

print("\nModel training completed.")


# ============================================================
# 7. VALIDATION PREDICTION
# ============================================================

val_predictions = model.predict(X_val_tfidf)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_precision = precision_score(
    y_val,
    val_predictions,
    pos_label="ACTION",
    zero_division=0
)

val_recall = recall_score(
    y_val,
    val_predictions,
    pos_label="ACTION",
    zero_division=0
)

val_f1 = f1_score(
    y_val,
    val_predictions,
    pos_label="ACTION",
    zero_division=0
)

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print("Accuracy :", round(val_accuracy, 4))
print("Precision:", round(val_precision, 4))
print("Recall   :", round(val_recall, 4))
print("F1-score :", round(val_f1, 4))


# ============================================================
# 8. FINAL TEST PREDICTION
# ============================================================

test_predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    pos_label="ACTION",
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    pos_label="ACTION",
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    pos_label="ACTION",
    zero_division=0
)


# ============================================================
# 9. FINAL TEST RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        test_predictions,
        labels=["ACTION", "NON_ACTION"],
        zero_division=0
    )
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=["NON_ACTION", "ACTION"]
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("                 Predicted")
print("              NON_ACTION  ACTION")
print("Actual")

print(
    f"NON_ACTION       {cm[0][0]:3d}       {cm[0][1]:3d}"
)

print(
    f"ACTION           {cm[1][0]:3d}       {cm[1][1]:3d}"
)


# ============================================================
# 12. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "text": X_test.values,
    "actual_label": y_test.values,
    "predicted_label": test_predictions
})

prediction_file = (
    OUTPUT_DIR / "tfidf_test_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# 13. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "metric": [
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ],
    "value": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

metrics_file = (
    OUTPUT_DIR / "tfidf_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# 14. SAVE DATA SPLITS
# ============================================================

train_df = pd.DataFrame({
    "text": X_train,
    "label": y_train
})

val_df = pd.DataFrame({
    "text": X_val,
    "label": y_val
})

test_df = pd.DataFrame({
    "text": X_test,
    "label": y_test
})

train_df.to_csv(
    OUTPUT_DIR / "tfidf_train.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "tfidf_validation.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "tfidf_test.csv",
    index=False
)


# ============================================================
# 15. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print("Test predictions :", prediction_file)
print("Metrics           :", metrics_file)
print("Output directory  :", OUTPUT_DIR)

print("\nTF-IDF BASELINE COMPLETED.")