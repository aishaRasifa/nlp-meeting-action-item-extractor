import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "action_annotation_expanded_labeled.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "action_classifier_distilbert"
)

RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "action_model_results"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128

BATCH_SIZE = 8

EPOCHS = 3

LEARNING_RATE = 2e-5


# ============================================================
# 4. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DISTILBERT ACTION / NON-ACTION CLASSIFIER")
print("=" * 70)

print("\nDevice:", device)


# ============================================================
# 5. LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\nTotal samples:", len(df))

# Remove missing values
df = df.dropna(subset=["text", "label"]).copy()

df["text"] = df["text"].astype(str)

df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# Keep only expected labels
df = df[
    df["label"].isin(
        ["ACTION", "NON_ACTION"]
    )
].reset_index(drop=True)


# ============================================================
# 6. LABEL ENCODING
# ============================================================

label2id = {
    "NON_ACTION": 0,
    "ACTION": 1
}

id2label = {
    0: "NON_ACTION",
    1: "ACTION"
}

df["label_id"] = df["label"].map(label2id)


print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X = df["text"]

y = df["label_id"]


# 60% training
# 20% validation
# 20% testing

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=SEED,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)


print("\n" + "=" * 70)
print("DATASET SPLIT")
print("=" * 70)

print("Training   :", len(X_train))
print("Validation :", len(X_val))
print("Testing    :", len(X_test))


# ============================================================
# 8. SAVE SPLITS
# ============================================================

train_df = pd.DataFrame({
    "text": X_train.values,
    "label_id": y_train.values
})

val_df = pd.DataFrame({
    "text": X_val.values,
    "label_id": y_val.values
})

test_df = pd.DataFrame({
    "text": X_test.values,
    "label_id": y_test.values
})

train_df["label"] = train_df["label_id"].map(id2label)
val_df["label"] = val_df["label_id"].map(id2label)
test_df["label"] = test_df["label_id"].map(id2label)

train_df.to_csv(
    RESULTS_DIR / "distilbert_train.csv",
    index=False
)

val_df.to_csv(
    RESULTS_DIR / "distilbert_validation.csv",
    index=False
)

test_df.to_csv(
    RESULTS_DIR / "distilbert_test.csv",
    index=False
)


# ============================================================
# 9. TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 10. CUSTOM DATASET
# ============================================================

class ActionDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length
    ):

        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                label,
                dtype=torch.long
            )
        }

        return item


# ============================================================
# 11. CREATE DATASETS
# ============================================================

train_dataset = ActionDataset(
    X_train,
    y_train,
    tokenizer,
    MAX_LENGTH
)

val_dataset = ActionDataset(
    X_val,
    y_val,
    tokenizer,
    MAX_LENGTH
)

test_dataset = ActionDataset(
    X_test,
    y_test,
    tokenizer,
    MAX_LENGTH
)


# ============================================================
# 12. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 13. LOAD DISTILBERT
# ============================================================

print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

model.to(device)


# ============================================================
# 14. OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 15. EVALUATION FUNCTION
# ============================================================

def evaluate(model, data_loader):

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0

    with torch.no_grad():

        for batch in data_loader:

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

            predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    avg_loss = (
        total_loss / len(data_loader)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0
    )

    return (
        avg_loss,
        accuracy,
        precision,
        recall,
        f1,
        all_labels,
        all_predictions
    )


# ============================================================
# 16. TRAINING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

training_history = []

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0

    for batch in train_loader:

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        total_train_loss += loss.item()

        loss.backward()

        optimizer.step()

    avg_train_loss = (
        total_train_loss / len(train_loader)
    )

    (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1,
        _,
        _
    ) = evaluate(
        model,
        val_loader
    )

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    print(
        f"Train Loss : {avg_train_loss:.4f}"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    print(
        f"Val Precision: {val_precision:.4f}"
    )

    print(
        f"Val Recall: {val_recall:.4f}"
    )

    print(
        f"Val F1: {val_f1:.4f}"
    )

    training_history.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1
    })


# ============================================================
# 17. FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

(
    test_loss,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1,
    test_labels,
    test_predictions
) = evaluate(
    model,
    test_loader
)

print(
    f"Test Loss    : {test_loss:.4f}"
)

print(
    f"Accuracy     : {test_accuracy:.4f}"
)

print(
    f"Precision    : {test_precision:.4f}"
)

print(
    f"Recall       : {test_recall:.4f}"
)

print(
    f"F1-score     : {test_f1:.4f}"
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        test_labels,
        test_predictions,
        labels=[0, 1],
        target_names=[
            "NON_ACTION",
            "ACTION"
        ],
        zero_division=0
    )
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions,
    labels=[0, 1]
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "                 Predicted"
)

print(
    "              NON_ACTION  ACTION"
)

print(
    f"NON_ACTION       {cm[0][0]:3d}       {cm[0][1]:3d}"
)

print(
    f"ACTION           {cm[1][0]:3d}       {cm[1][1]:3d}"
)


# ============================================================
# 20. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "text": X_test.values,
    "actual_label": [
        id2label[x]
        for x in test_labels
    ],
    "predicted_label": [
        id2label[x]
        for x in test_predictions
    ]
})

prediction_file = (
    RESULTS_DIR
    / "distilbert_test_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# 21. SAVE METRICS
# ============================================================

metrics = {
    "model": "DistilBERT",
    "dataset_size": len(df),
    "train_size": len(X_train),
    "validation_size": len(X_val),
    "test_size": len(X_test),
    "accuracy": test_accuracy,
    "precision": test_precision,
    "recall": test_recall,
    "f1_score": test_f1,
    "confusion_matrix": cm.tolist()
}

with open(
    RESULTS_DIR / "distilbert_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# 22. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    training_history
)

history_df.to_csv(
    RESULTS_DIR / "distilbert_training_history.csv",
    index=False
)


# ============================================================
# 23. SAVE MODEL
# ============================================================

print("\nSaving model...")

model.save_pretrained(
    MODEL_DIR
)

tokenizer.save_pretrained(
    MODEL_DIR
)


# ============================================================
# 24. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("DISTILBERT TRAINING COMPLETED")
print("=" * 70)

print("\nModel saved to:")
print(MODEL_DIR)

print("\nResults saved to:")
print(RESULTS_DIR)