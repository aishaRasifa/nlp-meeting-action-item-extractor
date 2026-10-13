import pandas as pd
import os

# ============================================================
# STUDENT 2 → STUDENT 3 HANDOFF
# ============================================================

INPUT_FILE = "data/processed/action_items_extracted_v2.csv"

OUTPUT_FILE = "data/processed/final_action_items.csv"


print("=" * 70)
print("PREPARING FINAL ACTION ITEM DATASET")
print("=" * 70)

# Load extracted action items
df = pd.read_csv(INPUT_FILE)

print(f"Total extracted action items: {len(df)}")


# ------------------------------------------------------------
# Select required columns
# ------------------------------------------------------------

required_columns = [
    "meeting_id",
    "speaker",
    "segment_id",
    "start_time",
    "end_time",
    "task",
    "assignee",
    "deadline",
    "confidence"
]

# Keep only columns that exist
available_columns = [
    col for col in required_columns
    if col in df.columns
]

df = df[available_columns].copy()


# ------------------------------------------------------------
# Clean missing values
# ------------------------------------------------------------

df["task"] = df["task"].fillna("").astype(str)
df["assignee"] = df["assignee"].fillna("Not specified")
df["deadline"] = df["deadline"].fillna("Not specified")


# ------------------------------------------------------------
# Remove empty tasks
# ------------------------------------------------------------

df = df[df["task"].str.strip() != ""]


# ------------------------------------------------------------
# Remove duplicate action items
# ------------------------------------------------------------

df = df.drop_duplicates(
    subset=["task", "assignee", "deadline"]
)


# ------------------------------------------------------------
# Reset index
# ------------------------------------------------------------

df = df.reset_index(drop=True)


# ------------------------------------------------------------
# Add Action Item ID
# ------------------------------------------------------------

df.insert(
    0,
    "action_item_id",
    range(1, len(df) + 1)
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("\nFinal action items:", len(df))

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\nAssignee distribution:")
print(df["assignee"].value_counts())

print("\nDeadline distribution:")
print(df["deadline"].value_counts())

print("\nFirst 10 final action items:")
print(
    df[
        [
            "action_item_id",
            "task",
            "assignee",
            "deadline",
            "confidence"
        ]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 70)
print("STUDENT 2 HANDOFF COMPLETED")
print("=" * 70)

print("\nOutput:")
print(os.path.abspath(OUTPUT_FILE))