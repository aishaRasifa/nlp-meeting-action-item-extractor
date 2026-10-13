import pandas as pd
import os

INPUT_FILE = "data/processed/final_action_items.csv"

OUTPUT_DIR = "data/processed/action_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("STUDENT 2 - ACTION ITEM ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\nTotal final action items:", len(df))


# ------------------------------------------------------------
# 1. Action items by speaker
# ------------------------------------------------------------

speaker_counts = (
    df["speaker"]
    .value_counts()
    .reset_index()
)

speaker_counts.columns = [
    "speaker",
    "action_item_count"
]

speaker_counts.to_csv(
    f"{OUTPUT_DIR}/action_items_by_speaker.csv",
    index=False
)


# ------------------------------------------------------------
# 2. Action items by assignee
# ------------------------------------------------------------

assignee_counts = (
    df["assignee"]
    .value_counts()
    .reset_index()
)

assignee_counts.columns = [
    "assignee",
    "action_item_count"
]

assignee_counts.to_csv(
    f"{OUTPUT_DIR}/action_items_by_assignee.csv",
    index=False
)


# ------------------------------------------------------------
# 3. Action items by deadline
# ------------------------------------------------------------

deadline_counts = (
    df["deadline"]
    .value_counts()
    .reset_index()
)

deadline_counts.columns = [
    "deadline",
    "action_item_count"
]

deadline_counts.to_csv(
    f"{OUTPUT_DIR}/action_items_by_deadline.csv",
    index=False
)


# ------------------------------------------------------------
# 4. Confidence statistics
# ------------------------------------------------------------

confidence_stats = pd.DataFrame({
    "metric": [
        "Average confidence",
        "Minimum confidence",
        "Maximum confidence"
    ],
    "value": [
        df["confidence"].mean(),
        df["confidence"].min(),
        df["confidence"].max()
    ]
})

confidence_stats.to_csv(
    f"{OUTPUT_DIR}/confidence_statistics.csv",
    index=False
)


# ------------------------------------------------------------
# 5. Overall statistics
# ------------------------------------------------------------

statistics = pd.DataFrame({
    "metric": [
        "Total action items",
        "Unique speakers",
        "Unique assignees",
        "Action items with deadline",
        "Action items without deadline"
    ],
    "value": [
        len(df),
        df["speaker"].nunique(),
        df["assignee"].nunique(),
        (df["deadline"] != "Not specified").sum(),
        (df["deadline"] == "Not specified").sum()
    ]
})

statistics.to_csv(
    f"{OUTPUT_DIR}/action_item_statistics.csv",
    index=False
)


# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

print("\nAction items by speaker:")
print(speaker_counts.to_string(index=False))

print("\nAction items by assignee:")
print(assignee_counts.to_string(index=False))

print("\nAction items by deadline:")
print(deadline_counts.to_string(index=False))

print("\nConfidence:")
print(confidence_stats.to_string(index=False))

print("\nOverall statistics:")
print(statistics.to_string(index=False))


print("\n" + "=" * 70)
print("ACTION ITEM ANALYSIS COMPLETED")
print("=" * 70)

print("\nResults saved in:")
print(os.path.abspath(OUTPUT_DIR))