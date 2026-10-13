import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/action_candidates.csv"
)

OUTPUT_FILE = Path(
    "data/processed/action_annotation.csv"
)


# ============================================================
# LOAD CANDIDATES
# ============================================================

print("=" * 60)
print("STUDENT 2 - ACTION ITEM ANNOTATION")
print("=" * 60)

print("\nLoading action candidates...")

df = pd.read_csv(INPUT_FILE)

print(f"Total candidate sentences: {len(df)}")


# ============================================================
# SELECT COLUMNS FOR ANNOTATION
# ============================================================

annotation_df = df[
    [
        "meeting_id",
        "speaker",
        "segment_id",
        "start_time",
        "end_time",
        "text",
        "matched_pattern"
    ]
].copy()


# ============================================================
# ADD LABEL COLUMNS
# ============================================================

annotation_df["label"] = ""

annotation_df["task"] = ""

annotation_df["assignee"] = ""

annotation_df["deadline"] = ""

annotation_df["notes"] = ""


# ============================================================
# SAVE ANNOTATION FILE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

annotation_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("ANNOTATION FILE CREATED")
print("=" * 60)

print(f"\nRows to annotate: {len(annotation_df)}")

print("\nColumns:")
for column in annotation_df.columns:
    print(f"- {column}")

print("\nOutput file:")
print(OUTPUT_FILE.resolve())


# ============================================================
# SHOW FIRST 10 ROWS
# ============================================================

print("\n" + "=" * 60)
print("FIRST 10 SENTENCES TO ANNOTATE")
print("=" * 60)

for i, row in annotation_df.head(10).iterrows():

    print("\n------------------------------------------------------------")

    print(f"Row       : {i}")
    print(f"Speaker   : {row['speaker']}")
    print(f"Time      : {row['start_time']} - {row['end_time']}")
    print(f"Text      : {row['text']}")
    print(f"Pattern   : {row['matched_pattern']}")


print("\n" + "=" * 60)
print("ANNOTATION INSTRUCTIONS")
print("=" * 60)

print("""
Open action_annotation.csv in Excel.

For each row, fill the following columns:

label:
    ACTION
    or
    NON_ACTION

task:
    The actual task/action.
    Example:
    "update the document"

assignee:
    Person responsible for the task.
    Example:
    "John"

deadline:
    Deadline if mentioned.
    Example:
    "Friday"

If no assignee or deadline is mentioned,
leave that field blank.

notes:
    Optional explanation for difficult cases.
""")

print("=" * 60)
print("ANNOTATION FILE READY")
print("=" * 60)