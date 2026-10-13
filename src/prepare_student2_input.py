import pandas as pd
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "EN2001a_transcript_processed.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "action_extraction_input.csv"
)


# ============================================================
# 2. LOAD PROCESSED TRANSCRIPT
# ============================================================

print("=" * 60)
print("PREPARING STUDENT 2 INPUT")
print("=" * 60)

print("\nLoading processed transcript...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Total processed rows:",
    len(df)
)


# ============================================================
# 3. SELECT REQUIRED COLUMNS
# ============================================================

required_columns = [
    "meeting_id",
    "speaker",
    "segment_id",
    "start_time",
    "end_time",
    "text",
    "clean_text",
    "tokens",
    "filtered_tokens",
    "lemmas",
    "pos_tags",
    "named_entities"
]


# Check that all required columns exist

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    print(
        "\nERROR: The following columns are missing:"
    )

    for column in missing_columns:

        print(
            "-",
            column
        )

    raise ValueError(
        "Required columns are missing."
    )


# ============================================================
# 4. CREATE STUDENT 2 DATASET
# ============================================================

student2_df = df[
    required_columns
].copy()


# ============================================================
# 5. REMOVE EMPTY TEXT JUST IN CASE
# ============================================================

student2_df = student2_df[
    student2_df["text"]
    .fillna("")
    .str.strip()
    != ""
].copy()


# Reset index

student2_df = student2_df.reset_index(
    drop=True
)


# ============================================================
# 6. SAVE STUDENT 2 INPUT
# ============================================================

student2_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)

print(
    "STUDENT 2 INPUT CREATED"
)

print("=" * 60)

print(
    "\nRows prepared:",
    len(student2_df)
)

print(
    "\nColumns:"
)

for column in student2_df.columns:

    print(
        "-",
        column
    )


print(
    "\nOutput file:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 8. DISPLAY FIRST 5 ROWS
# ============================================================

print(
    "\nFirst 5 rows:"
)

print("-" * 60)

for _, row in student2_df.head(5).iterrows():

    print(
        f"\nSpeaker: {row['speaker']}"
    )

    print(
        f"Time: {row['start_time']} - "
        f"{row['end_time']}"
    )

    print(
        f"Text: {row['text']}"
    )


# ============================================================
# 9. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)

print(
    "STUDENT 1 HANDOFF COMPLETED"
)

print("=" * 60)

print(
    "\nStudent 2 can now use:"
)

print(
    "action_extraction_input.csv"
)