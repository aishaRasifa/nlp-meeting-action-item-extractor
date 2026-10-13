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
    / "EN2001a_transcript.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "EN2001a_transcript_cleaned.csv"
)


# ============================================================
# 2. LOAD TRANSCRIPT
# ============================================================

print("=" * 60)
print("TRANSCRIPT CLEANING")
print("=" * 60)

df = pd.read_csv(
    INPUT_FILE
)

print("\nOriginal dataset:")
print("Total rows:", len(df))


# ============================================================
# 3. CHECK MISSING TEXT
# ============================================================

empty_text = (
    df["text"]
    .fillna("")
    .str.strip()
    .eq("")
)

empty_count = empty_text.sum()

print(
    "Empty text rows:",
    empty_count
)


# ============================================================
# 4. REMOVE EMPTY TEXT ROWS
# ============================================================

df_cleaned = df[
    ~empty_text
].copy()


# ============================================================
# 5. RESET INDEX
# ============================================================

df_cleaned = df_cleaned.reset_index(
    drop=True
)


# ============================================================
# 6. SAVE CLEANED DATASET
# ============================================================

df_cleaned.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 7. FINAL RESULTS
# ============================================================

print("\nCleaning completed!")

print(
    "Original rows:",
    len(df)
)

print(
    "Rows removed:",
    empty_count
)

print(
    "Remaining rows:",
    len(df_cleaned)
)

print(
    "\nCleaned file saved at:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 8. DISPLAY FIRST 5 ROWS
# ============================================================

print(
    "\nFirst 5 cleaned transcript rows:"
)

print("-" * 60)

for _, row in df_cleaned.head(5).iterrows():

    print(
        f"[{row['start_time']}] "
        f"Speaker {row['speaker']}: "
        f"{row['text']}"
    )