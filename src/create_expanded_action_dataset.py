import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "EN2001a_transcript_processed.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "action_annotation_expanded.csv"
)

# --------------------------------------------------
# 2. Load processed transcript
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Total transcript rows:", len(df))

# Remove empty text
df = df.dropna(subset=["text"])
df = df[df["text"].astype(str).str.strip() != ""]

# --------------------------------------------------
# 3. Action-related candidates
# --------------------------------------------------

action_patterns = (
    r"\bshould\b|"
    r"\bneed to\b|"
    r"\bneeds to\b|"
    r"\bhave to\b|"
    r"\bhas to\b|"
    r"\bmust\b|"
    r"\blet's\b|"
    r"\bcan you\b|"
    r"\bcould you\b|"
    r"\bwould you\b|"
    r"\bwill\b|"
    r"\bprepare\b|"
    r"\bcreate\b|"
    r"\bsend\b|"
    r"\bcheck\b|"
    r"\breview\b|"
    r"\bupdate\b|"
    r"\bfinish\b|"
    r"\bcomplete\b|"
    r"\bimplement\b|"
    r"\bdesign\b|"
    r"\btest\b|"
    r"\bsubmit\b|"
    r"\bwrite\b|"
    r"\badd\b|"
    r"\bremove\b|"
    r"\bfix\b|"
    r"\bchange\b|"
    r"\bdevelop\b"
)

action_candidates = df[
    df["text"].str.lower().str.contains(
        action_patterns,
        regex=True,
        na=False
    )
].copy()

# --------------------------------------------------
# 4. Non-action candidates
# --------------------------------------------------

non_action_candidates = df[
    ~df.index.isin(action_candidates.index)
].copy()

# --------------------------------------------------
# 5. Select 150 from each group
# --------------------------------------------------

action_sample = action_candidates.sample(
    n=min(150, len(action_candidates)),
    random_state=42
)

non_action_sample = non_action_candidates.sample(
    n=min(150, len(non_action_candidates)),
    random_state=42
)

# --------------------------------------------------
# 6. Combine
# --------------------------------------------------

selected = pd.concat(
    [action_sample, non_action_sample],
    ignore_index=True
)

# Shuffle
selected = selected.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# --------------------------------------------------
# 7. Create annotation columns
# --------------------------------------------------

columns = [
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

selected = selected[columns]

selected["label"] = ""
selected["task"] = ""
selected["assignee"] = ""
selected["deadline"] = ""
selected["notes"] = ""

# --------------------------------------------------
# 8. Save
# --------------------------------------------------

selected.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nExpanded annotation dataset created!")
print("Rows:", len(selected))
print("Action candidates selected:", len(action_sample))
print("Non-action candidates selected:", len(non_action_sample))
print("Output:")
print(OUTPUT_FILE)

print("\nFirst 10 rows:")
print(
    selected[
        ["speaker", "segment_id", "text", "label"]
    ].head(10).to_string(index=False)
)