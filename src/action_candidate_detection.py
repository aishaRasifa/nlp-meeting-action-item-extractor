import pandas as pd
import re
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/action_extraction_input.csv"
)

OUTPUT_FILE = Path(
    "data/processed/action_candidates.csv"
)


# ============================================================
# ACTION PATTERNS
# ============================================================

ACTION_PATTERNS = {

    # Requirement / obligation
    "should": r"\bshould\b",
    "need_to": r"\bneed(?:s)?\s+to\b",
    "have_to": r"\bhave\s+to\b",
    "has_to": r"\bhas\s+to\b",
    "must": r"\bmust\b",

    # Future commitment
    "will": r"\bwill\b",

    # Requests / suggestions
    "lets": r"\blet'?s\b",
    "can_you": r"\bcan\s+you\b",
    "could_you": r"\bcould\s+you\b",
    "would_you": r"\bwould\s+you\b",

    # Common action verbs
    "prepare": r"\bprepare\b",
    "create": r"\bcreate\b",
    "send": r"\bsend\b",
    "check": r"\bcheck\b",
    "review": r"\breview\b",
    "update": r"\bupdate\b",
    "finish": r"\bfinish\b",
    "complete": r"\bcomplete\b",
    "implement": r"\bimplement\b",
    "design": r"\bdesign\b",
    "test": r"\btest\b",
    "submit": r"\bsubmit\b",
    "write": r"\bwrite\b",
    "add": r"\badd\b",
    "remove": r"\bremove\b",
    "fix": r"\bfix\b",
    "change": r"\bchange\b",
    "develop": r"\bdevelop\b",
    "work_on": r"\bwork\s+on\b"
}


# ============================================================
# FUNCTION TO DETECT ACTION CANDIDATE
# ============================================================

def detect_action_candidate(text):

    if pd.isna(text):
        return None

    text = str(text).strip()

    if not text:
        return None

    matched_patterns = []

    for pattern_name, pattern in ACTION_PATTERNS.items():

        if re.search(pattern, text, flags=re.IGNORECASE):
            matched_patterns.append(pattern_name)

    if matched_patterns:
        return ", ".join(matched_patterns)

    return None


# ============================================================
# MAIN PROGRAM
# ============================================================

print("=" * 60)
print("STUDENT 2 - ACTION CANDIDATE DETECTION")
print("=" * 60)


# ------------------------------------------------------------
# Load input
# ------------------------------------------------------------

print("\nLoading Student 1 output...")

df = pd.read_csv(INPUT_FILE)

print(f"Total transcript segments: {len(df)}")


# ------------------------------------------------------------
# Detect candidates
# ------------------------------------------------------------

df["matched_pattern"] = df["text"].apply(
    detect_action_candidate
)


df["action_candidate"] = df["matched_pattern"].notna().astype(int)


# ------------------------------------------------------------
# Keep only candidates
# ------------------------------------------------------------

candidates = df[df["action_candidate"] == 1].copy()


# ------------------------------------------------------------
# Save result
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

candidates.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

total_segments = len(df)
candidate_count = len(candidates)

if total_segments > 0:
    percentage = (
        candidate_count / total_segments
    ) * 100
else:
    percentage = 0


print("\n" + "=" * 60)
print("ACTION CANDIDATE DETECTION COMPLETED")
print("=" * 60)

print(f"\nTotal transcript segments : {total_segments}")
print(f"Action candidates         : {candidate_count}")
print(f"Candidate percentage      : {percentage:.2f}%")

print("\nOutput file:")
print(OUTPUT_FILE.resolve())


# ============================================================
# DISPLAY FIRST 20 CANDIDATES
# ============================================================

print("\n" + "=" * 60)
print("FIRST 20 ACTION CANDIDATES")
print("=" * 60)

for index, row in candidates.head(20).iterrows():

    print("\n------------------------------------------------------------")

    print(f"Speaker : {row['speaker']}")
    print(
        f"Time    : "
        f"{row['start_time']} - {row['end_time']}"
    )

    print(f"Text    : {row['text']}")

    print(
        f"Matched pattern : "
        f"{row['matched_pattern']}"
    )


print("\n" + "=" * 60)
print("NEXT STEP")
print("=" * 60)

print(
    "Review the candidate sentences before building "
    "the BERT-based action classifier."
)