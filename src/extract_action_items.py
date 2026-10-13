import pandas as pd
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "action_extraction_input.csv"
)

MODEL_DIR = BASE_DIR / "models" / "action_classifier"

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "action_items_extracted_v2.csv"


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING ACTION ITEM CLASSIFIER")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.to(device)
model.eval()


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading transcript data...")

df = pd.read_csv(INPUT_FILE)

print("Total transcript segments:", len(df))


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def clean_task(text):
    """
    Extract a cleaner task/action phrase.
    """

    text = str(text).strip()

    # Remove common conversational beginnings
    prefixes = [
        r"^yeah[,.]?\s*",
        r"^yes[,.]?\s*",
        r"^okay[,.]?\s*",
        r"^ok[,.]?\s*",
        r"^well[,.]?\s*",
        r"^so[,.]?\s*",
        r"^um[,.]?\s*",
        r"^uh[,.]?\s*",
        r"^maybe[,.]?\s*",
        r"^i think[,.]?\s*",
    ]

    for pattern in prefixes:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE
        )

    # Remove excessive spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove trailing conversational phrases
    text = re.sub(
        r"\b(i don't know|you know|right|yeah)\.?$",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip(" .,;:-")


# ------------------------------------------------------------
# Deadline extraction
# ------------------------------------------------------------

def extract_deadline(text):

    text_lower = text.lower()

    deadline_patterns = [

        r"\btoday\b",

        r"\btomorrow\b",

        r"\bnext week\b",

        r"\bnext month\b",

        r"\bthis week\b",

        r"\bthis month\b",

        r"\bnext monday\b",

        r"\bnext tuesday\b",

        r"\bnext wednesday\b",

        r"\bnext thursday\b",

        r"\bnext friday\b",

        r"\bnext saturday\b",

        r"\bnext sunday\b",

        r"\bby monday\b",

        r"\bby tuesday\b",

        r"\bby wednesday\b",

        r"\bby thursday\b",

        r"\bby friday\b",

        r"\bby saturday\b",

        r"\bby sunday\b",

        r"\bbefore monday\b",

        r"\bbefore tuesday\b",

        r"\bbefore wednesday\b",

        r"\bbefore thursday\b",

        r"\bbefore friday\b",

        r"\bbefore saturday\b",

        r"\bbefore sunday\b",

        r"\bwithin \d+ (?:day|days|week|weeks|month|months)\b",

        r"\bin \d+ (?:day|days|week|weeks|month|months)\b",

        r"\bweek \d+\b",
    ]

    for pattern in deadline_patterns:

        match = re.search(
            pattern,
            text_lower
        )

        if match:
            return match.group(0)

    return "Not specified"


# ------------------------------------------------------------
# Assignee extraction
# ------------------------------------------------------------

def extract_assignee(text):

    # Known technical words that should NOT be people
    technical_words = {
        "x_m_l",
        "xml",
        "sql",
        "java",
        "python",
        "database",
        "framework",
        "parser",
        "algorithm",
        "code",
        "file",
        "files",
        "dictionary",
        "hash",
        "table",
        "search",
        "perl",
        "nite",
    }

    # --------------------------------------------------------
    # Explicit "X should..."
    # --------------------------------------------------------

    patterns = [

        r"\b([A-Z][a-z]+)\s+should\b",

        r"\b([A-Z][a-z]+)\s+will\b",

        r"\b([A-Z][a-z]+)\s+can\b",

        r"\b([A-Z][a-z]+)\s+has to\b",

        r"\b([A-Z][a-z]+)\s+needs to\b",

        r"\b([A-Z][a-z]+)\s+will probably\b",

        r"\b([A-Z][a-z]+)\s+said\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text
        )

        if match:

            person = match.group(1)

            if person.lower() not in technical_words:
                return person


    # --------------------------------------------------------
    # "ask Steve"
    # --------------------------------------------------------

    match = re.search(
        r"\bask\s+([A-Z][a-z]+)\b",
        text
    )

    if match:

        person = match.group(1)

        if person.lower() not in technical_words:
            return person


    # --------------------------------------------------------
    # "send ... to Steve"
    # --------------------------------------------------------

    match = re.search(
        r"\bto\s+([A-Z][a-z]+)\b",
        text
    )

    if match:

        person = match.group(1)

        if person.lower() not in technical_words:
            return person


    # --------------------------------------------------------
    # Explicit team wording
    # --------------------------------------------------------

    if re.search(
        r"\bwe should\b|\bwe need to\b|\bwe have to\b|\blet's\b",
        text,
        re.IGNORECASE
    ):
        return "Team"


    if re.search(
        r"\ball of us\b|\beveryone\b|\beach of us\b",
        text,
        re.IGNORECASE
    ):
        return "All team members"


    # "you should..."
    if re.search(
        r"\byou should\b|\byou need to\b|\byou have to\b",
        text,
        re.IGNORECASE
    ):
        return "You / assigned person"


    return "Not specified"


# ------------------------------------------------------------
# Determine whether a sentence is probably a real action
# ------------------------------------------------------------

def is_real_action(text):

    text_lower = text.lower().strip()


    # --------------------------------------------------------
    # Reject very short / incomplete statements
    # --------------------------------------------------------

    words = text_lower.split()

    if len(words) < 4:
        return False


    # --------------------------------------------------------
    # Reject obvious conversational statements
    # --------------------------------------------------------

    non_action_patterns = [

        r"^that's not really what i meant",

        r"^i don't think that's a good idea",

        r"^i don't know",

        r"^i have no idea",

        r"^yeah\.?$",

        r"^okay\.?$",

        r"^right\.?$",

        r"^yes\.?$",

        r"^no\.?$",

        r"^mm-hmm",

        r"^gosh",

    ]

    for pattern in non_action_patterns:

        if re.search(
            pattern,
            text_lower
        ):
            return False


    # --------------------------------------------------------
    # Reject descriptive / explanatory statements
    # --------------------------------------------------------

    explanation_patterns = [

        r"^the basic .* is",

        r"^the .* is off-line",

        r"^the .* is online",

        r"^this is",

        r"^that's",

        r"^it is",

    ]

    for pattern in explanation_patterns:

        if re.search(
            pattern,
            text_lower
        ):
            return False


    # --------------------------------------------------------
    # Strong action patterns
    # --------------------------------------------------------

    action_patterns = [

        r"\bwe should\b",

        r"\bwe need to\b",

        r"\bwe have to\b",

        r"\bwe will\b",

        r"\blet's\b",

        r"\byou should\b",

        r"\byou need to\b",

        r"\byou have to\b",

        r"\bshould probably\b",

        r"\bneed to\b",

        r"\bhas to\b",

        r"\bhave to\b",

        r"\bmust\b",

        r"\bcan you\b",

        r"\bcould you\b",

        r"\bwould you\b",

        r"\bask [A-Z][a-z]+\b",

        r"\bI will\b",

        r"\bI'll\b",

        r"\bI can\b",

        r"\bI need to\b",

        r"\bI have to\b",

        r"\bI should\b",

    ]

    for pattern in action_patterns:

        if re.search(
            pattern,
            text,
            re.IGNORECASE
        ):
            return True


    # --------------------------------------------------------
    # Direct imperative verbs
    # --------------------------------------------------------

    imperative_verbs = [

        "prepare",
        "create",
        "send",
        "check",
        "review",
        "update",
        "finish",
        "complete",
        "implement",
        "design",
        "test",
        "submit",
        "write",
        "add",
        "remove",
        "fix",
        "change",
        "develop",
        "build",
        "store",
        "calculate",
        "filter",
        "process",
        "extract",
        "convert",
        "load",
        "save",
        "implement",
        "prioritize",
        "look",
        "ask",
    ]

    for verb in imperative_verbs:

        if re.search(
            rf"\b{verb}\b",
            text_lower
        ):

            # Avoid treating every descriptive occurrence as action
            if len(words) >= 5:
                return True


    return False


# ============================================================
# 5. PREDICTION
# ============================================================

print("\nRunning DistilBERT predictions...")

predictions = []
confidences = []

for index, row in df.iterrows():

    text = str(row["text"])

    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    encoding = {
        key: value.to(device)
        for key, value in encoding.items()
    }

    with torch.no_grad():

        outputs = model(**encoding)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = prediction.item()

    confidence_value = confidence.item()

    # label mapping
    # 0 = NON_ACTION
    # 1 = ACTION

    if predicted_class == 1:
        label = "ACTION"
    else:
        label = "NON_ACTION"


    # --------------------------------------------------------
    # Additional rule-based filtering
    # --------------------------------------------------------

    if label == "ACTION":

        if not is_real_action(text):

            label = "NON_ACTION"


    predictions.append(label)
    confidences.append(round(confidence_value, 4))


# ============================================================
# 6. ADD RESULTS
# ============================================================

df["predicted_label"] = predictions
df["confidence"] = confidences


# ============================================================
# 7. EXTRACT ACTION INFORMATION
# ============================================================

action_rows = []

for _, row in df.iterrows():

    if row["predicted_label"] != "ACTION":
        continue


    text = str(row["text"])

    confidence = float(row["confidence"])


    # --------------------------------------------------------
    # Confidence threshold
    # --------------------------------------------------------

    if confidence < 0.60:
        continue


    task = clean_task(text)

    assignee = extract_assignee(text)

    deadline = extract_deadline(text)


    action_rows.append({

        "meeting_id": row["meeting_id"],

        "speaker": row["speaker"],

        "segment_id": row["segment_id"],

        "start_time": row["start_time"],

        "end_time": row["end_time"],

        "original_text": text,

        "predicted_label": row["predicted_label"],

        "confidence": confidence,

        "task": task,

        "assignee": assignee,

        "deadline": deadline

    })


# ============================================================
# 8. CREATE OUTPUT DATAFRAME
# ============================================================

action_df = pd.DataFrame(action_rows)


# ============================================================
# 9. REMOVE DUPLICATE ACTIONS
# ============================================================

if not action_df.empty:

    action_df = action_df.drop_duplicates(
        subset=["original_text"]
    )


# ============================================================
# 10. SAVE OUTPUT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

action_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 11. SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("ACTION ITEM EXTRACTION V2 COMPLETED")
print("=" * 70)

print()

print(
    "Total transcript segments:",
    len(df)
)

print(
    "Predicted ACTION:",
    (df["predicted_label"] == "ACTION").sum()
)

print(
    "Predicted NON_ACTION:",
    (df["predicted_label"] == "NON_ACTION").sum()
)

print(
    "Final extracted action items:",
    len(action_df)
)


if not action_df.empty:

    print(
        "\nAverage confidence:",
        round(
            action_df["confidence"].mean(),
            4
        )
    )

    print("\nAssignee distribution:")

    print(
        action_df["assignee"].value_counts()
    )

    print("\nDeadline distribution:")

    print(
        action_df["deadline"].value_counts()
    )


print("\nOutput file:")

print(OUTPUT_FILE)


# ============================================================
# 12. SHOW FIRST 10
# ============================================================

print("\nFirst 10 extracted action items:")

if not action_df.empty:

    print(
        action_df[
            [
                "speaker",
                "task",
                "assignee",
                "deadline",
                "confidence"
            ]
        ].head(10).to_string(
            index=False
        )
    )

else:

    print("No action items extracted.")