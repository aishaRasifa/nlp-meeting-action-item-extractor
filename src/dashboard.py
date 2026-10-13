import streamlit as st
import pandas as pd
import torch
import re

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Meeting Action Item Extractor",
    page_icon="📋",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📋 Meeting Action Item Extractor")

st.write(
    "Upload or paste a meeting transcript to automatically "
    "identify action items, assignees, deadlines and scores."
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = "models/action_classifier_distilbert"


# ============================================================
# LOAD DISTILBERT MODEL
# ============================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


try:

    tokenizer, model = load_model()

except Exception as e:

    st.error(
        "Unable to load the DistilBERT model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("🔎 Model Information"):

    st.write(
        "Model: DistilBERT"
    )

    st.write(
        "Task: Action / Non-Action Classification"
    )

    st.write(
        "Model Path:",
        MODEL_PATH
    )

    st.write(
        "Label Mapping:",
        model.config.id2label
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_action(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            prediction
        ].item()

    return prediction, confidence


# ============================================================
# EXTRACT SPEAKER
# ============================================================

def extract_speaker(text):

    match = re.match(
        r"^\s*([A-Za-z]+)\s*:",
        text
    )

    if match:

        return match.group(1).strip()

    return "Unknown"


# ============================================================
# REMOVE SPEAKER PREFIX
# ============================================================

def remove_speaker_prefix(text):

    return re.sub(
        r"^\s*[A-Za-z]+\s*:\s*",
        "",
        text
    ).strip()


# ============================================================
# OBVIOUS NON-ACTION / CONVERSATIONAL STATEMENTS
# ============================================================

def is_likely_non_action(text):

    # Remove speaker name before applying rules
    text_lower = remove_speaker_prefix(
        text
    ).lower().strip()

    non_action_patterns = [

        # ----------------------------------------------------
        # Greetings
        # ----------------------------------------------------

        r"^good morning",
        r"^good afternoon",
        r"^good evening",
        r"^hello everyone",
        r"^hi everyone",
        r"^welcome everyone",

        # ----------------------------------------------------
        # Meeting discussion
        # ----------------------------------------------------

        r"today we need to discuss",
        r"today we will discuss",
        r"today let's discuss",
        r"today let us discuss",

        r"we need to discuss the progress",

        r"let's discuss",
        r"let us discuss",

        # ----------------------------------------------------
        # Opinions
        # ----------------------------------------------------

        r"^i think",
        r"^i believe",
        r"^i feel",

        # ----------------------------------------------------
        # Status statements
        # ----------------------------------------------------

        r"^the .* is almost complete",
        r"^the .* is complete",
        r"^the .* is ready",

        r"^the .* are almost complete",
        r"^the .* are complete",
        r"^the .* are ready",

        # ----------------------------------------------------
        # General status / opinion
        # ----------------------------------------------------

        r"^the .* looks good",
        r"^the .* looks great",

        r"^everything looks good",
        r"^everything is ready",

        # ----------------------------------------------------
        # Information/status statements
        # ----------------------------------------------------

        r"^we have received",
        r"^we already received",

        # ----------------------------------------------------
        # Accuracy/status statements
        # ----------------------------------------------------

        r"^the current accuracy is",
        r"^the accuracy is",

    ]

    for pattern in non_action_patterns:

        if re.search(
            pattern,
            text_lower
        ):

            return True

    return False


# ============================================================
# STRONG LINGUISTIC ACTION PATTERNS
# ============================================================

def strong_action_rule(text):

    text_lower = remove_speaker_prefix(
        text
    ).lower().strip()

    strong_patterns = [

        # ----------------------------------------------------
        # Explicit requests
        # ----------------------------------------------------

        r"\bplease\b",
        r"\bkindly\b",

        # ----------------------------------------------------
        # First-person commitments
        # ----------------------------------------------------

        r"\bi will\b",
        r"\bi'll\b",
        r"\bi can\b",
        r"\bi need to\b",
        r"\bi have to\b",
        r"\bi must\b",

        # ----------------------------------------------------
        # Team commitments
        # ----------------------------------------------------

        r"\bwe should\b",
        r"\bwe also should\b",

        r"\bwe need to\b",
        r"\bwe also need to\b",

        r"\bwe have to\b",
        r"\bwe also have to\b",

        r"\bwe will\b",
        r"\bwe can\b",
        r"\bwe must\b",

        # ----------------------------------------------------
        # Second-person assignments
        # ----------------------------------------------------

        r"\byou should\b",
        r"\byou need to\b",
        r"\byou have to\b",
        r"\byou will\b",
        r"\byou must\b",

        # ----------------------------------------------------
        # Obligation expressions
        # ----------------------------------------------------

        r"\bneeds to\b",
        r"\bneed to\b",
        r"\bhas to\b",
        r"\bhave to\b",
        r"\bmust\b",
        r"\bshould\b",

        # ----------------------------------------------------
        # Task verbs with context
        # ----------------------------------------------------

        r"\bprepare the\b",
        r"\bupdate the\b",
        r"\bvalidate the\b",
        r"\breview the\b",
        r"\bcheck the\b",
        r"\bcreate the\b",
        r"\bsubmit the\b",
        r"\btest the\b",
        r"\bdocument the\b",
        r"\bclean the\b",
        r"\bfix the\b",
        r"\bdevelop the\b",
        r"\bimplement the\b",
        r"\bimprove the\b",
        r"\banalyze the\b",
        r"\banalyse the\b",
        r"\bcomplete the\b",
        r"\bfinish the\b",

        # ----------------------------------------------------
        # Direct task expressions
        # ----------------------------------------------------

        r"\bprepare\b.*\bfor\b",
        r"\bupdate\b.*\bby\b",
        r"\bvalidate\b.*\band\b",
        r"\breview\b.*\bbefore\b",
        r"\bcheck\b.*\band\b",

    ]

    for pattern in strong_patterns:

        if re.search(
            pattern,
            text_lower
        ):

            return True

    return False


# ============================================================
# ASSIGNEE EXTRACTION
# ============================================================

def extract_assignee(text):

    text_clean = remove_speaker_prefix(
        text
    )

    text_lower = text_clean.lower()

    speaker = extract_speaker(
        text
    )

    # ========================================================
    # EXPLICIT PERSON ASSIGNMENT
    #
    # Example:
    # Peter, please validate the model.
    # Sarah, please prepare the samples.
    # ========================================================

    match = re.search(
        r"^\s*([A-Z][a-z]+)\s*,\s*"
        r"(please|kindly)",
        text_clean
    )

    if match:

        person = match.group(1)

        if person.lower() not in [
            "we",
            "i",
            "you",
            "they",
            "he",
            "she"
        ]:

            return person

    # ========================================================
    # PERSON + ACTION
    #
    # Example:
    # Peter should validate...
    # Sarah needs to prepare...
    # John will update...
    # ========================================================

    match = re.search(
        r"^\s*([A-Z][a-z]+)\s+"
        r"(should|needs to|has to|will|can|must)",
        text_clean
    )

    if match:

        person = match.group(1)

        if person.lower() not in [
            "we",
            "i",
            "you",
            "they",
            "he",
            "she"
        ]:

            return person

    # ========================================================
    # FIRST-PERSON COMMITMENT
    #
    # I will...
    # I'll...
    # I can...
    # I need to...
    # ========================================================

    if re.search(
        r"\b(i will|i'll|i can|i need to|"
        r"i have to|i must)\b",
        text_lower
    ):

        if speaker != "Unknown":

            return speaker

        return "Speaker"

    # ========================================================
    # TEAM COMMITMENT
    #
    # We should...
    # We also should...
    # We need to...
    # We also need to...
    # ========================================================

    if re.search(
        r"\b(we should|we also should|"
        r"we need to|we also need to|"
        r"we have to|we also have to|"
        r"we will|we can|we must)\b",
        text_lower
    ):

        return "Team"

    # ========================================================
    # SECOND PERSON
    # ========================================================

    if re.search(
        r"\b(you should|you need to|you have to|"
        r"you will|you must)\b",
        text_lower
    ):

        return "You"

    # ========================================================
    # GENERIC REQUEST
    # ========================================================

    if re.search(
        r"^\s*(please|kindly)\b",
        text_lower
    ):

        return "Team"

    # ========================================================
    # NO ASSIGNEE
    # ========================================================

    return "Not specified"


# ============================================================
# DEADLINE EXTRACTION
# ============================================================

def extract_deadline(text):

    text_clean = remove_speaker_prefix(
        text
    )

    text_lower = text_clean.lower()

    # ========================================================
    # BY TODAY
    # ========================================================

    if re.search(
        r"\bby\s+today\b",
        text_lower
    ):

        return "Today"

    # ========================================================
    # BY TOMORROW
    # ========================================================

    if re.search(
        r"\bby\s+tomorrow\b",
        text_lower
    ):

        return "Tomorrow"

    # ========================================================
    # BY WEEKDAY
    # ========================================================

    match = re.search(
        r"\bby\s+"
        r"(monday|tuesday|wednesday|"
        r"thursday|friday|saturday|sunday)\b",
        text_lower
    )

    if match:

        return match.group(1).capitalize()

    # ========================================================
    # BY NEXT WEEK
    # ========================================================

    if re.search(
        r"\bby\s+next\s+week\b",
        text_lower
    ):

        return "Next week"

    # ========================================================
    # BY NEXT MONTH
    # ========================================================

    if re.search(
        r"\bby\s+next\s+month\b",
        text_lower
    ):

        return "Next month"

    # ========================================================
    # DUE TODAY / TOMORROW
    # ========================================================

    match = re.search(
        r"\bdue\s+(today|tomorrow)\b",
        text_lower
    )

    if match:

        return match.group(1).capitalize()

    # ========================================================
    # CONTEXTUAL TODAY / TOMORROW
    #
    # Important:
    # "today" alone is NOT automatically a deadline.
    #
    # It must appear together with an action context.
    # ========================================================

    action_context = [

        "i will",
        "i'll",
        "i can",

        "we will",
        "we should",
        "we need to",
        "we also need to",

        "will",
        "need to",
        "needs to",
        "should",
        "have to",
        "has to",
        "must",

        "complete",
        "finish",
        "prepare",
        "validate",
        "update",
        "review",
        "check",
        "create",
        "submit",
        "test",
        "document",
        "clean",
        "fix",
        "improve",
        "analyze",
        "analyse"

    ]

    has_action_context = any(
        word in text_lower
        for word in action_context
    )

    if has_action_context:

        if re.search(
            r"\btoday\b",
            text_lower
        ):

            return "Today"

        if re.search(
            r"\btomorrow\b",
            text_lower
        ):

            return "Tomorrow"

    # ========================================================
    # NEXT WEEK
    # ========================================================

    if "next week" in text_lower:

        return "Next week"

    # ========================================================
    # NEXT MONTH
    # ========================================================

    if "next month" in text_lower:

        return "Next month"

    # ========================================================
    # NO DEADLINE
    # ========================================================

    return "Not specified"


# ============================================================
# SPLIT TRANSCRIPT
# ============================================================

def split_transcript(text):

    lines = text.splitlines()

    segments = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        line = re.sub(
            r"\s+",
            " ",
            line
        )

        segments.append(line)

    # ========================================================
    # If everything is pasted as one paragraph
    # ========================================================

    if len(segments) == 1:

        segments = re.split(
            r"(?<=[.!?])\s+",
            segments[0]
        )

        segments = [
            s.strip()
            for s in segments
            if s.strip()
        ]

    return segments


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

confidence_threshold = st.sidebar.slider(
    "Minimum Model Confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.05
)

st.sidebar.info(
    "Final action detection combines "
    "DistilBERT predictions with linguistic rules."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header("1️⃣ Meeting Transcript")

uploaded_file = st.file_uploader(
    "Upload a meeting transcript",
    type=["txt"]
)

st.write(
    "**OR paste the transcript below:**"
)

pasted_text = st.text_area(
    "Meeting Transcript",
    height=250,
    placeholder=(
        "Example:\n\n"
        "John: We should prepare the report by Friday.\n"
        "Sarah: I will update the database tomorrow.\n"
        "Peter: I think the current design looks good.\n"
        "John: Peter, please validate the model results."
    )
)


# ============================================================
# GET TRANSCRIPT
# ============================================================

transcript_text = ""

if uploaded_file is not None:

    transcript_text = (
        uploaded_file
        .read()
        .decode("utf-8")
    )

elif pasted_text.strip():

    transcript_text = pasted_text


# ============================================================
# EXTRACT BUTTON
# ============================================================

extract_button = st.button(
    "🔍 Extract Action Items",
    type="primary",
    use_container_width=True
)


# ============================================================
# PROCESS TRANSCRIPT
# ============================================================

if extract_button:

    if not transcript_text.strip():

        st.warning(
            "Please upload or paste a transcript."
        )

        st.stop()

    # ========================================================
    # SPLIT TRANSCRIPT
    # ========================================================

    segments = split_transcript(
        transcript_text
    )

    st.info(
        f"Processing {len(segments)} transcript segments..."
    )

    # ========================================================
    # MODEL PREDICTIONS
    # ========================================================

    raw_predictions = []

    progress = st.progress(0)

    for i, segment in enumerate(segments):

        prediction, model_confidence = (
            predict_action(segment)
        )

        # ----------------------------------------------------
        # MODEL LABEL
        #
        # Assumption:
        # 0 = NON_ACTION
        # 1 = ACTION
        # ----------------------------------------------------

        model_action = (
            prediction == 1
        )

        # ----------------------------------------------------
        # LINGUISTIC RULE
        # ----------------------------------------------------

        rule_action = strong_action_rule(
            segment
        )

        # ----------------------------------------------------
        # OBVIOUS NON-ACTION
        # ----------------------------------------------------

        obvious_non_action = (
            is_likely_non_action(
                segment
            )
        )

        # ----------------------------------------------------
        # FINAL HYBRID DECISION
        # ----------------------------------------------------

        if obvious_non_action:

            final_action = False

        elif model_action:

            final_action = True

        elif rule_action:

            final_action = True

        else:

            final_action = False

        raw_predictions.append({

            "text":
                segment,

            "model_action":
                model_action,

            "rule_action":
                rule_action,

            "final_action":
                final_action,

            "model_confidence":
                model_confidence

        })

        progress.progress(
            (i + 1) / len(segments)
        )

    # ========================================================
    # BUILD FINAL ACTION ITEMS
    # ========================================================

    results = []

    for item in raw_predictions:

        if not item["final_action"]:

            continue

        text = item["text"]

        model_action = item[
            "model_action"
        ]

        model_confidence = item[
            "model_confidence"
        ]

        # ====================================================
        # SCORE
        # ====================================================

        if model_action:

            score = model_confidence

            detection_method = "Model"

        else:

            score = 0.75

            detection_method = "Rule"

        # ====================================================
        # CONFIDENCE FILTER
        # ====================================================

        if (
            model_action
            and
            score < confidence_threshold
        ):

            continue

        # ====================================================
        # EXTRACT INFORMATION
        # ====================================================

        speaker = extract_speaker(
            text
        )

        task = remove_speaker_prefix(
            text
        )

        assignee = extract_assignee(
            text
        )

        deadline = extract_deadline(
            text
        )

        results.append({

            "Action Item ID":
                f"AI-{len(results) + 1:03d}",

            "Speaker":
                speaker,

            "Task":
                task,

            "Assignee":
                assignee,

            "Deadline":
                deadline,

            "Score":
                round(
                    score,
                    3
                ),

            "Detection":
                detection_method

        })

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    st.header(
        "2️⃣ Extraction Summary"
    )

    total_segments = len(
        segments
    )

    total_actions = len(
        result_df
    )

    total_non_actions = (
        total_segments
        - total_actions
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Transcript Segments",
        total_segments
    )

    col2.metric(
        "Action Items",
        total_actions
    )

    col3.metric(
        "Non-Action Segments",
        total_non_actions
    )

    # ========================================================
    # EXTRACTED ACTION ITEMS
    # ========================================================

    st.header(
        "3️⃣ Extracted Action Items"
    )

    if result_df.empty:

        st.warning(
            "No action items were detected."
        )

    else:

        st.success(
            f"{len(result_df)} action items detected."
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # ACTION ITEM ANALYSIS
        # ====================================================

        st.header(
            "4️⃣ Action Item Analysis"
        )

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # ASSIGNEE DISTRIBUTION
        # ----------------------------------------------------

        with col1:

            st.subheader(
                "👤 Assignee Distribution"
            )

            assignee_counts = (
                result_df[
                    "Assignee"
                ]
                .value_counts()
                .reset_index()
            )

            assignee_counts.columns = [
                "Assignee",
                "Count"
            ]

            st.bar_chart(
                assignee_counts.set_index(
                    "Assignee"
                )
            )

        # ----------------------------------------------------
        # DEADLINE DISTRIBUTION
        # ----------------------------------------------------

        with col2:

            st.subheader(
                "📅 Deadline Distribution"
            )

            deadline_counts = (
                result_df[
                    "Deadline"
                ]
                .value_counts()
                .reset_index()
            )

            deadline_counts.columns = [
                "Deadline",
                "Count"
            ]

            st.bar_chart(
                deadline_counts.set_index(
                    "Deadline"
                )
            )

        # ====================================================
        # SCORE DISTRIBUTION
        # ====================================================

        st.subheader(
            "📊 Action Item Scores"
        )

        score_chart = (
            result_df[
                [
                    "Action Item ID",
                    "Score"
                ]
            ]
            .set_index(
                "Action Item ID"
            )
        )

        st.bar_chart(
            score_chart
        )

        # ====================================================
        # DETECTION METHOD
        # ====================================================

        st.subheader(
            "🔍 Detection Method"
        )

        detection_counts = (
            result_df[
                "Detection"
            ]
            .value_counts()
            .reset_index()
        )

        detection_counts.columns = [
            "Detection Method",
            "Count"
        ]

        st.dataframe(
            detection_counts,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        st.header(
            "5️⃣ Download Results"
        )

        csv_data = result_df.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Action Items CSV",
            data=csv_data,
            file_name=(
                "extracted_action_items.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NLP Meeting Analyzer | "
    "DistilBERT + Linguistic Rule Refinement"
)