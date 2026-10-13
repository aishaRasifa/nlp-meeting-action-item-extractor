import pandas as pd
import spacy
import re
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "EN2001a_transcript_cleaned.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "EN2001a_transcript_processed.csv"
)


# ============================================================
# 2. LOAD SPACY MODEL
# ============================================================

print("=" * 60)
print("NLP PREPROCESSING")
print("=" * 60)

print("\nLoading spaCy model...")

nlp = spacy.load(
    "en_core_web_sm"
)

print("spaCy model loaded successfully!")


# ============================================================
# 3. LOAD CLEANED TRANSCRIPT
# ============================================================

print("\nLoading transcript...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Total transcript rows:",
    len(df)
)


# ============================================================
# 4. BASIC TEXT CLEANING
# ============================================================

def clean_text(text):

    # Convert to string
    text = str(text)

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove leading/trailing spaces
    text = text.strip()

    return text


df["clean_text"] = df["text"].apply(
    clean_text
)


# ============================================================
# 5. PROCESS TEXT USING SPACY
# ============================================================

print("\nProcessing transcript with spaCy...")

processed_rows = []


for index, row in df.iterrows():

    text = row["clean_text"]

    # Process text
    doc = nlp(text)

    # --------------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------------

    tokens = []

    for token in doc:

        if not token.is_space:

            tokens.append(
                token.text
            )


    # --------------------------------------------------------
    # STOPWORD REMOVAL
    # --------------------------------------------------------

    filtered_tokens = []

    for token in doc:

        if (
            not token.is_space
            and not token.is_stop
            and not token.is_punct
        ):

            filtered_tokens.append(
                token.text
            )


    # --------------------------------------------------------
    # LEMMATIZATION
    # --------------------------------------------------------

    lemmas = []

    for token in doc:

        if (
            not token.is_space
            and not token.is_stop
            and not token.is_punct
        ):

            lemma = token.lemma_

            lemmas.append(
                lemma.lower()
            )


    # --------------------------------------------------------
    # POS TAGGING
    # --------------------------------------------------------

    pos_tags = []

    for token in doc:

        if not token.is_space:

            pos_tags.append(
                f"{token.text}/{token.pos_}"
            )


    # --------------------------------------------------------
    # NAMED ENTITY RECOGNITION
    # --------------------------------------------------------

    entities = []

    for ent in doc.ents:

        entities.append(
            f"{ent.text}/{ent.label_}"
        )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    processed_rows.append({

        "meeting_id":
            row["meeting_id"],

        "speaker":
            row["speaker"],

        "segment_id":
            row["segment_id"],

        "start_time":
            row["start_time"],

        "end_time":
            row["end_time"],

        "text":
            row["text"],

        "clean_text":
            row["clean_text"],

        "tokens":
            " ".join(tokens),

        "filtered_tokens":
            " ".join(filtered_tokens),

        "lemmas":
            " ".join(lemmas),

        "pos_tags":
            " ".join(pos_tags),

        "named_entities":
            " | ".join(entities)
    })


# ============================================================
# 6. CREATE PROCESSED DATAFRAME
# ============================================================

processed_df = pd.DataFrame(
    processed_rows
)


# ============================================================
# 7. SAVE PROCESSED DATASET
# ============================================================

processed_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)

print(
    "NLP PREPROCESSING COMPLETED"
)

print("=" * 60)

print(
    "\nTotal processed rows:",
    len(processed_df)
)

print(
    "\nOutput file:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 9. DISPLAY SAMPLE
# ============================================================

print(
    "\nSample processed transcript:"
)

print("-" * 60)

for _, row in processed_df.head(5).iterrows():

    print(
        f"\nSpeaker {row['speaker']}"
    )

    print(
        "Original:",
        row["text"]
    )

    print(
        "Tokens:",
        row["tokens"]
    )

    print(
        "After stopword removal:",
        row["filtered_tokens"]
    )

    print(
        "Lemmas:",
        row["lemmas"]
    )

    print(
        "POS:",
        row["pos_tags"]
    )

    print(
        "Entities:",
        row["named_entities"]
    )


# ============================================================
# 10. BASIC STATISTICS
# ============================================================

print("\n" + "=" * 60)

print(
    "BASIC NLP STATISTICS"
)

print("=" * 60)


# Number of tokens
total_tokens = 0

for tokens in processed_df["tokens"]:

    if isinstance(tokens, str):

        total_tokens += len(
            tokens.split()
        )


# Number of filtered tokens
total_filtered_tokens = 0

for tokens in processed_df["filtered_tokens"]:

    if isinstance(tokens, str):

        total_filtered_tokens += len(
            tokens.split()
        )


# Number of lemmas
total_lemmas = 0

for lemmas in processed_df["lemmas"]:

    if isinstance(lemmas, str):

        total_lemmas += len(
            lemmas.split()
        )


print(
    "Total tokens:",
    total_tokens
)

print(
    "Tokens after stopword removal:",
    total_filtered_tokens
)

print(
    "Total lemmas:",
    total_lemmas
)


# ============================================================
# 11. SPEAKER-WISE SEGMENT COUNT
# ============================================================

print(
    "\nSpeaker-wise transcript segments:"
)

speaker_counts = (
    processed_df["speaker"]
    .value_counts()
    .sort_index()
)

for speaker, count in speaker_counts.items():

    print(
        f"Speaker {speaker}: {count}"
    )


print("\nProcessing finished successfully!")