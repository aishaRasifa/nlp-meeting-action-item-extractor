import pandas as pd
from pathlib import Path
from collections import Counter


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

ANALYSIS_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "analysis"
)

ANALYSIS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD PROCESSED DATA
# ============================================================

print("=" * 60)
print("BASIC NLP ANALYSIS")
print("=" * 60)

print("\nLoading processed transcript...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Total transcript rows:",
    len(df)
)


# ============================================================
# 3. TOP KEYWORDS
# ============================================================

print("\n" + "=" * 60)
print("TOP 20 KEYWORDS")
print("=" * 60)


all_lemmas = []

for text in df["lemmas"].fillna(""):

    words = text.split()

    all_lemmas.extend(words)


# Count lemma frequency
lemma_counts = Counter(
    all_lemmas
)


# Get top 20
top_keywords = (
    lemma_counts
    .most_common(20)
)


print(
    "\nRank | Keyword | Frequency"
)

print("-" * 35)

for rank, (word, count) in enumerate(
    top_keywords,
    start=1
):

    print(
        f"{rank:4} | {word:15} | {count}"
    )


# Save keywords
keywords_df = pd.DataFrame(
    top_keywords,
    columns=[
        "keyword",
        "frequency"
    ]
)

keywords_df.insert(
    0,
    "rank",
    range(
        1,
        len(keywords_df) + 1
    )
)

keywords_df.to_csv(
    ANALYSIS_FOLDER
    / "top_20_keywords.csv",
    index=False
)


# ============================================================
# 4. TOP NOUNS
# ============================================================

print("\n" + "=" * 60)
print("TOP 20 NOUNS")
print("=" * 60)


noun_counter = Counter()


for pos_text in df["pos_tags"].fillna(""):

    tags = pos_text.split()

    for tag in tags:

        # Format:
        # word/POS

        if "/" not in tag:
            continue

        word, pos = tag.rsplit(
            "/",
            1
        )

        if pos in [
            "NOUN",
            "PROPN"
        ]:

            noun_counter[
                word.lower()
            ] += 1


top_nouns = noun_counter.most_common(20)


print(
    "\nRank | Noun | Frequency"
)

print("-" * 35)

for rank, (word, count) in enumerate(
    top_nouns,
    start=1
):

    print(
        f"{rank:4} | {word:15} | {count}"
    )


# Save nouns
nouns_df = pd.DataFrame(
    top_nouns,
    columns=[
        "noun",
        "frequency"
    ]
)

nouns_df.insert(
    0,
    "rank",
    range(
        1,
        len(nouns_df) + 1
    )
)

nouns_df.to_csv(
    ANALYSIS_FOLDER
    / "top_20_nouns.csv",
    index=False
)


# ============================================================
# 5. TOP VERBS
# ============================================================

print("\n" + "=" * 60)
print("TOP 20 VERBS")
print("=" * 60)


verb_counter = Counter()


for pos_text in df["pos_tags"].fillna(""):

    tags = pos_text.split()

    for tag in tags:

        if "/" not in tag:
            continue

        word, pos = tag.rsplit(
            "/",
            1
        )

        if pos in [
            "VERB",
            "AUX"
        ]:

            verb_counter[
                word.lower()
            ] += 1


top_verbs = verb_counter.most_common(20)


print(
    "\nRank | Verb | Frequency"
)

print("-" * 35)

for rank, (word, count) in enumerate(
    top_verbs,
    start=1
):

    print(
        f"{rank:4} | {word:15} | {count}"
    )


# Save verbs
verbs_df = pd.DataFrame(
    top_verbs,
    columns=[
        "verb",
        "frequency"
    ]
)

verbs_df.insert(
    0,
    "rank",
    range(
        1,
        len(verbs_df) + 1
    )
)

verbs_df.to_csv(
    ANALYSIS_FOLDER
    / "top_20_verbs.csv",
    index=False
)


# ============================================================
# 6. NAMED ENTITY ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("NAMED ENTITY ANALYSIS")
print("=" * 60)


entity_counter = Counter()


for entity_text in df[
    "named_entities"
].fillna(""):

    entities = entity_text.split("|")

    for entity in entities:

        entity = entity.strip()

        if not entity:
            continue

        if "/" not in entity:
            continue

        entity_name, entity_type = entity.rsplit(
            "/",
            1
        )

        key = (
            entity_name,
            entity_type
        )

        entity_counter[key] += 1


top_entities = (
    entity_counter
    .most_common(20)
)


print(
    "\nRank | Entity | Type | Frequency"
)

print("-" * 50)

for rank, ((name, entity_type), count) in enumerate(
    top_entities,
    start=1
):

    print(
        f"{rank:4} | "
        f"{name:20} | "
        f"{entity_type:10} | "
        f"{count}"
    )


# Save entities
entity_rows = []

for rank, ((name, entity_type), count) in enumerate(
    top_entities,
    start=1
):

    entity_rows.append({

        "rank": rank,

        "entity": name,

        "entity_type": entity_type,

        "frequency": count
    })


entities_df = pd.DataFrame(
    entity_rows
)

entities_df.to_csv(
    ANALYSIS_FOLDER
    / "top_20_entities.csv",
    index=False
)


# ============================================================
# 7. SPEAKER CONTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("SPEAKER CONTRIBUTION")
print("=" * 60)


speaker_counts = (
    df["speaker"]
    .value_counts()
    .sort_index()
)


total_segments = len(df)


speaker_rows = []


print(
    "\nSpeaker | Segments | Percentage"
)

print("-" * 40)


for speaker, count in speaker_counts.items():

    percentage = (
        count
        / total_segments
        * 100
    )

    print(
        f"{speaker:^7} | "
        f"{count:^8} | "
        f"{percentage:.2f}%"
    )

    speaker_rows.append({

        "speaker": speaker,

        "segments": count,

        "percentage": round(
            percentage,
            2
        )
    })


speaker_df = pd.DataFrame(
    speaker_rows
)


speaker_df.to_csv(
    ANALYSIS_FOLDER
    / "speaker_contribution.csv",
    index=False
)


# ============================================================
# 8. BASIC TRANSCRIPT STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("BASIC TRANSCRIPT STATISTICS")
print("=" * 60)


total_tokens = sum(
    len(str(text).split())
    for text in df["tokens"].fillna("")
)


total_filtered_tokens = sum(
    len(str(text).split())
    for text in df["filtered_tokens"].fillna("")
)


total_lemmas = sum(
    len(str(text).split())
    for text in df["lemmas"].fillna("")
)


unique_lemmas = len(
    set(all_lemmas)
)


total_entities = sum(
    len(
        [
            e
            for e in str(text).split("|")
            if e.strip()
        ]
    )
    for text in df["named_entities"].fillna("")
)


print(
    "\nTotal transcript segments:",
    len(df)
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

print(
    "Unique lemmas:",
    unique_lemmas
)

print(
    "Named entity mentions:",
    total_entities
)

print(
    "Number of speakers:",
    df["speaker"].nunique()
)


# ============================================================
# 9. SAVE BASIC STATISTICS
# ============================================================

statistics = pd.DataFrame({

    "metric": [
        "Transcript segments",
        "Total tokens",
        "Tokens after stopword removal",
        "Total lemmas",
        "Unique lemmas",
        "Named entity mentions",
        "Number of speakers"
    ],

    "value": [
        len(df),
        total_tokens,
        total_filtered_tokens,
        total_lemmas,
        unique_lemmas,
        total_entities,
        df["speaker"].nunique()
    ]
})


statistics.to_csv(
    ANALYSIS_FOLDER
    / "basic_statistics.csv",
    index=False
)


# ============================================================
# 10. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)

print(
    "NLP ANALYSIS COMPLETED"
)

print("=" * 60)

print(
    "\nAnalysis files saved in:"
)

print(
    ANALYSIS_FOLDER
)

print(
    "\nGenerated files:"
)

print(
    "1. top_20_keywords.csv"
)

print(
    "2. top_20_nouns.csv"
)

print(
    "3. top_20_verbs.csv"
)

print(
    "4. top_20_entities.csv"
)

print(
    "5. speaker_contribution.csv"
)

print(
    "6. basic_statistics.csv"
)

print(
    "\nStudent 1 NLP analysis completed!"
)