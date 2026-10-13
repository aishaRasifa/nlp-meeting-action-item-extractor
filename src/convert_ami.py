from pathlib import Path
import xml.etree.ElementTree as ET
import re
import csv


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AMI_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ami_public_manual_1.6.2"
)

SEGMENTS_FOLDER = AMI_FOLDER / "segments"
WORDS_FOLDER = AMI_FOLDER / "words"

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_FOLDER
    / "EN2001a_transcript.csv"
)


# ============================================================
# 2. LOAD WORDS AND WORD-RELATED ANNOTATIONS
# ============================================================

def load_words(words_file):

    tree = ET.parse(words_file)
    root = tree.getroot()

    words = []

    for element in root.iter():

        # Get NITE ID
        element_id = element.get(
            "{http://nite.sourceforge.net/}id"
        )

        # Ignore elements without ID
        if element_id is None:
            continue

        # Keep elements that can be referenced
        # by the segments file
        if element.tag not in [
            "w",
            "disfmarker",
            "gap",
            "vocalsound"
        ]:
            continue

        # Get text
        element_text = element.text or ""

        words.append(
            (element_id, element_text)
        )

    return words


# ============================================================
# 3. LOAD SEGMENTS
# ============================================================

def load_segments(segments_file):

    tree = ET.parse(segments_file)
    root = tree.getroot()

    segments = []

    for segment in root.findall(".//segment"):

        # Get segment ID
        segment_id = segment.get(
            "{http://nite.sourceforge.net/}id"
        )

        # Get start time
        start_time = segment.get(
            "transcriber_start"
        )

        # Get end time
        end_time = segment.get(
            "transcriber_end"
        )

        # Find NITE child
        child = segment.find(
            "nite:child",
            {
                "nite": "http://nite.sourceforge.net/"
            }
        )

        # If no child reference exists
        if child is None:
            continue

        # Get href
        href = child.get(
            "href",
            ""
        )

        segments.append({
            "segment_id": segment_id,
            "start_time": start_time,
            "end_time": end_time,
            "href": href
        })

    return segments


# ============================================================
# 4. CREATE WORD INDEX
# ============================================================

def create_word_index(words):

    word_index = {}

    for position, (word_id, word_text) in enumerate(words):

        word_index[word_id] = position

    return word_index


# ============================================================
# 5. EXTRACT START AND END WORD IDs
# ============================================================

def extract_word_ids(href):

    pattern = (
        r"#id\(([^)]+)\)"
        r"(?:\.\.id\(([^)]+)\))?"
    )

    match = re.search(
        pattern,
        href
    )

    if not match:
        return None, None

    start_id = match.group(1)

    end_id = match.group(2)

    # If only one ID exists
    if end_id is None:
        end_id = start_id

    return start_id, end_id


# ============================================================
# 6. BUILD TEXT FOR A SEGMENT
# ============================================================

def build_segment_text(
    start_id,
    end_id,
    words,
    word_index
):

    start_position = word_index[start_id]

    end_position = word_index[end_id]

    selected_elements = words[
        start_position:end_position + 1
    ]

    texts = []

    for element_id, element_text in selected_elements:

        # Ignore empty annotations
        if element_text.strip():

            texts.append(
                element_text.strip()
            )

    # Join words
    text = " ".join(texts)

    # Remove spaces before punctuation
    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text
    )

    # Remove unnecessary spaces around brackets
    text = re.sub(
        r"\(\s+",
        "(",
        text
    )

    text = re.sub(
        r"\s+\)",
        ")",
        text
    )

    return text


# ============================================================
# 7. FIND ALL CHANNEL FILES
# ============================================================

def find_channel_files():

    channel_files = []

    # Find all segment XML files
    segment_files = sorted(
        SEGMENTS_FOLDER.glob("EN2001a.*.segments.xml")
    )

    for segments_file in segment_files:

        filename = segments_file.name

        # Example:
        # EN2001a.A.segments.xml

        match = re.match(
            r"EN2001a\.([A-Z])\.segments\.xml$",
            filename
        )

        if not match:
            continue

        speaker = match.group(1)

        # Corresponding words file
        words_file = (
            WORDS_FOLDER
            / f"EN2001a.{speaker}.words.xml"
        )

        # Check if words file exists
        if not words_file.exists():

            print(
                f"WARNING: Words file not found for speaker {speaker}"
            )

            continue

        channel_files.append({
            "speaker": speaker,
            "segments_file": segments_file,
            "words_file": words_file
        })

    return channel_files


# ============================================================
# 8. PROCESS ONE CHANNEL
# ============================================================

def process_channel(
    speaker,
    segments_file,
    words_file
):

    print("\n" + "=" * 60)

    print(
        f"PROCESSING SPEAKER {speaker}"
    )

    print("=" * 60)

    print(
        "Segments file:",
        segments_file.name
    )

    print(
        "Words file:",
        words_file.name
    )

    # --------------------------------------------------------
    # Load words
    # --------------------------------------------------------

    words = load_words(
        words_file
    )

    print(
        "Word-related elements loaded:",
        len(words)
    )

    # --------------------------------------------------------
    # Create word index
    # --------------------------------------------------------

    word_index = create_word_index(
        words
    )

    # --------------------------------------------------------
    # Load segments
    # --------------------------------------------------------

    segments = load_segments(
        segments_file
    )

    print(
        "Segments loaded:",
        len(segments)
    )

    # --------------------------------------------------------
    # Create transcript rows
    # --------------------------------------------------------

    rows = []

    skipped_segments = []

    for segment in segments:

        # Extract start and end IDs
        start_id, end_id = extract_word_ids(
            segment["href"]
        )

        # ----------------------------------------------------
        # Check IDs
        # ----------------------------------------------------

        if start_id is None:

            skipped_segments.append(
                (
                    segment["segment_id"],
                    "Could not extract word IDs"
                )
            )

            continue

        # ----------------------------------------------------
        # Check starting ID
        # ----------------------------------------------------

        if start_id not in word_index:

            skipped_segments.append(
                (
                    segment["segment_id"],
                    f"Start ID not found: {start_id}"
                )
            )

            continue

        # ----------------------------------------------------
        # Check ending ID
        # ----------------------------------------------------

        if end_id not in word_index:

            skipped_segments.append(
                (
                    segment["segment_id"],
                    f"End ID not found: {end_id}"
                )
            )

            continue

        # ----------------------------------------------------
        # Build transcript text
        # ----------------------------------------------------

        text = build_segment_text(
            start_id,
            end_id,
            words,
            word_index
        )

        # ----------------------------------------------------
        # Create row
        # ----------------------------------------------------

        rows.append({

            "meeting_id": "EN2001a",

            "speaker": speaker,

            "segment_id": segment["segment_id"],

            "start_time": segment["start_time"],

            "end_time": segment["end_time"],

            "text": text
        })

    # --------------------------------------------------------
    # Print channel result
    # --------------------------------------------------------

    print(
        "Transcript segments created:",
        len(rows)
    )

    print(
        "Skipped segments:",
        len(skipped_segments)
    )

    if skipped_segments:

        print("\nSkipped segment details:")

        for segment_id, reason in skipped_segments:

            print(
                segment_id,
                "->",
                reason
            )

    else:

        print(
            "No segments were skipped!"
        )

    return rows


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)

    print(
        "AMI MEETING TRANSCRIPT EXTRACTION"
    )

    print("=" * 60)

    print(
        "\nAMI folder:",
        AMI_FOLDER
    )

    # --------------------------------------------------------
    # Find A, B, C, D, E automatically
    # --------------------------------------------------------

    channel_files = find_channel_files()

    if not channel_files:

        print(
            "\nERROR: No channel files found."
        )

        print(
            "Please check the segments and words folders."
        )

        return

    print(
        "\nChannels found:",
        len(channel_files)
    )

    for channel in channel_files:

        print(
            f"Speaker {channel['speaker']} -> "
            f"{channel['segments_file'].name}"
        )

    # --------------------------------------------------------
    # Process every channel
    # --------------------------------------------------------

    all_rows = []

    for channel in channel_files:

        rows = process_channel(

            speaker=channel["speaker"],

            segments_file=channel["segments_file"],

            words_file=channel["words_file"]
        )

        all_rows.extend(rows)

    # --------------------------------------------------------
    # Sort entire meeting chronologically
    # --------------------------------------------------------

    all_rows.sort(
        key=lambda row: float(
            row["start_time"]
        )
    )

    # --------------------------------------------------------
    # Save combined transcript
    # --------------------------------------------------------

    fieldnames = [
        "meeting_id",
        "speaker",
        "segment_id",
        "start_time",
        "end_time",
        "text"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            all_rows
        )

    # ========================================================
    # 10. FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "TRANSCRIPT EXTRACTION COMPLETED"
    )

    print("=" * 60)

    print(
        "\nTotal channels processed:",
        len(channel_files)
    )

    print(
        "Total transcript segments:",
        len(all_rows)
    )

    print(
        "\nCSV saved at:"
    )

    print(
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Show first 10 rows
    # --------------------------------------------------------

    print(
        "\nFirst 10 transcript rows:"
    )

    print("-" * 60)

    for row in all_rows[:10]:

        print(
            f"[{row['start_time']}] "
            f"Speaker {row['speaker']}: "
            f"{row['text']}"
        )

    # --------------------------------------------------------
    # Show speaker-wise counts
    # --------------------------------------------------------

    print(
        "\nSpeaker-wise segment count:"
    )

    speaker_counts = {}

    for row in all_rows:

        speaker = row["speaker"]

        speaker_counts[speaker] = (
            speaker_counts.get(
                speaker,
                0
            ) + 1
        )

    for speaker in sorted(
        speaker_counts
    ):

        print(
            f"Speaker {speaker}: "
            f"{speaker_counts[speaker]} segments"
        )


# ============================================================
# 11. RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()