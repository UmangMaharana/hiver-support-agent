from pathlib import Path
import re

import pandas as pd
import html


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "retrieval" / "tesco_retrieval_corpus.parquet"
OUTPUT_PATH = ROOT / "data" / "retrieval" / "tesco_resolution_corpus.parquet"


# These signals indicate that a Tesco response is doing something
# operational rather than merely acknowledging the customer.
ACTION_PATTERNS = [
    # Contact / information gathering
    r"\bdm\b",
    r"\bplease (?:dm|message|contact|email|call)\b",
    r"\bcan you (?:dm|message|contact|email|call)\b",
    r"\bprovide\b.*\b(?:details|information|name|address|order)\b",
    r"\bconfirm\b.*\b(?:details|information|name|address|order)\b",

    # Investigation / checking
    r"\b(?:check|checking|look|looking|investigate|investigating)\b",
    r"\bi(?:'ll| will) (?:check|look|investigate)\b",
    r"\bwe(?:'ll| will) (?:check|look|investigate)\b",

    # Refund / compensation
    r"\brefund\b",
    r"\brefund(?:ed|ing)?\b",
    r"\breimburse(?:d|ment)?\b",
    r"\bcompensation\b",
    r"\bmone(?:y|yback) back\b",
    r"\btesco (?:money|gift)card\b",
    r"\bvoucher\b",

    # Replacement / exchange
    r"\breplace(?:d|ment)?\b",
    r"\bexchange\b",
    r"\breturn\b.*\b(?:store|item|product|order)\b",

    # Escalation / internal action
    r"\bescalat(?:e|ed|ing|ion)\b",
    r"\bmanagement\b",
    r"\bstore manager\b",
    r"\bstore management\b",
    r"\bteam\b.*\b(?:look|check|investigate|contact)\b",
    r"\bpass(?:ed)? .*?(?:details|feedback|information) .*?\b(?:team|management|store)\b",
    r"\bfeedback\b",

    # Delivery / order actions
    r"\border\b.*\b(?:check|look|investigate|track)\b",
    r"\bdelivery\b.*\b(?:check|look|investigate|arrange)\b",
    r"\btrack(?:ing)?\b",

    # Advice / concrete next steps
    r"\byou(?:'ll| will) need to\b",
    r"\byou can\b.*\b(?:return|visit|contact|call|dm|email)\b",
    r"\bplease (?:return|visit|contact|call|dm|email)\b",
    r"\bwe(?:'ll| will) (?:arrange|send|provide|process|sort)\b",
    r"\bi(?:'ll| will) (?:arrange|send|provide|process|sort)\b",
]


COMPILED_ACTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in ACTION_PATTERNS
]


def extract_tesco_turns(transcript):
    """
    Extract Tesco turns from the reconstructed transcript.

    A turn begins with 'TESCO:' and continues until the next
    CUSTOMER: or TESCO: turn.
    """
    lines = transcript.splitlines()

    turns = []
    current = None

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("TESCO:"):
            if current is not None:
                turns.append(current.strip())

            current = stripped[len("TESCO:"):].strip()

        elif stripped.startswith("CUSTOMER:"):
            if current is not None:
                turns.append(current.strip())
                current = None

        elif current is not None and stripped:
            current += " " + stripped

    if current is not None:
        turns.append(current.strip())

    return [turn for turn in turns if turn]


def is_action_bearing(turn):
    return any(
        pattern.search(turn)
        for pattern in COMPILED_ACTION_PATTERNS
    )


def extract_resolution(transcript):
    """
    Select the latest action-bearing Tesco response.

    If no action-bearing response exists, fall back to the
    final Tesco response.
    """
    tesco_turns = extract_tesco_turns(transcript)

    if not tesco_turns:
        return "", "no_tesco_response"

    action_turns = [
        turn
        for turn in tesco_turns
        if is_action_bearing(turn)
    ]

    if action_turns:
        return action_turns[-1], "latest_action_bearing"

    return tesco_turns[-1], "final_tesco_response"


def clean_tesco_response(text):
    """
    Remove dataset-specific agent handles and message-part suffixes
    while preserving the actual Tesco response.
    """
    text = str(text).strip()

    # Remove leading agent handle, e.g. "@223459 "
    text = re.sub(r"^@\d+\s*", "", text)

    # Remove trailing multipart markers such as "1/4", "2/4", etc.
    text = re.sub(r"\s+\d+/\d+\s*$", "", text)

    text = html.unescape(text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    print("Loading retrieval corpus...")
    corpus = pd.read_parquet(INPUT_PATH)

    print(f"Input cases: {len(corpus)}")

    resolutions = []
    resolution_sources = []
    tesco_turn_counts = []

    for transcript in corpus["transcript"]:
        resolution, source = extract_resolution(str(transcript))

        tesco_turns = extract_tesco_turns(str(transcript))

        resolutions.append(clean_tesco_response(resolution))
        resolution_sources.append(source)
        tesco_turn_counts.append(len(tesco_turns))

    corpus = corpus.copy()

    corpus["resolution"] = resolutions
    corpus["resolution_source"] = resolution_sources
    corpus["tesco_turn_count"] = tesco_turn_counts

    # Remove cases where no Tesco response exists.
    corpus = corpus[
        corpus["resolution"].notna()
        & (corpus["resolution"].str.strip() != "")
    ].copy()

    output_columns = [
        "case_id",
        "initial_customer_message",
        "resolution",
        "resolution_source",
        "tesco_turn_count",
        "transcript",
    ]

    corpus = corpus[output_columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(OUTPUT_PATH, index=False)

    print("\n" + "=" * 80)
    print("HISTORICAL RESOLUTION CORPUS")
    print("=" * 80)

    print(f"Cases with Tesco response: {len(corpus)}")
    print("\nResolution source:")
    print(corpus["resolution_source"].value_counts())

    print("\nAverage Tesco turns per case:")
    print(f"{corpus['tesco_turn_count'].mean():.2f}")

    print("\nExample resolutions:")

    for _, row in corpus.head(5).iterrows():
        print("\n" + "-" * 80)
        print(f"Case: {row['case_id']}")
        print(f"Customer: {row['initial_customer_message']}")
        print(f"Source: {row['resolution_source']}")
        print(f"Resolution: {row['resolution']}")

    print("\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()