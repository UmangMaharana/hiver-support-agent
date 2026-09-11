from pathlib import Path

import pandas as pd


CASES_PATH = Path("data/processed/tesco_cases.parquet")
REVIEW_PATH = Path("data/golden/taxonomy_review_100.csv")
OUTPUT_PATH = Path("data/golden/golden_candidates_250.csv")

RANDOM_SEED = 42
N_CANDIDATES = 250


def main() -> None:
    cases = pd.read_parquet(CASES_PATH)
    review = pd.read_csv(REVIEW_PATH)

    # Taxonomy review examples must never enter the golden set.
    used_case_ids = set(review["case_id"].astype(str))

    eligible = cases[
        ~cases["case_id"].astype(str).isin(used_case_ids)
    ].copy()

    # Only messages suitable for intent classification.
    eligible = eligible[
        eligible["initial_customer_message"].notna()
        & eligible["initial_customer_message"].str.strip().ne("")
    ].copy()

    sample = eligible.sample(
        n=N_CANDIDATES,
        random_state=RANDOM_SEED,
    ).copy()

    sample.insert(
        0,
        "golden_id",
        [f"GOLD_{i:03d}" for i in range(1, len(sample) + 1)],
    )

    sample["intent"] = ""
    sample["confidence"] = ""
    sample["notes"] = ""

    columns = [
        "golden_id",
        "case_id",
        "root_tweet_id",
        "customer_id",
        "created_at",
        "turn_count",
        "initial_customer_message",
        "intent",
        "confidence",
        "notes",
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample[columns].to_csv(OUTPUT_PATH, index=False)

    print(f"Eligible cases: {len(eligible):,}")
    print(f"Golden candidates: {len(sample):,}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()