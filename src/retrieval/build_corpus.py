from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CASES_PATH = ROOT / "data" / "processed" / "tesco_cases.parquet"
GOLDEN_PATH = ROOT / "data" / "golden" / "tesco_golden_250_labeled.csv"
REVIEW_PATH = ROOT / "data" / "golden" / "taxonomy_review_100.csv"

OUTPUT_DIR = ROOT / "data" / "retrieval"
OUTPUT_PATH = OUTPUT_DIR / "tesco_retrieval_corpus.parquet"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading reconstructed Tesco cases...")
    cases = pd.read_parquet(CASES_PATH)

    print("Loading golden evaluation cases...")
    golden = pd.read_csv(GOLDEN_PATH)

    print("Loading taxonomy-review cases...")
    review = pd.read_csv(REVIEW_PATH)

    golden_case_ids = set(golden["case_id"].astype(str))
    review_case_ids = set(review["case_id"].astype(str))

    cases["case_id"] = cases["case_id"].astype(str)

    print(f"Total cases: {len(cases)}")
    print(f"Golden cases to exclude: {len(golden_case_ids)}")
    print(f"Review cases to exclude: {len(review_case_ids)}")

    # Exclude all evaluation / taxonomy-review cases.
    corpus = cases[
        ~cases["case_id"].isin(golden_case_ids)
        & ~cases["case_id"].isin(review_case_ids)
    ].copy()

    # Keep only the fields needed for retrieval.
    corpus = corpus[
        [
            "case_id",
            "initial_customer_message",
            "transcript",
        ]
    ].copy()

    # Remove cases with missing retrieval text.
    corpus = corpus[
        corpus["initial_customer_message"].notna()
        & corpus["transcript"].notna()
    ].copy()

    corpus["initial_customer_message"] = (
        corpus["initial_customer_message"]
        .astype(str)
        .str.strip()
    )

    corpus["transcript"] = (
        corpus["transcript"]
        .astype(str)
        .str.strip()
    )

    corpus = corpus[
        (corpus["initial_customer_message"] != "")
        & (corpus["transcript"] != "")
    ].copy()

    # Defensive exact deduplication.
    corpus = corpus.drop_duplicates(
        subset=["initial_customer_message", "transcript"]
    ).reset_index(drop=True)

    corpus.to_parquet(OUTPUT_PATH, index=False)

    print("\nRetrieval corpus created.")
    print(f"Cases: {len(corpus)}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nExample:")
    example = corpus.iloc[0]

    print(f"case_id: {example['case_id']}")
    print(f"customer: {example['initial_customer_message']}")
    print("transcript:")
    print(example["transcript"])


if __name__ == "__main__":
    main()