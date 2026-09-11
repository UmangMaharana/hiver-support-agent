from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RETRIEVAL_EVAL_PATH = (
    ROOT / "data" / "retrieval" / "tfidf_retrieval_eval.csv"
)

RESOLUTION_PATH = (
    ROOT / "data" / "retrieval" / "tesco_resolution_corpus.parquet"
)

OUTPUT_DIR = ROOT / "data" / "retrieval"
OUTPUT_PATH = OUTPUT_DIR / "retrieval_human_eval_150.csv"


def main():
    print("Loading retrieval results...")
    results = pd.read_csv(RETRIEVAL_EVAL_PATH)

    print("Loading historical resolutions...")
    resolutions = pd.read_parquet(RESOLUTION_PATH)

    resolution_lookup = (
        resolutions[
            ["case_id", "resolution", "resolution_source"]
        ]
        .drop_duplicates("case_id")
    )

    results = results.merge(
        resolution_lookup,
        left_on="retrieved_case_id",
        right_on="case_id",
        how="left",
    )

    # Pick 50 golden queries.
    # Stratify by golden intent so common intents don't dominate.
    sampled_queries = (
        results[["golden_id", "gold_intent"]]
        .drop_duplicates()
        .groupby("gold_intent", group_keys=False)
        .apply(
            lambda x: x.sample(
                n=min(4, len(x)),
                random_state=42
            )
        )
        .reset_index(drop=True)
    )

    # If the stratified sample is below 50, fill the remainder randomly.
    target_queries = 50

    if len(sampled_queries) < target_queries:
        remaining = (
            results[["golden_id", "gold_intent"]]
            .drop_duplicates()
            .loc[
                lambda x: ~x["golden_id"].isin(
                    sampled_queries["golden_id"]
                )
            ]
        )

        extra = remaining.sample(
            n=min(
                target_queries - len(sampled_queries),
                len(remaining)
            ),
            random_state=42,
        )

        sampled_queries = pd.concat(
            [sampled_queries, extra],
            ignore_index=True,
        )

    sampled_ids = set(sampled_queries["golden_id"])

    benchmark = results[
        results["golden_id"].isin(sampled_ids)
        & results["rank"].isin([1, 2, 3])
    ].copy()

    benchmark = benchmark.merge(
        sampled_queries,
        on=["golden_id", "gold_intent"],
        how="inner",
    )

    benchmark = benchmark[
        [
            "golden_id",
            "gold_intent",
            "query",
            "rank",
            "retrieved_case_id",
            "similarity",
            "retrieved_message",
            "resolution",
            "resolution_source",
        ]
    ].copy()

    benchmark["human_score"] = ""
    benchmark["human_notes"] = ""

    benchmark = benchmark.sort_values(
        ["golden_id", "rank"]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 80)
    print("HUMAN RETRIEVAL EVALUATION SET")
    print("=" * 80)

    print(f"Golden queries sampled: {benchmark['golden_id'].nunique()}")
    print(f"Retrieval pairs:        {len(benchmark)}")

    print("\nQueries per intent:")
    print(
        benchmark[
            ["golden_id", "gold_intent"]
        ]
        .drop_duplicates()["gold_intent"]
        .value_counts()
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()