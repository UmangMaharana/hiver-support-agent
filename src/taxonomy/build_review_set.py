from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/tesco_intent_clusters.parquet"
)

OUTPUT_PATH = Path(
    "data/golden/taxonomy_review_100.csv"
)


SAMPLES_PER_CLUSTER = 5


def main() -> None:
    print("=" * 80)
    print("BUILDING TESCO TAXONOMY REVIEW SET")
    print("=" * 80)

    df = pd.read_parquet(INPUT_PATH)

    print(f"\nAvailable cases: {len(df):,}")

    # ---------------------------------------------------------------
    # Sample across every exploratory cluster.
    # This is deliberately NOT a random sample of the whole dataset.
    # We want difficult/diverse examples for taxonomy validation.
    # ---------------------------------------------------------------

    parts = []

    for cluster_id, group in df.groupby("cluster"):
        n = min(SAMPLES_PER_CLUSTER, len(group))

        sample = group.sample(
            n=n,
            random_state=42 + int(cluster_id),
        )

        parts.append(sample)

    review = pd.concat(
        parts,
        ignore_index=True,
    )

    # Shuffle so cluster membership isn't obvious while labelling.
    review = review.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Create annotation columns.
    # ---------------------------------------------------------------

    review.insert(
        0,
        "review_id",
        [
            f"REV_{i:03d}"
            for i in range(1, len(review) + 1)
        ],
    )

    review["intent"] = ""
    review["confidence"] = ""
    review["notes"] = ""

    # Keep the annotation file compact.
    output = review[
        [
            "review_id",
            "case_id",
            "cluster",
            "initial_customer_message",
            "intent",
            "confidence",
            "notes",
        ]
    ].copy()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nReview examples: {len(output):,}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print("\nCluster distribution:")

    print(
        output["cluster"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nCandidate labels:")
    print("  delivery_issue")
    print("  delivery_slots")
    print("  online_website_issue")
    print("  online_order_issue")
    print("  click_collect")
    print("  product_availability")
    print("  product_quality_safety")
    print("  pricing_offers")
    print("  refund_return_compensation")
    print("  store_service_complaint")
    print("  product_information")
    print("  clubcard_account")
    print("  other")


if __name__ == "__main__":
    main()