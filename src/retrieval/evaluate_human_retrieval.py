from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/retrieval/retrieval_human_eval_150.csv")
OUTPUT_PATH = Path("data/retrieval/human_retrieval_metrics.csv")


def main():
    df = pd.read_csv(INPUT_PATH)

    df["human_score"] = pd.to_numeric(df["human_score"], errors="coerce")

    if df["human_score"].isna().any():
        raise ValueError("Some retrieval pairs are missing human scores.")

    # ------------------------------------------------------------------
    # Query-level metrics
    # ------------------------------------------------------------------
    query_metrics = (
        df.groupby(["golden_id", "gold_intent"], as_index=False)
        .agg(
            mean_score=("human_score", "mean"),
            top1_score=("human_score", lambda x: x.iloc[0]),
            top3_direct=("human_score", lambda x: int((x == 3).any())),
            top3_useful=("human_score", lambda x: int((x >= 2).any())),
            top1_direct=("human_score", lambda x: int(x.iloc[0] == 3)),
            top1_useful=("human_score", lambda x: int(x.iloc[0] >= 2)),
        )
    )

    # ------------------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------------------
    overall = {
        "queries": len(query_metrics),
        "retrieval_pairs": len(df),
        "mean_score_all_matches": df["human_score"].mean(),
        "mean_top1_score": query_metrics["top1_score"].mean(),
        "mean_top3_score": query_metrics["mean_score"].mean(),
        "direct_at_1": query_metrics["top1_direct"].mean(),
        "useful_at_1": query_metrics["top1_useful"].mean(),
        "direct_at_3": query_metrics["top3_direct"].mean(),
        "useful_at_3": query_metrics["top3_useful"].mean(),
        "score_0_pct": (df["human_score"] == 0).mean(),
        "score_1_pct": (df["human_score"] == 1).mean(),
        "score_2_pct": (df["human_score"] == 2).mean(),
        "score_3_pct": (df["human_score"] == 3).mean(),
    }

    # ------------------------------------------------------------------
    # Per-intent metrics
    # ------------------------------------------------------------------
    per_intent = (
        query_metrics.groupby("gold_intent", as_index=False)
        .agg(
            queries=("golden_id", "count"),
            mean_top1_score=("top1_score", "mean"),
            mean_top3_score=("mean_score", "mean"),
            direct_at_1=("top1_direct", "mean"),
            useful_at_1=("top1_useful", "mean"),
            direct_at_3=("top3_direct", "mean"),
            useful_at_3=("top3_useful", "mean"),
        )
        .sort_values("useful_at_3", ascending=False)
    )

    # Save per-intent metrics.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    per_intent.to_csv(OUTPUT_PATH, index=False)

    # ------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------
    print("=" * 80)
    print("HUMAN RETRIEVAL EVALUATION")
    print("=" * 80)

    print(f"\nQueries:          {overall['queries']}")
    print(f"Retrieval pairs:  {overall['retrieval_pairs']}")

    print("\nOVERALL METRICS")
    print("-" * 80)

    print(f"Mean score (all matches): {overall['mean_score_all_matches']:.3f}")
    print(f"Mean top-1 score:         {overall['mean_top1_score']:.3f}")
    print(f"Mean top-3 score:         {overall['mean_top3_score']:.3f}")

    print(f"\nDirect@1:                 {overall['direct_at_1']:.1%}")
    print(f"Useful@1 (score >= 2):   {overall['useful_at_1']:.1%}")
    print(f"Direct@3:                 {overall['direct_at_3']:.1%}")
    print(f"Useful@3 (score >= 2):   {overall['useful_at_3']:.1%}")

    print("\nSCORE DISTRIBUTION")
    print("-" * 80)

    print(f"0 — Irrelevant/misleading: {overall['score_0_pct']:.1%}")
    print(f"1 — Broadly related:       {overall['score_1_pct']:.1%}")
    print(f"2 — Somewhat useful:       {overall['score_2_pct']:.1%}")
    print(f"3 — Directly useful:       {overall['score_3_pct']:.1%}")

    print("\nPER-INTENT RESULTS")
    print("-" * 80)

    display = per_intent.copy()

    for col in [
        "direct_at_1",
        "useful_at_1",
        "direct_at_3",
        "useful_at_3",
    ]:
        display[col] = display[col].map(lambda x: f"{x:.1%}")

    print(display.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"Saved: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()