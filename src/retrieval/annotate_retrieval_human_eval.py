from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "data" / "retrieval" / "retrieval_human_eval_150.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "retrieval" / "retrieval_human_eval_150.csv"
)


def show_case(group):
    first = group.iloc[0]

    print("\n" + "=" * 100)
    print("GOLDEN CUSTOMER QUERY")
    print("=" * 100)

    print(f"\nGolden ID: {first['golden_id']}")
    print(f"Gold intent: {first['gold_intent']}")
    print(f"\nCUSTOMER:")
    print(first["query"])

    print("\n" + "-" * 100)
    print("HISTORICAL MATCHES")
    print("-" * 100)

    for _, row in group.sort_values("rank").iterrows():
        print(f"\n### MATCH {int(row['rank'])}")
        print(f"Similarity: {row['similarity']:.4f}")
        print(f"Case ID: {row['retrieved_case_id']}")

        print("\nHistorical customer:")
        print(row["retrieved_message"])

        print("\nHistorical resolution guidance:")
        print(row["resolution"])

        print("\nResolution source:")
        print(row["resolution_source"])


def get_score():
    while True:
        value = input(
            "\nScore [3=directly useful, "
            "2=related/useful, "
            "1=same topic but not useful, "
            "0=irrelevant/misleading, "
            "s=skip, q=quit]: "
        ).strip().lower()

        if value in {"0", "1", "2", "3", "s", "q"}:
            return value

        print("Please enter 0, 1, 2, 3, s, or q.")


def main():
    df = pd.read_csv(INPUT_PATH)

    # Ensure annotation columns exist.
    if "human_score" not in df.columns:
        df["human_score"] = ""

    if "human_notes" not in df.columns:
        df["human_notes"] = ""

    # Process one golden query at a time.
    golden_ids = (
        df["golden_id"]
        .drop_duplicates()
        .tolist()
    )

    print("=" * 100)
    print("TESCO HISTORICAL RETRIEVAL — HUMAN EVALUATION")
    print("=" * 100)

    print("\nScoring guide:")
    print("3 = Directly useful precedent; resolution/action is applicable")
    print("2 = Related and somewhat useful context")
    print("1 = Same broad topic, but resolution is not useful")
    print("0 = Irrelevant or potentially misleading")
    print("s = Skip this query")
    print("q = Save and quit")

    for golden_id in golden_ids:
        group = df[df["golden_id"] == golden_id].copy()

        # Skip already completed queries.
        scores = pd.to_numeric(
            group["human_score"],
            errors="coerce"
        )

        if scores.notna().all():
            continue

        show_case(group)

        print("\nJudge the THREE historical matches.")
        print("Important: judge usefulness of the HISTORICAL RESOLUTION,")
        print("not merely whether the customer messages look similar.")

        for index, row in group.sort_values("rank").iterrows():

            print("\n" + "-" * 80)
            print(f"MATCH {int(row['rank'])}")
            print("-" * 80)

            score = get_score()

            if score == "q":
                df.to_csv(OUTPUT_PATH, index=False)
                print("\nSaved progress. Exiting.")
                return

            if score == "s":
                print("Skipping this query.")
                break

            note = input(
                "Optional note (press Enter to skip): "
            ).strip()

            df.loc[index, "human_score"] = int(score)
            df.loc[index, "human_notes"] = note

        # Save after every query so progress is never lost.
        df.to_csv(OUTPUT_PATH, index=False)

        completed = (
            df["human_score"]
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

        print(
            f"\nProgress: {completed}/{len(df)} "
            "retrieval pairs annotated."
        )

    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 100)
    print("ANNOTATION COMPLETE")
    print("=" * 100)

    scores = pd.to_numeric(
        df["human_score"],
        errors="coerce"
    ).dropna()

    print(f"Annotated pairs: {len(scores)}")

    if len(scores):
        print(f"Mean human score: {scores.mean():.3f}")
        print(
            f"Useful (score >= 2): "
            f"{(scores >= 2).mean():.3f}"
        )

    print(f"\nSaved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()