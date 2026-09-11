from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/golden/taxonomy_review_100.csv"
)

OUTPUT_PATH = Path(
    "data/golden/taxonomy_review_100_labeled.csv"
)

BATCH_SIZE = 10


LABELS = [
    "delivery_issue",
    "delivery_slots",
    "online_website_issue",
    "online_order_issue",
    "click_collect",
    "product_availability",
    "product_quality_safety",
    "pricing_offers",
    "refund_return_compensation",
    "store_service_complaint",
    "product_information",
    "clubcard_account",
    "other",
]


def save(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )


def show_labels() -> None:
    print("\nLABELS")
    print("-" * 90)

    for i, label in enumerate(LABELS, 1):
        print(f"{i:2} = {label}")

    print("-" * 90)


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    for column in ["intent", "confidence", "notes"]:
        df[column] = df[column].fillna("")

    print("=" * 90)
    print("TESCO TAXONOMY BATCH REVIEW")
    print("=" * 90)

    show_labels()

    # Only show examples that haven't already been labelled.
    pending = df[
        df["intent"].astype(str).str.strip() == ""
    ]

    total_pending = len(pending)

    print(
        f"\nRemaining examples: {total_pending}"
    )

    while True:

        pending = df[
            df["intent"].astype(str).str.strip() == ""
        ]

        if pending.empty:
            break

        batch = pending.head(BATCH_SIZE)

        print("\n" + "=" * 90)
        print(
            f"BATCH — {len(batch)} examples"
        )
        print("=" * 90)

        for number, (_, row) in enumerate(
            batch.iterrows(),
            start=1,
        ):
            print("\n" + "-" * 90)
            print(
                f"[{number}] "
                f"{row['review_id']} | "
                f"Cluster {row['cluster']}"
            )
            print("-" * 90)
            print(row["initial_customer_message"])

        print("\n")
        print(
            "Enter labels in order, separated by spaces."
        )
        print(
            "Example: 5 1 13 7 8 10 6 3 9 13"
        )
        print(
            "Type q to quit."
        )

        while True:
            answer = input(
                "\nLabels: "
            ).strip()

            if answer.lower() == "q":
                save(df)

                print(
                    f"\nProgress saved to:"
                    f"\n{OUTPUT_PATH}"
                )

                return

            parts = answer.split()

            if len(parts) != len(batch):
                print(
                    f"Need exactly "
                    f"{len(batch)} labels."
                )
                continue

            valid = True
            labels = []

            for part in parts:
                try:
                    number = int(part)
                except ValueError:
                    valid = False
                    break

                if not 1 <= number <= len(LABELS):
                    valid = False
                    break

                labels.append(
                    LABELS[number - 1]
                )

            if not valid:
                print(
                    "Invalid label. "
                    "Use numbers 1-13."
                )
                continue

            break

        # Save labels.
        for (_, row), label in zip(
            batch.iterrows(),
            labels,
        ):
            df.loc[
                df["review_id"] == row["review_id"],
                "intent",
            ] = label

        save(df)

        print(
            f"\nSaved {len(batch)} labels."
        )

    print("\n" + "=" * 90)
    print("REVIEW COMPLETE")
    print("=" * 90)

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )

    print("\nLabel counts:")

    print(
        df["intent"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()