from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/golden/golden_candidates_250.csv")
OUTPUT_PATH = Path("data/golden/tesco_golden_250_labeled.csv")

BATCH_SIZE = 10

LABELS = {
    1: "delivery_issue",
    2: "delivery_slots",
    3: "online_website_issue",
    4: "online_order_issue",
    5: "click_collect",
    6: "product_availability",
    7: "product_quality_safety",
    8: "pricing_offers",
    9: "refund_return_compensation",
    10: "store_service_complaint",
    11: "product_information",
    12: "clubcard_account",
    13: "other",
}


def show_labels() -> None:
    print("\nINTENTS")
    print("-" * 60)

    for number, intent in LABELS.items():
        print(f"{number:2d}. {intent}")

    print("-" * 60)


def get_label_input() -> list[int]:
    while True:
        raw = input(
            f"\nEnter {BATCH_SIZE} labels separated by spaces: "
        ).strip()

        try:
            labels = [int(x) for x in raw.split()]
        except ValueError:
            print("❌ Labels must be numbers 1-13.")
            continue

        if len(labels) != BATCH_SIZE:
            print(f"❌ Expected exactly {BATCH_SIZE} labels.")
            continue

        if any(label not in LABELS for label in labels):
            print("❌ Every label must be between 1 and 13.")
            continue

        return labels


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    # Create output file on first run.
    if OUTPUT_PATH.exists():
        labeled = pd.read_csv(OUTPUT_PATH)

        # Preserve any existing annotations.
        if len(labeled) != len(df):
            raise ValueError(
                "Existing labeled file does not match candidate file."
            )

        df = labeled

    else:
        df["intent"] = ""
        df["confidence"] = ""
        df["notes"] = ""

    show_labels()

    total = len(df)

    while True:
        unlabeled = df[
            df["intent"].isna() | df["intent"].astype(str).str.strip().eq("")
        ]

        if unlabeled.empty:
            print("\n" + "=" * 70)
            print("GOLDEN ANNOTATION COMPLETE")
            print("=" * 70)
            print(f"Saved to: {OUTPUT_PATH}")
            print("\nLabel counts:")
            print(df["intent"].value_counts().to_string())
            break

        batch_indices = unlabeled.index[:BATCH_SIZE]

        print("\n" + "=" * 90)
        print(
            f"Examples {batch_indices[0] + 1}-{batch_indices[-1] + 1}"
            f" / {total}"
        )
        print("=" * 90)

        for display_number, idx in enumerate(batch_indices, start=1):
            row = df.loc[idx]

            print(f"\n[{display_number}] {row['golden_id']}")
            print("-" * 90)
            print(row["initial_customer_message"])
            print()

        labels = get_label_input()

        for idx, label in zip(batch_indices, labels):
            df.at[idx, "intent"] = LABELS[label]

        # Confidence is collected separately so the label input stays fast.
        print("\nConfidence for these 10 examples:")
        print("h = high, m = medium, l = low")

        while True:
            confidence = input(
                "Enter 10 confidence values separated by spaces: "
            ).strip().lower().split()

            if len(confidence) == BATCH_SIZE and all(
                value in {"h", "m", "l"} for value in confidence
            ):
                break

            print("❌ Use exactly 10 values, each h, m, or l.")

        confidence_map = {
            "h": "high",
            "m": "medium",
            "l": "low",
        }

        for idx, value in zip(batch_indices, confidence):
            df.at[idx, "confidence"] = confidence_map[value]

        # Notes are optional. Leave blank by pressing Enter.
        for display_number, idx in enumerate(batch_indices, start=1):
            note = input(
                f"Note for [{display_number}] "
                f"(Enter to leave blank): "
            ).strip()

            df.at[idx, "notes"] = note

        df.to_csv(OUTPUT_PATH, index=False)

        remaining = len(unlabeled) - len(batch_indices)

        print(
            f"\n✅ Saved batch. {remaining} examples remaining."
        )


if __name__ == "__main__":
    main()