from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = ROOT / "data" / "baselines" / "baseline_predictions.csv"
OUTPUT_DIR = ROOT / "data" / "baselines"


def print_confusion_pairs(df, prediction_col, top_n=15):
    """Print the most common gold → predicted confusions."""

    errors = df[df["gold_intent"] != df[prediction_col]].copy()

    pairs = (
        errors.groupby(["gold_intent", prediction_col])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print(f"\n{'=' * 80}")
    print(f"TOP CONFUSIONS: {prediction_col}")
    print(f"{'=' * 80}")

    if pairs.empty:
        print("No errors.")
        return

    print(pairs.head(top_n).to_string(index=False))


def print_error_counts(df, prediction_col):
    """Print error counts by gold intent."""

    errors = df[df["gold_intent"] != df[prediction_col]].copy()

    counts = (
        errors.groupby("gold_intent")
        .size()
        .reset_index(name="errors")
        .sort_values("errors", ascending=False)
    )

    print(f"\n{'=' * 80}")
    print(f"ERROR COUNTS BY GOLD INTENT: {prediction_col}")
    print(f"{'=' * 80}")

    print(counts.to_string(index=False))


def print_examples(df, prediction_col, max_examples=3):
    """Print representative mistakes for each major confusion."""

    errors = df[df["gold_intent"] != df[prediction_col]].copy()

    pairs = (
        errors.groupby(["gold_intent", prediction_col])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print(f"\n{'=' * 80}")
    print(f"REPRESENTATIVE ERRORS: {prediction_col}")
    print(f"{'=' * 80}")

    for _, row in pairs.head(10).iterrows():
        gold = row["gold_intent"]
        predicted = row[prediction_col]
        count = row["count"]

        examples = errors[
            (errors["gold_intent"] == gold)
            & (errors[prediction_col] == predicted)
        ].head(max_examples)

        print(f"\n{gold} -> {predicted} ({count} errors)")

        for _, example in examples.iterrows():
            message = str(example["initial_customer_message"])
            print(f"  - {message}")


def save_confusion_matrix(df, prediction_col, filename):
    """Save confusion matrix as CSV."""

    labels = sorted(df["gold_intent"].unique())

    matrix = confusion_matrix(
        df["gold_intent"],
        df[prediction_col],
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    output_path = OUTPUT_DIR / filename
    matrix_df.to_csv(output_path)

    print(f"Saved confusion matrix: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading baseline predictions...")

    df = pd.read_csv(PREDICTIONS_PATH)

    print(f"Golden examples: {len(df)}")

    models = [
        "logistic_prediction",
        "svm_prediction",
    ]

    for model in models:
        print_confusion_pairs(df, model)
        print_error_counts(df, model)
        print_examples(df, model)

    save_confusion_matrix(
        df,
        "logistic_prediction",
        "logistic_confusion_matrix.csv",
    )

    save_confusion_matrix(
        df,
        "svm_prediction",
        "svm_confusion_matrix.csv",
    )


if __name__ == "__main__":
    main()