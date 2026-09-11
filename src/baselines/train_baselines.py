from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]

SILVER_PATH = ROOT / "data" / "processed" / "tesco_silver_set.parquet"
GOLDEN_PATH = ROOT / "data" / "golden" / "tesco_golden_250_labeled.csv"
OUTPUT_DIR = ROOT / "data" / "baselines"

RANDOM_STATE = 42


def evaluate(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    print(f"\n{'=' * 80}")
    print(name)
    print(f"{'=' * 80}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print()
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    return predictions


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading silver set...")
    silver = pd.read_parquet(SILVER_PATH)

    print("Loading golden set...")
    golden = pd.read_csv(GOLDEN_PATH)

    # ------------------------------------------------------------------
    # Silver: weakly supervised training data
    # ------------------------------------------------------------------

    silver = silver[
        silver["initial_customer_message"].notna()
        & silver["silver_intent"].notna()
    ].copy()

    X_silver = silver["initial_customer_message"].astype(str)
    y_silver = silver["silver_intent"].astype(str)

    X_train, X_dev, y_train, y_dev = train_test_split(
        X_silver,
        y_silver,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_silver,
    )

    print(f"\nSilver examples : {len(silver)}")
    print(f"Train examples  : {len(X_train)}")
    print(f"Dev examples    : {len(X_dev)}")

    # ------------------------------------------------------------------
    # Golden: held-out human-labelled evaluation set
    # ------------------------------------------------------------------

    golden = golden[
        golden["initial_customer_message"].notna()
        & golden["intent"].notna()
    ].copy()

    X_golden = golden["initial_customer_message"].astype(str)
    y_golden = golden["intent"].astype(str)

    print(f"Golden examples : {len(golden)}")

    # ------------------------------------------------------------------
    # 1. Majority-class baseline
    # ------------------------------------------------------------------

    majority_class = y_train.value_counts().idxmax()

    majority_predictions = [majority_class] * len(y_golden)

    majority_accuracy = accuracy_score(
        y_golden,
        majority_predictions,
    )

    majority_macro_f1 = f1_score(
        y_golden,
        majority_predictions,
        average="macro",
    )

    print(f"\n{'=' * 80}")
    print("MAJORITY CLASS BASELINE")
    print(f"{'=' * 80}")
    print(f"Majority intent : {majority_class}")
    print(f"Accuracy        : {majority_accuracy:.4f}")
    print(f"Macro F1        : {majority_macro_f1:.4f}")

    # ------------------------------------------------------------------
    # 2. TF-IDF + Logistic Regression
    # ------------------------------------------------------------------

    logistic = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("\nTraining Logistic Regression...")
    logistic.fit(X_train, y_train)

    logistic_predictions = evaluate(
        "TF-IDF + LOGISTIC REGRESSION",
        logistic,
        X_golden,
        y_golden,
    )

    # ------------------------------------------------------------------
    # 3. TF-IDF + Linear SVM
    # ------------------------------------------------------------------

    svm = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("\nTraining Linear SVM...")
    svm.fit(X_train, y_train)

    svm_predictions = evaluate(
        "TF-IDF + LINEAR SVM",
        svm,
        X_golden,
        y_golden,
    )

    # ------------------------------------------------------------------
    # Save predictions for later failure analysis
    # ------------------------------------------------------------------

    predictions = golden[
        [
            "golden_id",
            "case_id",
            "initial_customer_message",
            "intent",
        ]
    ].copy()

    predictions = predictions.rename(
        columns={"intent": "gold_intent"}
    )

    predictions["majority_prediction"] = majority_predictions
    predictions["logistic_prediction"] = logistic_predictions
    predictions["svm_prediction"] = svm_predictions

    output_path = OUTPUT_DIR / "baseline_predictions.csv"

    predictions.to_csv(output_path, index=False)

    print(f"\nSaved predictions to: {output_path}")


if __name__ == "__main__":
    main()