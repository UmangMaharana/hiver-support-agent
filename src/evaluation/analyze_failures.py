from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS = ROOT / "data" / "generation" / "reply_predictions.csv"
REPLY_JUDGMENTS = ROOT / "data" / "evaluation" / "reply_judgments.csv"
ESCALATION_HUMAN = (
    ROOT / "data" / "escalation" / "escalation_human_eval_50.csv"
)
RETRIEVAL_HUMAN = (
    ROOT / "data" / "retrieval" / "retrieval_human_eval_150.csv"
)

OUTPUT_DIR = ROOT / "data" / "evaluation"
OUTPUT = OUTPUT_DIR / "failure_analysis.csv"


def parse_bool(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"true", "1", "yes", "y"}:
        return True

    if value in {"false", "0", "no", "n"}:
        return False

    return None


def require_columns(df, columns, name):
    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} missing columns: {sorted(missing)}"
        )


def load_main_data():
    predictions = pd.read_csv(PREDICTIONS)
    judgments = pd.read_csv(REPLY_JUDGMENTS)

    require_columns(
        predictions,
        {
            "golden_id",
            "gold_intent",
            "predicted_intent",
            "customer_message",
            "reply",
            "escalate",
        },
        "reply_predictions.csv",
    )

    require_columns(
        judgments,
        {
            "golden_id",
            "overall_quality",
            "unsupported_claim",
            "needs_human",
        },
        "reply_judgments.csv",
    )

    # Avoid duplicate columns after merge.
    judgment_columns = [
        "golden_id",
        "intent_alignment",
        "helpfulness",
        "grounding",
        "safety_factuality",
        "escalation_appropriateness",
        "overall_quality",
        "unsupported_claim",
        "needs_human",
        "concise_reason",
    ]

    judgment_columns = [
        c for c in judgment_columns
        if c in judgments.columns
    ]

    judgments = judgments[judgment_columns].copy()

    df = predictions.merge(
        judgments,
        on="golden_id",
        how="left",
    )

    df["escalate"] = df["escalate"].apply(parse_bool)
    df["unsupported_claim"] = (
        df["unsupported_claim"].apply(parse_bool)
    )
    df["needs_human"] = (
        df["needs_human"].apply(parse_bool)
    )

    for column in [
        "intent_alignment",
        "helpfulness",
        "grounding",
        "safety_factuality",
        "escalation_appropriateness",
        "overall_quality",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


def add_section(rows, section, df):
    for _, row in df.iterrows():
        rows.append(
            {
                "section": section,
                "golden_id": row.get("golden_id"),
                "gold_intent": row.get("gold_intent"),
                "predicted_intent": row.get("predicted_intent"),
                "customer_message": row.get("customer_message"),
                "reply": row.get("reply"),
                "escalate": row.get("escalate"),
                "human_escalate": row.get("human_escalate"),
                "policy_escalate": row.get("policy_escalate"),
                "intent_alignment": row.get(
                    "intent_alignment"
                ),
                "helpfulness": row.get("helpfulness"),
                "grounding": row.get("grounding"),
                "safety_factuality": row.get(
                    "safety_factuality"
                ),
                "escalation_appropriateness": row.get(
                    "escalation_appropriateness"
                ),
                "overall_quality": row.get(
                    "overall_quality"
                ),
                "unsupported_claim": row.get(
                    "unsupported_claim"
                ),
                "needs_human": row.get("needs_human"),
                "concise_reason": row.get(
                    "concise_reason"
                ),
                "failure_detail": row.get(
                    "failure_detail"
                ),
            }
        )


def analyze_intent_errors(df, rows):
    errors = df[
        df["gold_intent"] != df["predicted_intent"]
    ].copy()

    errors["failure_detail"] = (
        errors["gold_intent"]
        + " -> "
        + errors["predicted_intent"]
    )

    # Prioritize errors with lower-quality replies.
    errors = errors.sort_values(
        ["overall_quality"],
        ascending=True,
        na_position="last",
    ).head(20)

    add_section(
        rows,
        "intent_error",
        errors,
    )

    # Confusion-pair summary.
    confusion = (
        df[
            df["gold_intent"]
            != df["predicted_intent"]
        ]
        .groupby(
            [
                "gold_intent",
                "predicted_intent",
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
        .head(15)
    )

    for _, row in confusion.iterrows():
        rows.append(
            {
                "section": "intent_confusion_pair",
                "golden_id": "",
                "gold_intent": row["gold_intent"],
                "predicted_intent": row[
                    "predicted_intent"
                ],
                "failure_detail": (
                    f"{row['gold_intent']} -> "
                    f"{row['predicted_intent']} "
                    f"({row['count']} cases)"
                ),
            }
        )


def analyze_reply_quality(df, rows):
    quality = df[
        df["overall_quality"].notna()
    ].copy()

    quality = quality.sort_values(
        "overall_quality",
        ascending=True,
    ).head(20)

    quality["failure_detail"] = (
        "Overall quality = "
        + quality["overall_quality"]
        .round(2)
        .astype(str)
    )

    add_section(
        rows,
        "lowest_quality_reply",
        quality,
    )


def analyze_unsupported_claims(df, rows):
    unsupported = df[
        df["unsupported_claim"] == True
    ].copy()

    unsupported["failure_detail"] = (
        "Judge flagged unsupported claim"
    )

    add_section(
        rows,
        "unsupported_claim",
        unsupported,
    )


def analyze_escalation(df, rows):
    if not ESCALATION_HUMAN.exists():
        print(
            "WARNING: human escalation benchmark not found."
        )
        return

    escalation = pd.read_csv(
        ESCALATION_HUMAN
    )

    require_columns(
        escalation,
        {
            "golden_id",
            "human_escalate",
            "escalate",
            "policy_escalate",
        },
        "escalation_human_eval_50.csv",
    )

    escalation["human_escalate"] = (
        escalation["human_escalate"]
        .apply(parse_bool)
    )

    escalation["escalate"] = (
        escalation["escalate"]
        .apply(parse_bool)
    )

    escalation["policy_escalate"] = (
        escalation["policy_escalate"]
        .apply(parse_bool)
    )

    # Gemini false positives:
    # Human says NO, Gemini says YES.
    fp = escalation[
        (escalation["human_escalate"] == False)
        & (escalation["escalate"] == True)
    ].copy()

    fp["failure_detail"] = (
        "Gemini escalated but human said NO"
    )

    fp = fp.merge(
        df[
            [
                "golden_id",
                "gold_intent",
                "predicted_intent",
                "customer_message",
                "reply",
                "overall_quality",
            ]
        ],
        on="golden_id",
        how="left",
    )

    add_section(
        rows,
        "escalation_false_positive",
        fp,
    )

    # Gemini false negatives:
    # Human says YES, Gemini says NO.
    fn = escalation[
        (escalation["human_escalate"] == True)
        & (escalation["escalate"] == False)
    ].copy()

    fn["failure_detail"] = (
        "Gemini did not escalate but human said YES"
    )

    fn = fn.merge(
        df[
            [
                "golden_id",
                "gold_intent",
                "predicted_intent",
                "customer_message",
                "reply",
                "overall_quality",
            ]
        ],
        on="golden_id",
        how="left",
    )

    add_section(
        rows,
        "escalation_false_negative",
        fn,
    )

    # Rule baseline failures.
    rule_fp = escalation[
        (escalation["human_escalate"] == False)
        & (escalation["policy_escalate"] == True)
    ].copy()

    rule_fp["failure_detail"] = (
        "Rule escalated but human said NO"
    )

    rule_fp = rule_fp.merge(
        df[
            [
                "golden_id",
                "gold_intent",
                "predicted_intent",
                "customer_message",
                "reply",
                "overall_quality",
            ]
        ],
        on="golden_id",
        how="left",
    )

    add_section(
        rows,
        "rule_false_positive",
        rule_fp,
    )

    rule_fn = escalation[
        (escalation["human_escalate"] == True)
        & (escalation["policy_escalate"] == False)
    ].copy()

    rule_fn["failure_detail"] = (
        "Rule did not escalate but human said YES"
    )

    rule_fn = rule_fn.merge(
        df[
            [
                "golden_id",
                "gold_intent",
                "predicted_intent",
                "customer_message",
                "reply",
                "overall_quality",
            ]
        ],
        on="golden_id",
        how="left",
    )

    add_section(
        rows,
        "rule_false_negative",
        rule_fn,
    )


def analyze_wrong_intent_good_reply(df, rows):
    candidates = df[
        (
            df["gold_intent"]
            != df["predicted_intent"]
        )
        & (
            df["overall_quality"] >= 4
        )
    ].copy()

    candidates["failure_detail"] = (
        "Wrong intent but reply quality >= 4/5"
    )

    candidates = candidates.sort_values(
        "overall_quality",
        ascending=False,
    ).head(15)

    add_section(
        rows,
        "wrong_intent_good_reply",
        candidates,
    )


def analyze_correct_intent_bad_reply(df, rows):
    candidates = df[
        (
            df["gold_intent"]
            == df["predicted_intent"]
        )
        & (
            df["overall_quality"] <= 3
        )
    ].copy()

    candidates["failure_detail"] = (
        "Correct intent but reply quality <= 3/5"
    )

    candidates = candidates.sort_values(
        "overall_quality",
        ascending=True,
    ).head(15)

    add_section(
        rows,
        "correct_intent_bad_reply",
        candidates,
    )


def analyze_low_grounding(df, rows):
    candidates = df[
        df["grounding"].notna()
    ].copy()

    candidates = candidates.sort_values(
        "grounding",
        ascending=True,
    ).head(15)

    candidates["failure_detail"] = (
        "Low grounding score"
    )

    add_section(
        rows,
        "low_grounding",
        candidates,
    )


def analyze_low_safety(df, rows):
    candidates = df[
        df["safety_factuality"].notna()
    ].copy()

    candidates = candidates.sort_values(
        "safety_factuality",
        ascending=True,
    ).head(15)

    candidates["failure_detail"] = (
        "Low safety/factuality score"
    )

    add_section(
        rows,
        "low_safety",
        candidates,
    )


def analyze_retrieval(rows):
    if not RETRIEVAL_HUMAN.exists():
        print(
            "WARNING: human retrieval benchmark not found."
        )
        return

    retrieval = pd.read_csv(
        RETRIEVAL_HUMAN
    )

    # Identify the human score column.
    score_column = None

    for candidate in [
        "human_score",
        "score",
        "usefulness",
        "human_relevance",
    ]:
        if candidate in retrieval.columns:
            score_column = candidate
            break

    if score_column is None:
        print(
            "WARNING: retrieval score column not found."
        )
        return

    retrieval[score_column] = pd.to_numeric(
        retrieval[score_column],
        errors="coerce",
    )

    # Lowest-scoring retrieved pairs.
    worst = retrieval[
        retrieval[score_column].notna()
    ].sort_values(
        score_column,
        ascending=True,
    ).head(20)

    for _, row in worst.iterrows():
        # Try to recover useful identifiers regardless
        # of the exact benchmark schema.
        golden_id = row.get(
            "golden_id",
            row.get("query_id", ""),
        )

        rows.append(
            {
                "section": "retrieval_failure",
                "golden_id": golden_id,
                "gold_intent": row.get(
                    "gold_intent",
                    "",
                ),
                "customer_message": row.get(
                    "customer_message",
                    row.get("query", ""),
                ),
                "failure_detail": (
                    f"Human retrieval score = "
                    f"{row[score_column]}"
                ),
                "retrieval_score": row[
                    score_column
                ],
            }
        )


def print_summary(df, rows):
    print("\n" + "=" * 72)
    print("FAILURE ANALYSIS SUMMARY")
    print("=" * 72)

    sections = pd.Series(
        [
            row.get("section")
            for row in rows
        ]
    )

    print(
        sections.value_counts().to_string()
    )

    intent_errors = (
        df[
            df["gold_intent"]
            != df["predicted_intent"]
        ]
    )

    print(
        f"\nIntent errors: "
        f"{len(intent_errors)}/{len(df)} "
        f"({len(intent_errors) / len(df):.1%})"
    )

    unsupported = df[
        df["unsupported_claim"] == True
    ]

    print(
        f"Unsupported claims: "
        f"{len(unsupported)}/{len(df)} "
        f"({len(unsupported) / len(df):.1%})"
    )

    print("\nTop intent confusion pairs:")

    confusion = (
        intent_errors
        .groupby(
            [
                "gold_intent",
                "predicted_intent",
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    if confusion.empty:
        print("None.")
    else:
        print(confusion.to_string())


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("TESCO SUPPORT AGENT — FAILURE ANALYSIS")
    print("=" * 72)

    df = load_main_data()

    print(
        f"Loaded {len(df)} generated examples."
    )

    rows = []

    analyze_intent_errors(
        df,
        rows,
    )

    analyze_reply_quality(
        df,
        rows,
    )

    analyze_unsupported_claims(
        df,
        rows,
    )

    analyze_escalation(
        df,
        rows,
    )

    analyze_wrong_intent_good_reply(
        df,
        rows,
    )

    analyze_correct_intent_bad_reply(
        df,
        rows,
    )

    analyze_low_grounding(
        df,
        rows,
    )

    analyze_low_safety(
        df,
        rows,
    )

    analyze_retrieval(
        rows,
    )

    output_df = pd.DataFrame(rows)

    # Put the most useful columns first.
    preferred_columns = [
        "section",
        "golden_id",
        "gold_intent",
        "predicted_intent",
        "customer_message",
        "reply",
        "escalate",
        "human_escalate",
        "policy_escalate",
        "intent_alignment",
        "helpfulness",
        "grounding",
        "safety_factuality",
        "escalation_appropriateness",
        "overall_quality",
        "unsupported_claim",
        "needs_human",
        "failure_detail",
        "concise_reason",
        "retrieval_score",
    ]

    existing_columns = [
        column
        for column in preferred_columns
        if column in output_df.columns
    ]

    remaining_columns = [
        column
        for column in output_df.columns
        if column not in existing_columns
    ]

    output_df = output_df[
        existing_columns + remaining_columns
    ]

    output_df.to_csv(
        OUTPUT,
        index=False,
    )

    print_summary(
        df,
        rows,
    )

    print(
        f"\nSaved failure analysis:"
        f"\n{OUTPUT}"
    )


if __name__ == "__main__":
    main()