from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[2]

GOLDEN = ROOT / "data" / "golden" / "tesco_golden_250_labeled.csv"
PREDICTIONS = ROOT / "data" / "generation" / "reply_predictions.csv"

RETRIEVAL_HUMAN = (
    ROOT / "data" / "retrieval" / "retrieval_human_eval_150.csv"
)

ESCALATION_HUMAN = (
    ROOT / "data" / "escalation" / "escalation_human_eval_50.csv"
)

REPLY_JUDGMENTS = (
    ROOT / "data" / "evaluation" / "reply_judgments.csv"
)

OUTPUT_DIR = ROOT / "data" / "evaluation"
OUTPUT_JSON = OUTPUT_DIR / "full_eval_summary.json"
OUTPUT_CSV = OUTPUT_DIR / "full_eval_summary.csv"


def parse_bool(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"true", "1", "yes", "y"}:
        return True

    if value in {"false", "0", "no", "n"}:
        return False

    return None


def evaluate_intent():
    df = pd.read_csv(PREDICTIONS)

    required = {
        "gold_intent",
        "predicted_intent",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Generation predictions missing columns: {missing}"
        )

    y_true = df["gold_intent"]
    y_pred = df["predicted_intent"]

    return {
        "n": int(len(df)),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }


def evaluate_retrieval():
    if not RETRIEVAL_HUMAN.exists():
        return {
            "available": False,
            "reason": "Human retrieval benchmark not found",
        }

    df = pd.read_csv(RETRIEVAL_HUMAN)

    # The human retrieval annotation file contains one row per
    # query/retrieved-case pair. Scores use the 0-3 rubric:
    # 0 = irrelevant
    # 1 = weak/contextual
    # 2 = useful
    # 3 = directly applicable.
    score_column = None

    for candidate in [
        "human_score",
        "score",
        "usefulness",
        "human_relevance",
    ]:
        if candidate in df.columns:
            score_column = candidate
            break

    if score_column is None:
        return {
            "available": False,
            "reason": "Could not identify human retrieval score column",
        }

    scores = pd.to_numeric(
        df[score_column],
        errors="coerce",
    ).dropna()

    if scores.empty:
        return {
            "available": False,
            "reason": "No human retrieval scores found",
        }

    return {
        "available": True,
        "n_pairs": int(len(scores)),
        "mean_score": float(scores.mean()),
        "direct_rate": float(
            (scores == 3).mean()
        ),
        "useful_or_better_rate": float(
            (scores >= 2).mean()
        ),
        "irrelevant_rate": float(
            (scores == 0).mean()
        ),
    }


def evaluate_escalation():
    if not ESCALATION_HUMAN.exists():
        return {
            "available": False,
            "reason": "Human escalation benchmark not found",
        }

    df = pd.read_csv(ESCALATION_HUMAN)

    required = {
        "human_escalate",
        "escalate",
        "policy_escalate",
    }

    missing = required - set(df.columns)

    if missing:
        return {
            "available": False,
            "reason": f"Missing columns: {sorted(missing)}",
        }

    for col in [
        "human_escalate",
        "escalate",
        "policy_escalate",
    ]:
        df[col] = df[col].apply(parse_bool)

    # Only evaluate rows with valid human labels.
    df = df[df["human_escalate"].notna()].copy()

    if df.empty:
        return {
            "available": False,
            "reason": "No valid human escalation labels found",
        }

    y_true = df["human_escalate"].astype(bool)

    results = {
        "available": True,
        "n": int(len(df)),
        "human_escalation_rate": float(y_true.mean()),
    }

    for name, column in [
        ("gemini", "escalate"),
        ("rule_baseline", "policy_escalate"),
    ]:
        y_pred = df[column].astype(bool)

        results[name] = {
            "accuracy": float(
                accuracy_score(y_true, y_pred)
            ),
            "precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "escalation_rate": float(y_pred.mean()),
            "agreement": float(
                (y_true == y_pred).mean()
            ),
        }

    return results


def evaluate_generation_coverage():
    df = pd.read_csv(PREDICTIONS)

    required = {
        "golden_id",
        "customer_message",
        "reply",
        "escalate",
        "escalation_reason",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Generation predictions missing columns: {missing}"
        )

    total = len(df)

    if total == 0:
        raise ValueError(
            "Generation prediction file is empty."
        )

    nonempty_replies = (
        df["reply"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    valid_escalation_values = (
        df["escalate"]
        .apply(parse_bool)
        .notna()
        .sum()
    )

    return {
        "n": int(total),
        "reply_coverage": float(
            nonempty_replies / total
        ),
        "valid_escalation_output_rate": float(
            valid_escalation_values / total
        ),
    }


def evaluate_reply_quality():
    if not REPLY_JUDGMENTS.exists():
        return {
            "available": False,
            "reason": "Reply judgment file not found",
        }

    df = pd.read_csv(REPLY_JUDGMENTS)

    required = {
        "golden_id",
        "intent_alignment",
        "helpfulness",
        "grounding",
        "safety_factuality",
        "escalation_appropriateness",
        "overall_quality",
        "unsupported_claim",
        "needs_human",
    }

    missing = required - set(df.columns)

    if missing:
        return {
            "available": False,
            "reason": f"Missing columns: {sorted(missing)}",
        }

    score_fields = [
        "intent_alignment",
        "helpfulness",
        "grounding",
        "safety_factuality",
        "escalation_appropriateness",
        "overall_quality",
    ]

    for column in score_fields:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Keep only rows with a valid overall judgment.
    df = df[df["overall_quality"].notna()].copy()

    if df.empty:
        return {
            "available": False,
            "reason": "No valid reply judgments found",
        }

    df["unsupported_claim"] = (
        df["unsupported_claim"].apply(parse_bool)
    )

    df["needs_human"] = (
        df["needs_human"].apply(parse_bool)
    )

    results = {
        "available": True,
        "n": int(len(df)),
    }

    for column in score_fields:
        results[column] = float(
            df[column].mean()
        )

    unsupported = df[
        "unsupported_claim"
    ].dropna()

    if len(unsupported) > 0:
        results["unsupported_claim_rate"] = float(
            unsupported.astype(bool).mean()
        )
    else:
        results["unsupported_claim_rate"] = None

    needs_human = df[
        "needs_human"
    ].dropna()

    if len(needs_human) > 0:
        results["judge_human_needed_rate"] = float(
            needs_human.astype(bool).mean()
        )
    else:
        results["judge_human_needed_rate"] = None

    return results


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("TESCO SUPPORT AGENT — FULL EVALUATION")
    print("=" * 72)

    # ------------------------------------------------------------
    # 1. Intent classification
    # ------------------------------------------------------------

    print("\n[1/5] Intent classification")

    intent = evaluate_intent()

    print(
        f"Examples:   {intent['n']}"
    )
    print(
        f"Accuracy:   {intent['accuracy']:.3f}"
    )
    print(
        f"Macro-F1:   {intent['macro_f1']:.3f}"
    )

    # ------------------------------------------------------------
    # 2. Retrieval
    # ------------------------------------------------------------

    print("\n[2/5] Retrieval")

    retrieval = evaluate_retrieval()

    if retrieval["available"]:
        print(
            f"Pairs:      {retrieval['n_pairs']}"
        )
        print(
            f"Mean score: {retrieval['mean_score']:.3f}"
        )
        print(
            f"Direct:     {retrieval['direct_rate']:.1%}"
        )
        print(
            f"Useful+:    "
            f"{retrieval['useful_or_better_rate']:.1%}"
        )
        print(
            f"Irrelevant: "
            f"{retrieval['irrelevant_rate']:.1%}"
        )
    else:
        print(
            f"Unavailable: {retrieval['reason']}"
        )

    # ------------------------------------------------------------
    # 3. Escalation
    # ------------------------------------------------------------

    print("\n[3/5] Escalation")

    escalation = evaluate_escalation()

    if escalation["available"]:
        print(
            f"Human rate: "
            f"{escalation['human_escalation_rate']:.1%}"
        )

        for model in [
            "gemini",
            "rule_baseline",
        ]:
            metrics = escalation[model]

            print(
                f"{model}: "
                f"accuracy={metrics['accuracy']:.3f}, "
                f"precision={metrics['precision']:.3f}, "
                f"recall={metrics['recall']:.3f}, "
                f"F1={metrics['f1']:.3f}, "
                f"agreement={metrics['agreement']:.3f}"
            )
    else:
        print(
            f"Unavailable: {escalation['reason']}"
        )

    # ------------------------------------------------------------
    # 4. Generation coverage
    # ------------------------------------------------------------

    print("\n[4/5] Generation coverage")

    generation = evaluate_generation_coverage()

    print(
        f"Examples:   {generation['n']}"
    )
    print(
        f"Reply coverage: "
        f"{generation['reply_coverage']:.1%}"
    )
    print(
        f"Valid escalation JSON: "
        f"{generation['valid_escalation_output_rate']:.1%}"
    )

    # ------------------------------------------------------------
    # 5. Reply quality
    # ------------------------------------------------------------

    print("\n[5/5] Reply quality — LLM judge")

    reply_quality = evaluate_reply_quality()

    if reply_quality["available"]:
        print(
            f"Examples:              "
            f"{reply_quality['n']}"
        )
        print(
            f"Intent alignment:       "
            f"{reply_quality['intent_alignment']:.3f}/5"
        )
        print(
            f"Helpfulness:            "
            f"{reply_quality['helpfulness']:.3f}/5"
        )
        print(
            f"Grounding:              "
            f"{reply_quality['grounding']:.3f}/5"
        )
        print(
            f"Safety/factuality:      "
            f"{reply_quality['safety_factuality']:.3f}/5"
        )
        print(
            f"Escalation appropriateness: "
            f"{reply_quality['escalation_appropriateness']:.3f}/5"
        )
        print(
            f"Overall quality:        "
            f"{reply_quality['overall_quality']:.3f}/5"
        )

        if reply_quality[
            "unsupported_claim_rate"
        ] is not None:
            print(
                f"Unsupported claims:     "
                f"{reply_quality['unsupported_claim_rate']:.1%}"
            )

        if reply_quality[
            "judge_human_needed_rate"
        ] is not None:
            print(
                f"Judge human-needed:     "
                f"{reply_quality['judge_human_needed_rate']:.1%}"
            )

    else:
        print(
            f"Unavailable: {reply_quality['reason']}"
        )

    # ------------------------------------------------------------
    # Save machine-readable summary
    # ------------------------------------------------------------

    summary = {
        "intent": intent,
        "retrieval": retrieval,
        "escalation": escalation,
        "generation": generation,
        "reply_quality": reply_quality,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Save compact CSV
    # ------------------------------------------------------------

    rows = [
        {
            "component": "intent",
            "metric": "accuracy",
            "value": intent["accuracy"],
        },
        {
            "component": "intent",
            "metric": "macro_f1",
            "value": intent["macro_f1"],
        },
        {
            "component": "generation",
            "metric": "reply_coverage",
            "value": generation["reply_coverage"],
        },
        {
            "component": "generation",
            "metric": "valid_escalation_output_rate",
            "value": generation[
                "valid_escalation_output_rate"
            ],
        },
    ]

    if retrieval["available"]:
        rows.extend(
            [
                {
                    "component": "retrieval",
                    "metric": "mean_human_score",
                    "value": retrieval["mean_score"],
                },
                {
                    "component": "retrieval",
                    "metric": "direct_rate",
                    "value": retrieval["direct_rate"],
                },
                {
                    "component": "retrieval",
                    "metric": "useful_or_better_rate",
                    "value": retrieval[
                        "useful_or_better_rate"
                    ],
                },
                {
                    "component": "retrieval",
                    "metric": "irrelevant_rate",
                    "value": retrieval[
                        "irrelevant_rate"
                    ],
                },
            ]
        )

    if escalation["available"]:
        rows.extend(
            [
                {
                    "component": "escalation",
                    "metric": "gemini_accuracy",
                    "value": escalation[
                        "gemini"
                    ]["accuracy"],
                },
                {
                    "component": "escalation",
                    "metric": "gemini_precision",
                    "value": escalation[
                        "gemini"
                    ]["precision"],
                },
                {
                    "component": "escalation",
                    "metric": "gemini_recall",
                    "value": escalation[
                        "gemini"
                    ]["recall"],
                },
                {
                    "component": "escalation",
                    "metric": "gemini_f1",
                    "value": escalation[
                        "gemini"
                    ]["f1"],
                },
                {
                    "component": "escalation",
                    "metric": "rule_accuracy",
                    "value": escalation[
                        "rule_baseline"
                    ]["accuracy"],
                },
                {
                    "component": "escalation",
                    "metric": "rule_precision",
                    "value": escalation[
                        "rule_baseline"
                    ]["precision"],
                },
                {
                    "component": "escalation",
                    "metric": "rule_recall",
                    "value": escalation[
                        "rule_baseline"
                    ]["recall"],
                },
                {
                    "component": "escalation",
                    "metric": "rule_f1",
                    "value": escalation[
                        "rule_baseline"
                    ]["f1"],
                },
            ]
        )

    if reply_quality["available"]:
        reply_metrics = [
            "intent_alignment",
            "helpfulness",
            "grounding",
            "safety_factuality",
            "escalation_appropriateness",
            "overall_quality",
            "unsupported_claim_rate",
            "judge_human_needed_rate",
        ]

        for metric in reply_metrics:
            value = reply_quality.get(metric)

            if value is not None:
                rows.append(
                    {
                        "component": "reply_quality",
                        "metric": metric,
                        "value": value,
                    }
                )

    pd.DataFrame(rows).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------

    print("\n" + "=" * 72)
    print("EVALUATION SUMMARY SAVED")
    print("=" * 72)

    print(
        f"JSON: {OUTPUT_JSON}"
    )
    print(
        f"CSV:  {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()