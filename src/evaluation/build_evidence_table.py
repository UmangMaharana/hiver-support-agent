from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

FAILURES = ROOT / "data" / "evaluation" / "failure_analysis.csv"
PREDICTIONS = ROOT / "data" / "generation" / "reply_predictions.csv"
JUDGMENTS = ROOT / "data" / "evaluation" / "reply_judgments.csv"
ESCALATION = (
    ROOT / "data" / "escalation" / "escalation_human_eval_50.csv"
)
RETRIEVAL = (
    ROOT / "data" / "retrieval" / "retrieval_human_eval_150.csv"
)

OUTPUT_DIR = ROOT / "data" / "evaluation"
OUTPUT = OUTPUT_DIR / "report_evidence.csv"


def load():
    failures = pd.read_csv(FAILURES)
    predictions = pd.read_csv(PREDICTIONS)
    judgments = pd.read_csv(JUDGMENTS)

    df = predictions.merge(
        judgments[
            [
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
        ],
        on="golden_id",
        how="left",
    )

    # Add human escalation labels where available.
    if ESCALATION.exists():
        escalation = pd.read_csv(ESCALATION)

        escalation_cols = [
            "golden_id",
            "human_escalate",
            "human_reason",
            "policy_escalate",
            "policy_reason",
        ]

        escalation_cols = [
            c for c in escalation_cols
            if c in escalation.columns
        ]

        df = df.merge(
            escalation[escalation_cols],
            on="golden_id",
            how="left",
        )

    return df, failures


def get_case(df, golden_id):
    matches = df[
        df["golden_id"].astype(str)
        == str(golden_id)
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def add_case(rows, df, golden_id, category, interpretation):
    row = get_case(df, golden_id)

    if row is None:
        print(
            f"WARNING: {golden_id} not found."
        )
        return

    rows.append(
        {
            "golden_id": golden_id,
            "category": category,
            "engineering_interpretation": interpretation,
            "gold_intent": row.get("gold_intent"),
            "predicted_intent": row.get(
                "predicted_intent"
            ),
            "customer_message": row.get(
                "customer_message"
            ),
            "generated_reply": row.get("reply"),
            "model_escalate": row.get("escalate"),
            "human_escalate": row.get(
                "human_escalate"
            ),
            "policy_escalate": row.get(
                "policy_escalate"
            ),
            "intent_alignment": row.get(
                "intent_alignment"
            ),
            "helpfulness": row.get(
                "helpfulness"
            ),
            "grounding": row.get(
                "grounding"
            ),
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
            "needs_human_judge": row.get(
                "needs_human"
            ),
            "judge_reason": row.get(
                "concise_reason"
            ),
        }
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df, failures = load()

    rows = []

    # ------------------------------------------------------------
    # Carefully selected representative cases
    # ------------------------------------------------------------

    add_case(
        rows,
        df,
        "GOLD_028",
        "classification_cascade",
        "Product availability is misclassified as an online-order issue, causing the response to request unnecessary personal details.",
    )

    add_case(
        rows,
        df,
        "GOLD_029",
        "social_other_confusion",
        "Casual/non-support text is interpreted as a Tesco support issue, demonstrating the classifier's weakness on the other class.",
    )

    add_case(
        rows,
        df,
        "GOLD_038",
        "informational_over_escalation",
        "A general nutrition-label question is escalated even though it does not inherently require human intervention.",
    )

    add_case(
        rows,
        df,
        "GOLD_006",
        "conversational_quality",
        "The system handles casual social chatter safely but responds in a support-oriented style rather than naturally engaging with the message.",
    )

    add_case(
        rows,
        df,
        "GOLD_111",
        "live_information",
        "Current store stock and price require live information that historical retrieval cannot reliably provide.",
    )

    add_case(
        rows,
        df,
        "GOLD_031",
        "account_specific_escalation",
        "Delivery Saver eligibility and booking terms are account/current-policy dependent and appropriately require human intervention.",
    )

    add_case(
        rows,
        df,
        "GOLD_103",
        "retrieval_failure",
        "The retrieval system surfaces a poor historical precedent for a product-quality complaint, illustrating the limits of lexical similarity.",
    )

    add_case(
        rows,
        df,
        "GOLD_212",
        "incomplete_grounding",
        "The response is cautious and safe but does not fully exploit the actionable guidance available in historical support conversations.",
    )

    add_case(
        rows,
        df,
        "GOLD_219",
        "wrong_intent_good_reply",
        "The predicted intent is wrong, yet the generated response still addresses the underlying customer problem reasonably well.",
    )

    # One safety-critical human escalation case.
    add_case(
        rows,
        df,
        "GOLD_122",
        "product_safety",
        "A mould-related food-safety complaint requires human investigation and an appropriate resolution.",
    )

    # ------------------------------------------------------------
    # Automatically identify Gemini escalation false negative
    # ------------------------------------------------------------

    if "human_escalate" in df.columns:
        df["human_escalate_bool"] = (
            df["human_escalate"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "yes": True,
                    "no": False,
                }
            )
        )

        df["model_escalate_bool"] = (
            df["escalate"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "yes": True,
                    "no": False,
                }
            )
        )

        false_negatives = df[
            (df["human_escalate_bool"] == True)
            & (df["model_escalate_bool"] == False)
        ]

        if not false_negatives.empty:
            row = false_negatives.iloc[0]

            add_case(
                rows,
                df,
                str(row["golden_id"]),
                "escalation_false_negative",
                "The model failed to escalate a case that the human benchmark judged to require intervention.",
            )

    evidence = pd.DataFrame(rows)

    # Remove duplicate golden IDs while preserving
    # the first selected interpretation.
    evidence = evidence.drop_duplicates(
        subset=["golden_id"],
        keep="first",
    )

    evidence.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 72)
    print("REPORT EVIDENCE TABLE")
    print("=" * 72)

    print(
        f"Representative cases: {len(evidence)}"
    )

    print("\nCategories:")

    print(
        evidence["category"]
        .value_counts()
        .to_string()
    )

    print(
        f"\nSaved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()