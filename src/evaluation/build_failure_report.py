from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "evaluation" / "failure_analysis.csv"
OUTPUT = ROOT / "data" / "evaluation" / "failure_analysis_report.md"


def clean(value):
    if pd.isna(value):
        return "N/A"

    return str(value).strip()


def pct(value):
    return f"{float(value) * 100:.1f}%"


def add_case(lines, row, include_reason=True):
    lines.append(
        f"### {clean(row.get('golden_id'))}"
    )

    lines.append(
        f"**Gold intent:** `{clean(row.get('gold_intent'))}`  "
        f"**Predicted:** `{clean(row.get('predicted_intent'))}`"
    )

    if clean(row.get("customer_message")) != "N/A":
        lines.append(
            f"\n**Customer:**\n> {clean(row.get('customer_message'))}"
        )

    if clean(row.get("reply")) != "N/A":
        lines.append(
            f"\n**Generated reply:**\n> {clean(row.get('reply'))}"
        )

    if clean(row.get("failure_detail")) != "N/A":
        lines.append(
            f"\n**Failure:** {clean(row.get('failure_detail'))}"
        )

    if include_reason:
        reason = clean(row.get("concise_reason"))

        if reason != "N/A":
            lines.append(
                f"**Judge reason:** {reason}"
            )

    lines.append("")


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing failure analysis: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    required = {
        "section",
        "golden_id",
        "gold_intent",
        "predicted_intent",
        "customer_message",
        "reply",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Failure analysis missing columns: {sorted(missing)}"
        )

    lines = []

    lines.append("# Tesco Support Agent — Failure Analysis")
    lines.append("")
    lines.append(
        "This report is generated directly from the frozen "
        "evaluation artifacts. It is intended to support "
        "manual review and selection of representative cases "
        "for the final project report."
    )
    lines.append("")

    # ============================================================
    # Executive summary
    # ============================================================

    lines.append("## 1. Executive Summary")
    lines.append("")

    total = len(df)

    # Intent errors are counted from the actual prediction data
    # represented in the failure-analysis artifact.
    intent_errors = df[
        df["section"] == "intent_error"
    ]

    # The intent_error section contains the top 20 rather than
    # every error, so use the known evaluation total/rate here.
    lines.append(
        "- Intent classification: **101/250 errors (40.4%)**."
    )

    lines.append(
        "- The dominant classification problem is confusion "
        "among `other`, `store_service_complaint`, and "
        "`product_quality_safety`."
    )

    lines.append(
        "- Gemini escalation produced **7 false positives and "
        "1 false negative** on the 50-example human benchmark."
    )

    lines.append(
        "- The deterministic rule baseline produced **1 false "
        "positive and 23 false negatives**."
    )

    lines.append(
        "- The reply judge flagged **4/250 responses (1.6%)** "
        "for unsupported claims."
    )

    lines.append(
        "- **15 cases** had an incorrect predicted intent but "
        "still received a high-quality reply (judge score >= 4/5)."
    )

    lines.append(
        "- **2 cases** had the correct intent but a low-quality "
        "reply (judge score <= 3/5)."
    )

    lines.append("")

    # ============================================================
    # Intent confusion
    # ============================================================

    lines.append("## 2. Intent Classification Failures")
    lines.append("")

    confusion = df[
        df["section"] == "intent_confusion_pair"
    ].copy()

    if not confusion.empty:
        lines.append(
            "The most frequent confusion pairs are:"
        )
        lines.append("")

        for _, row in confusion.iterrows():
            lines.append(
                f"- `{clean(row.get('gold_intent'))}` → "
                f"`{clean(row.get('predicted_intent'))}`: "
                f"{clean(row.get('failure_detail'))}"
            )

        lines.append("")

    intent_errors = df[
        df["section"] == "intent_error"
    ].copy()

    if not intent_errors.empty:
        lines.append(
            "### Representative intent errors"
        )
        lines.append("")

        for _, row in intent_errors.head(10).iterrows():
            add_case(lines, row)

    # ============================================================
    # Retrieval
    # ============================================================

    lines.append("## 3. Retrieval Failures")
    lines.append("")

    retrieval = df[
        df["section"] == "retrieval_failure"
    ].copy()

    if retrieval.empty:
        lines.append(
            "No retrieval failures were found in the "
            "failure-analysis artifact."
        )
        lines.append("")
    else:
        lines.append(
            "The following are among the lowest-scoring "
            "human-evaluated retrieval pairs."
        )
        lines.append("")

        if "retrieval_score" in retrieval.columns:
            retrieval = retrieval.sort_values(
                "retrieval_score",
                ascending=True,
            )

        for _, row in retrieval.head(10).iterrows():
            lines.append(
                f"### {clean(row.get('golden_id'))}"
            )

            lines.append(
                f"**Human retrieval score:** "
                f"{clean(row.get('retrieval_score'))}"
            )

            message = clean(
                row.get("customer_message")
            )

            if message != "N/A":
                lines.append(
                    f"\n**Customer:**\n> {message}"
                )

            lines.append("")

    # ============================================================
    # Lowest quality replies
    # ============================================================

    lines.append("## 4. Lowest-Quality Generated Replies")
    lines.append("")

    quality = df[
        df["section"] == "lowest_quality_reply"
    ].copy()

    if not quality.empty:
        quality = quality.sort_values(
            "overall_quality",
            ascending=True,
        )

        for _, row in quality.head(10).iterrows():
            add_case(lines, row)

    # ============================================================
    # Unsupported claims
    # ============================================================

    lines.append("## 5. Unsupported Claims")
    lines.append("")

    unsupported = df[
        df["section"] == "unsupported_claim"
    ].copy()

    if unsupported.empty:
        lines.append(
            "No unsupported-claim cases were found."
        )
        lines.append("")
    else:
        lines.append(
            f"Found **{len(unsupported)}** judge-flagged "
            "unsupported-claim cases."
        )
        lines.append("")

        for _, row in unsupported.iterrows():
            add_case(lines, row)

    # ============================================================
    # Gemini escalation false positives
    # ============================================================

    lines.append("## 6. Gemini Escalation False Positives")
    lines.append("")

    fp = df[
        df["section"] == "escalation_false_positive"
    ].copy()

    if fp.empty:
        lines.append(
            "No Gemini escalation false positives were found."
        )
        lines.append("")
    else:
        lines.append(
            f"Found **{len(fp)}** cases where Gemini escalated "
            "but the human benchmark said no escalation was needed."
        )
        lines.append("")

        for _, row in fp.iterrows():
            add_case(lines, row)

    # ============================================================
    # Gemini escalation false negatives
    # ============================================================

    lines.append("## 7. Gemini Escalation False Negatives")
    lines.append("")

    fn = df[
        df["section"] == "escalation_false_negative"
    ].copy()

    if fn.empty:
        lines.append(
            "No Gemini escalation false negatives were found."
        )
        lines.append("")
    else:
        lines.append(
            f"Found **{len(fn)}** case(s) where Gemini did not "
            "escalate but the human benchmark said escalation was needed."
        )
        lines.append("")

        for _, row in fn.iterrows():
            add_case(lines, row)

    # ============================================================
    # Rule baseline failures
    # ============================================================

    lines.append("## 8. Rule Baseline Failures")
    lines.append("")

    rule_fn = df[
        df["section"] == "rule_false_negative"
    ].copy()

    rule_fp = df[
        df["section"] == "rule_false_positive"
    ].copy()

    lines.append(
        f"- Rule false negatives: **{len(rule_fn)}** "
        "in the 50-example human benchmark."
    )

    lines.append(
        f"- Rule false positives: **{len(rule_fp)}** "
        "in the 50-example human benchmark."
    )

    lines.append("")

    if not rule_fn.empty:
        lines.append(
            "The rule baseline's main weakness is under-escalation:"
        )
        lines.append("")

        for _, row in rule_fn.head(8).iterrows():
            add_case(lines, row, include_reason=False)

    # ============================================================
    # Wrong intent, good reply
    # ============================================================

    lines.append("## 9. Wrong Intent but Good Reply")
    lines.append("")

    wrong_good = df[
        df["section"] == "wrong_intent_good_reply"
    ].copy()

    if not wrong_good.empty:
        lines.append(
            "These cases demonstrate that classification accuracy "
            "is not perfectly coupled to end-to-end response quality."
        )
        lines.append("")

        wrong_good = wrong_good.sort_values(
            "overall_quality",
            ascending=False,
        )

        for _, row in wrong_good.head(8).iterrows():
            add_case(lines, row)

    # ============================================================
    # Correct intent, bad reply
    # ============================================================

    lines.append("## 10. Correct Intent but Bad Reply")
    lines.append("")

    correct_bad = df[
        df["section"] == "correct_intent_bad_reply"
    ].copy()

    if not correct_bad.empty:
        lines.append(
            "These cases isolate failures in retrieval, generation, "
            "grounding, or escalation despite correct classification."
        )
        lines.append("")

        correct_bad = correct_bad.sort_values(
            "overall_quality",
            ascending=True,
        )

        for _, row in correct_bad.iterrows():
            add_case(lines, row)

    # ============================================================
    # Grounding
    # ============================================================

    lines.append("## 11. Low Grounding")
    lines.append("")

    grounding = df[
        df["section"] == "low_grounding"
    ].copy()

    if not grounding.empty:
        grounding = grounding.sort_values(
            "grounding",
            ascending=True,
        )

        for _, row in grounding.head(8).iterrows():
            add_case(lines, row)

    # ============================================================
    # Safety
    # ============================================================

    lines.append("## 12. Low Safety / Factuality")
    lines.append("")

    safety = df[
        df["section"] == "low_safety"
    ].copy()

    if not safety.empty:
        safety = safety.sort_values(
            "safety_factuality",
            ascending=True,
        )

        for _, row in safety.head(8).iterrows():
            add_case(lines, row)

    # ============================================================
    # Recommended report cases
    # ============================================================

    lines.append("## 13. Recommended Cases for Final Report")
    lines.append("")

    lines.append(
        "The final 6-page report should not include dozens of examples. "
        "A compact set of representative cases is more useful."
    )
    lines.append("")

    recommendations = [
        (
            "Classification",
            intent_errors.head(2)
        ),
        (
            "Retrieval",
            retrieval.head(2)
        ),
        (
            "Escalation false positive",
            fp.head(1)
        ),
        (
            "Escalation false negative",
            fn.head(1)
        ),
        (
            "Unsupported claim",
            unsupported.head(1)
        ),
        (
            "Wrong intent but good reply",
            wrong_good.head(1)
        ),
        (
            "Correct intent but bad reply",
            correct_bad.head(1)
        ),
    ]

    for label, cases in recommendations:
        if cases is None or cases.empty:
            continue

        row = cases.iloc[0]

        lines.append(
            f"- **{label}:** `{clean(row.get('golden_id'))}` — "
            f"{clean(row.get('failure_detail'))}"
        )

    lines.append("")

    lines.append(
        "These examples should be manually reviewed before inclusion "
        "in the final report. Automated ranking is used only to surface "
        "candidates; it does not replace qualitative analysis."
    )

    lines.append("")

    # ============================================================
    # Write
    # ============================================================

    OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 72)
    print("FAILURE ANALYSIS REPORT")
    print("=" * 72)
    print(
        f"Generated from {total} failure-analysis rows."
    )
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()