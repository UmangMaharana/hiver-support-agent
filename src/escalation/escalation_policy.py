import re


# Intents where a human commonly needs to investigate
HIGH_RISK_INTENTS = {
    "delivery_issue",
    "online_order_issue",
    "product_quality_safety",
    "refund_return_compensation",
    "store_service_complaint",
    "clubcard_account",
}

# Intents that can often be handled automatically
LOW_RISK_INTENTS = {
    "other",
    "product_information",
    "pricing_offers",
    "product_availability",
    "delivery_slots",
    "click_collect",
    "online_website_issue",
}


ESCALATION_PATTERNS = [
    # Account / personal information
    r"\b(account|clubcard|voucher|points)\b.*\b(not working|wrong|missing|can't|cannot|unable)\b",

    # Refund / compensation
    r"\b(refund|refunds|compensation|money back|reimbursement)\b",

    # Serious product quality / safety
    r"\b(mould|mold|mouldy|moldy|expired|out of date|spoiled|spoilt)\b",
    r"\b(maggots?|caterpillar|insect|bug|foreign object)\b",
    r"\b(food poisoning|poisoned|unsafe|contaminated)\b",

    # Delivery / order investigation
    r"\b(delivery|delivered|order)\b.*\b(missing|late|late|wrong|failed|not arrived)\b",

    # Store / staff complaints
    r"\b(staff|employee|colleague|manager|driver)\b.*\b(rude|abusive|complaint|complain)\b",

    # Explicit escalation / complaint request
    r"\b(complaint|complain|escalate|manager|supervisor)\b",
]


def should_escalate(
    customer_message: str,
    predicted_intent: str,
) -> tuple[bool, str]:

    text = customer_message.lower().strip()

    # Explicit complaints or requests for human intervention
    if re.search(
        r"\b(complaint|complain|escalate|manager|supervisor)\b",
        text,
    ):
        return (
            True,
            "Customer explicitly requests or raises a complaint requiring human support.",
        )

    # Strong issue-specific signals
    for pattern in ESCALATION_PATTERNS:
        if re.search(pattern, text):
            return (
                True,
                "The issue contains signals that typically require human investigation or account-specific support.",
            )

    # High-risk intents are escalated by default when the customer
    # is describing an actual problem rather than simple praise/social chatter.
    if predicted_intent in HIGH_RISK_INTENTS:
        problem_signals = [
            "problem",
            "issue",
            "wrong",
            "missing",
            "failed",
            "can't",
            "cannot",
            "unable",
            "broken",
            "charged",
            "complaint",
            "refund",
            "late",
            "damaged",
            "not working",
        ]

        if any(signal in text for signal in problem_signals):
            return (
                True,
                f"{predicted_intent} contains a problem signal that may require human investigation.",
            )

    # Low-risk informational/social cases stay automated.
    return False, ""