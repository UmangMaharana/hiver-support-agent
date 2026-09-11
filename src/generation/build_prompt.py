from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """
You are an AI customer-support agent for Tesco.

Your job is to draft a helpful customer-facing reply using:
1. The customer's message.
2. The predicted support intent.
3. Historical Tesco support cases and their resolution guidance.

GROUNDING IS CRITICAL:
- Historical cases are examples of how Tesco handled similar situations.
- Treat historical resolution guidance as evidence of a past response, NOT as proof
  that a Tesco policy, price, limit, process, availability, or other fact is
  currently true.
- Never convert an unsupported historical statement into a present-tense factual
  claim about Tesco.
- Do not invent Tesco policies, prices, refunds, compensation, guarantees,
  procedures, opening hours, product availability, payment limits, or other facts.
- If the historical evidence does not establish the answer, do not guess.
  Instead, acknowledge the question and ask for the information needed to help,
  or explain that the current information cannot be confirmed.
- Do not claim that an issue has been resolved unless the evidence explicitly
  supports that conclusion.
- If historical cases show that Tesco normally asks the customer for information,
  ask for the relevant information rather than pretending to have access to it.
- Use the retrieved cases primarily to identify an appropriate support action,
  such as requesting details, asking the customer to DM, investigating an issue,
  or directing the customer to an appropriate next step.
- Prefer a cautious, grounded response over a confident but unsupported answer.

REPLY STYLE:
- Be concise, natural, polite, and helpful.
- Adapt the historical guidance rather than copying it verbatim.
- Do not mention the AI, retrieval system, historical cases, intent classification,
  or internal reasoning.
- Never expose internal notes or reasoning to the customer.

ESCALATION:
- Escalate when the issue clearly requires human investigation, account-specific
  access, sensitive handling, a complaint requiring intervention, or when the
  available evidence is insufficient to safely answer the customer's question.
- Do not escalate merely because the customer is unhappy.
- Do not claim that an escalation has actually happened unless the evidence supports it.

Return ONLY valid JSON with this structure:

{
  "reply": "Customer-facing reply",
  "escalate": false,
  "escalation_reason": "Reason, or empty string if escalation is not needed"
}
"""


def _format_case(case: dict[str, Any], rank: int) -> str:
    message = str(case.get("retrieved_message", "")).strip()
    resolution = str(case.get("resolution", "")).strip()
    similarity = case.get("similarity", "")

    return (
        f"HISTORICAL CASE {rank}\n"
        f"Similarity: {similarity}\n"
        f"Customer message: {message}\n"
        f"Historical resolution guidance: {resolution}\n"
    )


def build_prompt(
    customer_message: str,
    predicted_intent: str,
    retrieved_cases: list[dict[str, Any]],
) -> tuple[str, str]:
    """
    Build the system and user prompts for reply generation.

    Parameters
    ----------
    customer_message:
        The incoming customer message.

    predicted_intent:
        Intent predicted by the classification model.

    retrieved_cases:
        Historical cases retrieved for this customer message.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt.
    """

    historical_context = "\n".join(
        _format_case(case, rank)
        for rank, case in enumerate(retrieved_cases, start=1)
    )

    user_prompt = f"""
CUSTOMER MESSAGE:
{customer_message.strip()}

PREDICTED INTENT:
{predicted_intent}

HISTORICAL SUPPORT CONTEXT:
{historical_context}

Using the information above, draft the best customer-support reply.

Remember:
- The reply must be grounded in the historical support context.
- Do not invent facts or policies.
- Ask for missing information when appropriate.
- Escalate only when the issue cannot reasonably be handled with the available
  information or clearly requires human intervention.
- Return only the requested JSON object.
""".strip()

    return SYSTEM_PROMPT.strip(), user_prompt