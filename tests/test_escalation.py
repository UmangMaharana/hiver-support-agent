from src.escalation.escalation_policy import should_escalate


def test_explicit_complaint_escalates():
    decision, reason = should_escalate(
        "I want to make a complaint about the service",
        "store_service_complaint",
    )
    assert decision is True
    assert reason


def test_simple_social_message_does_not_escalate():
    decision, reason = should_escalate(
        "Love the new Clubcard prices!",
        "other",
    )
    assert decision is False
    assert reason == ""
