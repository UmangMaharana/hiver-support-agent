from src.generation.build_prompt import build_prompt


def test_prompt_keeps_customer_message_and_intent():
    system, user = build_prompt(
        "My delivery is late",
        "delivery_issue",
        [
            {
                "retrieved_message": "My delivery was late",
                "resolution": "Contact support with the order details.",
                "similarity": 0.5,
            }
        ],
    )

    assert "customer's actual message" in system.lower()
    assert "My delivery is late" in user
    assert "delivery_issue" in user
    assert "Historical resolution guidance" in user
