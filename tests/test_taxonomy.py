from src.taxonomy.taxonomy import INTENTS, INTENT_NAMES


def test_taxonomy_is_frozen_at_13_intents():
    assert len(INTENTS) == 13
    assert len(INTENT_NAMES) == 13
    assert set(INTENT_NAMES) == set(INTENTS)
