from app.services.rag_service import flare_triggers_follow_up


def test_flare_triggers_on_uncertainty_marker() -> None:
    assert flare_triggers_follow_up("The model uses Adam??? for optimization in the cited work.") is True


def test_flare_triggers_on_hedge_phrase() -> None:
    assert flare_triggers_follow_up("The learning rate is not stated in excerpt for this architecture.") is True


def test_flare_no_trigger_on_confident_draft() -> None:
    assert flare_triggers_follow_up("The paper describes a transformer encoder with multi-head attention.") is False
