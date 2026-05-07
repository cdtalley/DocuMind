import json
from pathlib import Path


def test_eval_cases_exist() -> None:
    payload = json.loads(Path("evaluation/test_cases.json").read_text(encoding="utf-8"))
    assert len(payload) == 10
    for case in payload:
        assert "question" in case
        assert case["query_mode"] in {"general", "compare", "methodology", "datasets", "reproduce"}
        assert isinstance(case["expected_keywords"], list)
