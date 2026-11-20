import pytest

from expensebench.review import agreement


def test_agreement_aligns_cases_by_id_and_handles_constant_ratings():
    a = {"a": "approve", "b": "review", "c": "reject"}
    b = {"b": "review", "c": "approve", "a": "approve"}
    scores = agreement(a, b)
    assert scores["agreements"] == 2
    assert scores["cohens_kappa"] == pytest.approx(0.5)
    assert scores["disagreed_cases"] == ["c"]
    assert agreement({"a": "approve"}, {"a": "approve"})["cohens_kappa"] is None
    with pytest.raises(ValueError):
        agreement(a, {"a": "approve"})
