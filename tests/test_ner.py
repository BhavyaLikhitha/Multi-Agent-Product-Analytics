"""Tests for spaCy NER entity extraction."""

import pytest

from src.features.ner_extractor import extract_batch, extract_from_text, load_nlp


@pytest.fixture(scope="module")
def nlp():
    return load_nlp()


SAMPLE_REVIEWS = [
    "The battery died after just 2 months of use.",
    "Bluetooth keeps disconnecting every few minutes.",
    "Screen is cracked right out of the box, clearly defective.",
    "The charger cable is loose and won't charge properly.",
    "Speaker has a buzzing noise at high volume.",
    "Camera quality is terrible, very dim in low light.",
    "USB port stopped working after a week.",
    "The keyboard feels great but the trackpad is slow and laggy.",
    "Fan is noisy and the laptop keeps overheating.",
    "Remote control buttons are broken, very disappointed.",
]


def test_extract_single_review(nlp):
    result = extract_from_text(nlp, SAMPLE_REVIEWS[0])
    assert "battery" in result["components"]


def test_extract_bluetooth_issue(nlp):
    result = extract_from_text(nlp, SAMPLE_REVIEWS[1])
    assert "bluetooth" in result["components"]
    assert any("disconnect" in i for i in result["issues"])


def test_extract_multiple_entities(nlp):
    result = extract_from_text(nlp, SAMPLE_REVIEWS[7])
    assert "keyboard" in result["components"] or "trackpad" in result["components"]


def test_extract_batch(nlp):
    results = extract_batch(nlp, SAMPLE_REVIEWS)
    assert len(results) == 10
    assert all(isinstance(r, dict) for r in results)
    assert all(
        "components" in r and "issues" in r and "time_refs" in r for r in results
    )


def test_empty_text(nlp):
    result = extract_from_text(nlp, "")
    assert result["components"] == []
    assert result["issues"] == []


def test_no_entities(nlp):
    result = extract_from_text(nlp, "Great product, works perfectly fine!")
    assert isinstance(result["components"], list)
    assert isinstance(result["issues"], list)


def test_expanded_issues(nlp):
    """Test new issue patterns added after EDA revealed weak detection."""
    result = extract_from_text(nlp, "This is garbage, total waste of money")
    assert "garbage" in result["issues"] or "waste of money" in result["issues"]


def test_negative_quality_words(nlp):
    result = extract_from_text(nlp, "The battery died and the charger is flimsy")
    assert "battery" in result["components"]
    assert "charger" in result["components"]
    assert "died" in result["issues"] or "flimsy" in result["issues"]


def test_shipping_issues(nlp):
    result = extract_from_text(nlp, "Item arrived damaged with missing parts")
    assert "arrived damaged" in result["issues"] or "missing parts" in result["issues"]
