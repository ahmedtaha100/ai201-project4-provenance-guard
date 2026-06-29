"""Pipeline tests — deterministic (force the offline fallback, no network).

Run: ``pytest -q`` from the repo root.

These tests pin the behaviours the spec cares about:
* signals move in the right direction on clearly-AI vs clearly-human text,
* the combined score is calibrated (not a binary flip) and asymmetric,
* every transparency label variant is reachable and matches its canonical text,
* the audit log + appeals workflow persist and update correctly.
"""

from __future__ import annotations

import pytest

import config

# --------------------------------------------------------------------------- #
# Starter inputs from the project spec
# --------------------------------------------------------------------------- #
CLEARLY_AI = (
    "Artificial intelligence represents a transformative paradigm shift in modern "
    "society. It is important to note that while the benefits of AI are numerous, "
    "it is equally essential to consider the ethical implications. Furthermore, "
    "stakeholders across various sectors must collaborate to ensure responsible "
    "deployment."
)
CLEARLY_HUMAN = (
    "ok so i finally tried that new ramen place downtown and honestly? "
    "underwhelming. the broth was fine but they put WAY too much sodium in it and "
    "i was thirsty for like three hours after. my friend got the spicy version and "
    "said it was better. probably won't go back unless someone drags me there"
)
FORMAL_HUMAN = (
    "The relationship between monetary policy and asset price inflation has been "
    "extensively studied in the literature. Central banks face a fundamental "
    "tension between their mandate for price stability and the unintended "
    "consequences of prolonged low interest rates on equity and real estate "
    "valuations."
)
EDITED_AI = (
    "I've been thinking a lot about remote work lately. There are genuine "
    "tradeoffs — flexibility and no commute on one side, isolation and blurred "
    "work-life boundaries on the other. Studies show productivity varies widely by "
    "individual and role type."
)


@pytest.fixture(autouse=True)
def _force_fallback(monkeypatch):
    """Make the LLM signal deterministic by removing the API key."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "")


# --------------------------------------------------------------------------- #
# Signal 1 — LLM fallback heuristic
# --------------------------------------------------------------------------- #
def test_llm_fallback_separates_ai_from_human():
    from signals import llm_signal

    ai = llm_signal.assess(CLEARLY_AI)
    human = llm_signal.assess(CLEARLY_HUMAN)
    assert ai.fallback_used and human.fallback_used
    assert ai.ai_probability > human.ai_probability
    assert ai.ai_probability > 0.55
    assert human.ai_probability < 0.45


# --------------------------------------------------------------------------- #
# Signal 2 — stylometric
# --------------------------------------------------------------------------- #
def test_stylometric_separates_uniform_from_bursty():
    from signals import stylometric

    ai = stylometric.assess(CLEARLY_AI)
    human = stylometric.assess(CLEARLY_HUMAN)
    assert ai.ai_probability > human.ai_probability
    assert ai.name == config.SIGNAL_STYLOMETRIC


def test_stylometric_short_text_pulled_to_neutral():
    from signals import stylometric

    res = stylometric.assess("Short and sharp.")
    # Short text still contributes (LLM + stylometric must stay >=2 signals), but
    # its score is shrunk toward neutral and flagged unreliable in the detail.
    assert res.available
    assert res.detail["reliable"] is False
    assert 0.35 <= res.ai_probability <= 0.65


def test_repetition_signal_dropped_on_short_text():
    from signals import ensemble

    res = ensemble.assess("Short and sharp and not enough tokens here.")
    assert not res.available  # below MIN_REP_WORDS -> dropped by scoring


# --------------------------------------------------------------------------- #
# Signal 3 — repetition / predictability (ensemble)
# --------------------------------------------------------------------------- #
def test_repetition_signal_runs_and_is_bounded():
    from signals import ensemble

    res = ensemble.assess(CLEARLY_AI)
    assert res.name == config.SIGNAL_REPETITION
    assert 0.0 <= res.ai_probability <= 1.0


# --------------------------------------------------------------------------- #
# Scoring — calibration, asymmetry, disagreement
# --------------------------------------------------------------------------- #
def _sig(name, p):
    from signals import SignalResult

    return SignalResult(name=name, ai_probability=p)


def test_scoring_both_high_is_likely_ai():
    import scoring

    res = scoring.combine([_sig(config.SIGNAL_LLM, 0.9), _sig(config.SIGNAL_STYLOMETRIC, 0.85)])
    assert res.attribution == config.ATTRIBUTION_LIKELY_AI
    assert res.confidence >= config.AI_THRESHOLD


def test_scoring_both_low_is_likely_human():
    import scoring

    res = scoring.combine([_sig(config.SIGNAL_LLM, 0.1), _sig(config.SIGNAL_STYLOMETRIC, 0.15)])
    assert res.attribution == config.ATTRIBUTION_LIKELY_HUMAN
    assert res.confidence <= config.HUMAN_THRESHOLD


def test_scoring_disagreement_yields_uncertain():
    import scoring

    res = scoring.combine([_sig(config.SIGNAL_LLM, 0.9), _sig(config.SIGNAL_STYLOMETRIC, 0.1)])
    assert res.attribution == config.ATTRIBUTION_UNCERTAIN
    assert res.disagreement >= 0.5
    assert 0.4 < res.confidence < 0.7  # pulled toward the uncertain middle


def test_scoring_is_not_a_binary_flip_at_half():
    import scoring

    low = scoring.combine([_sig(config.SIGNAL_LLM, 0.52), _sig(config.SIGNAL_STYLOMETRIC, 0.5)])
    high = scoring.combine([_sig(config.SIGNAL_LLM, 0.96), _sig(config.SIGNAL_STYLOMETRIC, 0.94)])
    # A 0.51-ish blend must NOT read the same as a 0.95 blend.
    assert low.attribution != high.attribution
    assert high.confidence - low.confidence > 0.2


def test_asymmetry_human_bar_is_easier_than_ai_bar():
    # The distance from 0.5 required to call AI is larger than to call human.
    assert (config.AI_THRESHOLD - 0.5) > (0.5 - config.HUMAN_THRESHOLD)


# --------------------------------------------------------------------------- #
# Labels — all three reachable, text is canonical
# --------------------------------------------------------------------------- #
def test_labels_all_three_reachable_and_canonical():
    import labels
    import scoring

    seen = set()
    for p in (0.95, 0.55, 0.05):
        score = scoring.combine([_sig(config.SIGNAL_LLM, p), _sig(config.SIGNAL_STYLOMETRIC, p)])
        lab = labels.make_label(score)
        seen.add(lab["attribution"])
        template = {
            config.ATTRIBUTION_LIKELY_AI: labels.LABEL_LIKELY_AI,
            config.ATTRIBUTION_UNCERTAIN: labels.LABEL_UNCERTAIN,
            config.ATTRIBUTION_LIKELY_HUMAN: labels.LABEL_LIKELY_HUMAN,
        }[lab["attribution"]]
        assert lab["text"] == template.format(pct=round(score.confidence * 100))
    assert seen == {
        config.ATTRIBUTION_LIKELY_AI,
        config.ATTRIBUTION_UNCERTAIN,
        config.ATTRIBUTION_LIKELY_HUMAN,
    }


def test_label_carries_verified_badge():
    import labels
    import scoring

    score = scoring.combine([_sig(config.SIGNAL_LLM, 0.1), _sig(config.SIGNAL_STYLOMETRIC, 0.1)])
    lab = labels.make_label(score, verified_human=True)
    assert lab["badge"] == labels.VERIFIED_HUMAN_BADGE
    assert labels.VERIFIED_HUMAN_BADGE in lab["text"]


# --------------------------------------------------------------------------- #
# End-to-end pipeline on the spec's starter set (fallback path)
# --------------------------------------------------------------------------- #
def test_pipeline_scores_match_intuition():
    import pipeline

    ai = pipeline.classify(CLEARLY_AI).score
    human = pipeline.classify(CLEARLY_HUMAN).score
    # Clearly-AI text scores noticeably higher than clearly-human text.
    assert ai.confidence > human.confidence + 0.2
    assert human.attribution == config.ATTRIBUTION_LIKELY_HUMAN
    assert ai.attribution in (config.ATTRIBUTION_LIKELY_AI, config.ATTRIBUTION_UNCERTAIN)


def test_pipeline_borderline_inputs_are_bounded():
    import pipeline

    for text in (FORMAL_HUMAN, EDITED_AI):
        score = pipeline.classify(text).score
        assert 0.0 <= score.confidence <= 1.0
        assert score.attribution in (
            config.ATTRIBUTION_LIKELY_AI,
            config.ATTRIBUTION_UNCERTAIN,
            config.ATTRIBUTION_LIKELY_HUMAN,
        )


# --------------------------------------------------------------------------- #
# Audit log + appeals
# --------------------------------------------------------------------------- #
@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    db = tmp_path / "test_audit.db"
    monkeypatch.setattr(config, "DB_PATH", str(db))
    import audit
    import certificate

    audit.init_db()
    certificate.init_db()
    return str(db)


def test_audit_record_and_retrieve(temp_db):
    import audit

    audit.record_submission(
        {
            "content_id": "cid-1",
            "creator_id": "user-1",
            "attribution": config.ATTRIBUTION_LIKELY_AI,
            "confidence": 0.81,
            "llm_score": 0.84,
            "stylometric_score": 0.77,
            "repetition_score": 0.7,
            "signal_detail": {"llm": {"x": 1}},
            "fallback_used": True,
        }
    )
    row = audit.get_by_content_id("cid-1")
    assert row is not None
    assert row["attribution"] == config.ATTRIBUTION_LIKELY_AI
    assert row["llm_score"] == 0.84 and row["stylometric_score"] == 0.77
    assert row["status"] == "classified"
    assert isinstance(row["signal_detail"], dict)


def test_appeal_updates_status_and_logs_reasoning(temp_db):
    import appeals
    import audit

    audit.record_submission(
        {
            "content_id": "cid-2",
            "creator_id": "user-2",
            "attribution": config.ATTRIBUTION_LIKELY_AI,
            "confidence": 0.8,
        }
    )
    confirmation = appeals.submit_appeal("cid-2", "I wrote this myself over three days.")
    assert confirmation["status"] == "under_review"
    row = audit.get_by_content_id("cid-2")
    assert row["status"] == "under_review"
    assert "three days" in row["appeal_reasoning"]
    assert row["appeal_timestamp"] is not None
    # The reviewer queue surfaces it.
    assert any(r["content_id"] == "cid-2" for r in audit.get_under_review())


def test_appeal_unknown_content_id_raises_404(temp_db):
    import appeals

    with pytest.raises(appeals.AppealError) as exc:
        appeals.submit_appeal("does-not-exist", "reasoning")
    assert exc.value.status_code == 404


# --------------------------------------------------------------------------- #
# Certificate (stretch)
# --------------------------------------------------------------------------- #
def test_certificate_issued_for_human_sample(temp_db):
    import certificate

    cert = certificate.request_certificate("creator-h", CLEARLY_HUMAN)
    assert cert["verified_human"] is True
    assert certificate.is_verified_human("creator-h")


def test_certificate_rejects_short_sample(temp_db):
    import certificate

    with pytest.raises(certificate.CertificateError) as exc:
        certificate.request_certificate("creator-x", "too short")
    assert exc.value.status_code == 400
