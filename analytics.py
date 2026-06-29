"""Analytics dashboard (stretch).

Aggregates the audit log into platform-level detection metrics:

* **Detection patterns** — distribution of attributions (likely_ai / uncertain /
  likely_human) and the mean confidence.
* **Appeal rate** — appeals filed ÷ total submissions, plus the current
  reviewer-queue depth.
* **Signal agreement** (the "one additional metric") — how often the LLM and
  stylometric signals land in the same band; a low value flags either noisy
  signals or genuinely ambiguous content.
* Plus fallback-usage rate and verified-human creator count.

``compute()`` returns plain JSON; ``render_html()`` wraps it in a minimal,
dependency-free dashboard page.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from typing import Any

import audit
import config
import scoring


def _band(p: float | None) -> str | None:
    return scoring.attribution_for(p) if p is not None else None


def compute() -> dict[str, Any]:
    """Compute dashboard metrics from the full audit log."""
    rows = audit.get_all()
    total = len(rows)

    attribution_counts = Counter(r["attribution"] for r in rows)
    appeals = [r for r in rows if r["status"] == "under_review" or r.get("appeal_reasoning")]
    under_review = [r for r in rows if r["status"] == "under_review"]
    fallback = [r for r in rows if r.get("fallback_used")]

    confidences = [r["confidence"] for r in rows if r.get("confidence") is not None]
    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else None

    # Signal agreement: LLM band vs stylometric band on the same submission.
    agree = 0
    comparable = 0
    for r in rows:
        lb, sb = _band(r.get("llm_score")), _band(r.get("stylometric_score"))
        if lb is not None and sb is not None:
            comparable += 1
            if lb == sb:
                agree += 1
    agreement_rate = round(agree / comparable, 4) if comparable else None

    try:
        verified_humans = _verified_human_count()
    except sqlite3.Error:
        verified_humans = 0

    def pct(n: int) -> float:
        return round(100 * n / total, 1) if total else 0.0

    return {
        "total_submissions": total,
        "detection_patterns": {
            "likely_ai": attribution_counts.get(config.ATTRIBUTION_LIKELY_AI, 0),
            "uncertain": attribution_counts.get(config.ATTRIBUTION_UNCERTAIN, 0),
            "likely_human": attribution_counts.get(config.ATTRIBUTION_LIKELY_HUMAN, 0),
            "likely_ai_pct": pct(attribution_counts.get(config.ATTRIBUTION_LIKELY_AI, 0)),
            "uncertain_pct": pct(attribution_counts.get(config.ATTRIBUTION_UNCERTAIN, 0)),
            "likely_human_pct": pct(attribution_counts.get(config.ATTRIBUTION_LIKELY_HUMAN, 0)),
        },
        "average_confidence": avg_conf,
        "appeals": {
            "total_appeals": len(appeals),
            "appeal_rate": round(len(appeals) / total, 4) if total else 0.0,
            "under_review_queue": len(under_review),
        },
        "signal_agreement_rate": agreement_rate,
        "fallback_usage_rate": round(len(fallback) / total, 4) if total else 0.0,
        "verified_human_creators": verified_humans,
    }


def _verified_human_count() -> int:
    conn = sqlite3.connect(config.DB_PATH)
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM certificates WHERE status = 'verified'"
        )
        return int(cur.fetchone()[0])
    finally:
        conn.close()


def render_html() -> str:
    """Render the metrics as a minimal standalone HTML dashboard."""
    m = compute()
    dp = m["detection_patterns"]
    ap = m["appeals"]
    agree = m["signal_agreement_rate"]
    agree_str = f"{round(agree * 100)}%" if agree is not None else "n/a"
    avg_conf = m["average_confidence"]
    avg_str = f"{round(avg_conf * 100)}%" if avg_conf is not None else "n/a"

    def bar(count: int, total: int) -> str:
        width = round(100 * count / total) if total else 0
        return f'<div class="bar"><span style="width:{width}%"></span></div>'

    total = m["total_submissions"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Provenance Guard — Analytics</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 720px;
         margin: 2rem auto; padding: 0 1rem; color: #1c2330; background: #f7f9fc; }}
  h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1rem; margin-top: 1.6rem; color:#3a4a63; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr)); gap:.8rem; }}
  .card {{ background:#fff; border:1px solid #e2e8f0; border-radius:10px; padding:.9rem 1rem; }}
  .card .n {{ font-size:1.6rem; font-weight:700; }} .card .l {{ color:#64748b; font-size:.8rem; }}
  .row {{ display:flex; align-items:center; gap:.6rem; margin:.3rem 0; }}
  .row .k {{ width:130px; font-size:.85rem; }} .bar {{ flex:1; background:#eef2f7; border-radius:6px; height:14px; }}
  .bar span {{ display:block; height:100%; border-radius:6px; background:#5b8def; }}
  .muted {{ color:#94a3b8; font-size:.8rem; }}
</style></head><body>
<h1>🛡️ Provenance Guard — Analytics</h1>
<p class="muted">Aggregated from {total} logged submission(s).</p>
<div class="cards">
  <div class="card"><div class="n">{total}</div><div class="l">Submissions</div></div>
  <div class="card"><div class="n">{ap['total_appeals']}</div><div class="l">Appeals ({round(ap['appeal_rate']*100)}%)</div></div>
  <div class="card"><div class="n">{ap['under_review_queue']}</div><div class="l">In review queue</div></div>
  <div class="card"><div class="n">{agree_str}</div><div class="l">Signal agreement</div></div>
  <div class="card"><div class="n">{avg_str}</div><div class="l">Avg AI-likelihood</div></div>
  <div class="card"><div class="n">{m['verified_human_creators']}</div><div class="l">Verified humans</div></div>
</div>
<h2>Detection patterns</h2>
<div class="row"><div class="k">Likely AI ({dp['likely_ai']})</div>{bar(dp['likely_ai'], total)}</div>
<div class="row"><div class="k">Uncertain ({dp['uncertain']})</div>{bar(dp['uncertain'], total)}</div>
<div class="row"><div class="k">Likely human ({dp['likely_human']})</div>{bar(dp['likely_human'], total)}</div>
<p class="muted">Fallback-classifier usage: {round(m['fallback_usage_rate']*100)}% of submissions.</p>
</body></html>"""
