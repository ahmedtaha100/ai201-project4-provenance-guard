"""Provenance Guard — Flask backend.

A content-attribution service a creative-sharing platform can plug into:
classify submitted text as AI- vs human-written, score calibrated confidence,
return a plain-language transparency label, and handle creator appeals — with
rate limiting and a structured audit log.

Routes
------
POST /submit                 Classify content; returns content_id + attribution +
                             confidence + label. (rate-limited)
POST /appeal                 Contest a classification; sets status=under_review.
GET  /log                    Recent audit-log entries (JSON).
POST /certificate/verify     Earn a "Verified Human" certificate (stretch).
GET  /certificate/<creator>  Fetch a creator's certificate (stretch).
GET  /analytics              Detection/appeal analytics as JSON (stretch).
GET  /dashboard              Same metrics as an HTML dashboard (stretch).
GET  /reviewer/queue         Appeals awaiting human review.
GET  /health                 Liveness probe.
GET  /                       API map.
"""

from __future__ import annotations

import logging
import uuid

# Load .env BEFORE importing config so GROQ_API_KEY / overrides are visible at
# config import time (config reads os.getenv at module load).
from dotenv import load_dotenv

load_dotenv()

import config  # noqa: E402
import analytics  # noqa: E402
import audit  # noqa: E402
import certificate  # noqa: E402
import pipeline  # noqa: E402
from appeals import AppealError, submit_appeal  # noqa: E402
from certificate import CertificateError  # noqa: E402
from labels import make_label  # noqa: E402

from flask import Flask, Response, jsonify, render_template, request  # noqa: E402
from flask_limiter import Limiter  # noqa: E402
from flask_limiter.util import get_remote_address  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("provenance_guard")

app = Flask(__name__)

# Rate limiting — in-memory storage is the documented local-dev setup. Limits
# are applied per-route (see /submit); no global default so reads stay open.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri=config.RATELIMIT_STORAGE_URI,
)

# Persistence is ready before the first request.
audit.init_db()
certificate.init_db()

if config.CERT_SECRET_EPHEMERAL:
    logger.warning(
        "PROVENANCE_CERT_SECRET is not set — using an ephemeral random signing "
        "secret. Verified-Human certificates will not survive a restart; set "
        "PROVENANCE_CERT_SECRET (from a secret manager) in production."
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class BadRequest(Exception):
    """Malformed /submit payload (-> HTTP 400)."""


def _validate_submission(data: object) -> tuple[str, str, str]:
    """Validate a /submit JSON body. Returns (text, creator_id, content_type)."""
    if not isinstance(data, dict):
        raise BadRequest("Request body must be a JSON object.")
    text = data.get("text")
    creator_id = data.get("creator_id")
    content_type = data.get("content_type", config.CONTENT_TYPE_TEXT)

    if not isinstance(text, str) or not text.strip():
        raise BadRequest("`text` is required and must be a non-empty string.")
    if len(text) > config.MAX_TEXT_CHARS:
        raise BadRequest(f"`text` exceeds the {config.MAX_TEXT_CHARS}-character limit.")
    if not isinstance(creator_id, str) or not creator_id.strip():
        raise BadRequest("`creator_id` is required and must be a non-empty string.")
    if content_type not in config.SUPPORTED_CONTENT_TYPES:
        raise BadRequest(
            f"`content_type` must be one of {list(config.SUPPORTED_CONTENT_TYPES)}."
        )
    return text, creator_id.strip(), content_type


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/submit", methods=["POST"])
@limiter.limit(config.RATE_LIMITS)
def submit() -> tuple[Response, int]:
    """Classify a piece of content and log the decision."""
    try:
        text, creator_id, content_type = _validate_submission(request.get_json(silent=True))
    except BadRequest as exc:
        return jsonify({"error": "bad_request", "message": str(exc)}), 400

    result = pipeline.classify(text, content_type=content_type)
    score = result.score
    verified_human = certificate.is_verified_human(creator_id)
    label = make_label(score, verified_human=verified_human)

    content_id = str(uuid.uuid4())
    word_count = sum(1 for _ in text.split())

    audit.record_submission(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "content_type": content_type,
            "text_excerpt": text[:200],
            "word_count": word_count,
            "attribution": score.attribution,
            "confidence": score.confidence,
            "llm_score": result.signal_score(config.SIGNAL_LLM),
            "stylometric_score": result.signal_score(config.SIGNAL_STYLOMETRIC),
            "repetition_score": result.signal_score(config.SIGNAL_REPETITION),
            "signal_detail": result.signal_detail(),
            "fallback_used": result.fallback_used,
            "status": "classified",
        }
    )

    response = {
        "content_id": content_id,
        "creator_id": creator_id,
        "content_type": content_type,
        "attribution": score.attribution,
        "confidence": score.confidence,  # P(AI), 0..1
        "label": label["text"],
        "label_detail": label,
        "signals": {
            "llm": result.signal_score(config.SIGNAL_LLM),
            "stylometric": result.signal_score(config.SIGNAL_STYLOMETRIC),
            "repetition": result.signal_score(config.SIGNAL_REPETITION),
        },
        "signal_detail": result.signal_detail(),
        "scoring": {
            "raw_weighted": score.raw_weighted,
            "disagreement": score.disagreement,
            "weights": score.weights,
            "notes": score.notes,
        },
        "certificate": {"verified_human": verified_human, "badge": label.get("badge")},
        "fallback_used": result.fallback_used,
        "status": "classified",
        "timestamp": audit.utc_now_iso(),
    }
    return jsonify(response), 200


@app.route("/appeal", methods=["POST"])
def appeal() -> tuple[Response, int]:
    """Contest a classification (sets status to under_review)."""
    data = request.get_json(silent=True) or {}
    try:
        confirmation = submit_appeal(
            data.get("content_id", ""), data.get("creator_reasoning", "")
        )
    except AppealError as exc:
        return jsonify({"error": "appeal_error", "message": exc.message}), exc.status_code
    return jsonify(confirmation), 200


@app.route("/log", methods=["GET"])
def get_log() -> tuple[Response, int]:
    """Return recent audit-log entries as JSON."""
    try:
        limit = min(max(int(request.args.get("limit", 20)), 1), 200)
    except (TypeError, ValueError):
        limit = 20
    entries = audit.get_recent(limit)
    return jsonify({"count": len(entries), "entries": entries}), 200


@app.route("/certificate/verify", methods=["POST"])
def certificate_verify() -> tuple[Response, int]:
    """Issue a 'Verified Human' certificate from a writing sample (stretch)."""
    data = request.get_json(silent=True) or {}
    try:
        cert = certificate.request_certificate(
            data.get("creator_id", ""), data.get("sample_text", "")
        )
    except CertificateError as exc:
        return jsonify({"error": "certificate_error", "message": exc.message}), exc.status_code
    return jsonify(cert), 200


@app.route("/certificate/<creator_id>", methods=["GET"])
def certificate_get(creator_id: str) -> tuple[Response, int]:
    """Fetch a creator's certificate, if any (stretch)."""
    cert = certificate.get_certificate(creator_id)
    if not cert:
        return jsonify({"creator_id": creator_id, "verified_human": False}), 404
    return jsonify(cert), 200


@app.route("/analytics", methods=["GET"])
def analytics_json() -> tuple[Response, int]:
    """Detection/appeal analytics as JSON (stretch)."""
    return jsonify(analytics.compute()), 200


@app.route("/dashboard", methods=["GET"])
def dashboard() -> Response:
    """Analytics as an HTML dashboard (stretch)."""
    return Response(analytics.render_html(), mimetype="text/html")


@app.route("/reviewer/queue", methods=["GET"])
def reviewer_queue() -> tuple[Response, int]:
    """Appeals awaiting human review (what a reviewer sees)."""
    queue = audit.get_under_review()
    return jsonify({"count": len(queue), "queue": queue}), 200


@app.route("/health", methods=["GET"])
def health() -> tuple[Response, int]:
    """Liveness probe + active configuration summary."""
    return (
        jsonify(
            {
                "status": "ok",
                "groq_model": config.GROQ_MODEL,
                "groq_key_present": bool(config.GROQ_API_KEY),
                "ensemble_enabled": config.USE_ENSEMBLE,
                "thresholds": {"ai": config.AI_THRESHOLD, "human": config.HUMAN_THRESHOLD},
                "rate_limits": config.RATE_LIMITS,
            }
        ),
        200,
    )


@app.route("/", methods=["GET"])
@app.route("/demo", methods=["GET"])
def home() -> str:
    """Minimal web UI: paste text, see the attribution + confidence + label, appeal."""
    return render_template("index.html")


@app.route("/api", methods=["GET"])
def index() -> tuple[Response, int]:
    """API map."""
    return (
        jsonify(
            {
                "service": "Provenance Guard",
                "description": "Multi-signal AI-vs-human content attribution with "
                "calibrated confidence, transparency labels, and appeals.",
                "endpoints": {
                    "POST /submit": "Classify content {text, creator_id, content_type?}",
                    "POST /appeal": "Contest a decision {content_id, creator_reasoning}",
                    "GET /log": "Recent audit entries (?limit=N)",
                    "POST /certificate/verify": "Earn Verified-Human cert {creator_id, sample_text}",
                    "GET /certificate/<creator_id>": "Fetch a creator's certificate",
                    "GET /analytics": "Detection/appeal analytics (JSON)",
                    "GET /dashboard": "Analytics dashboard (HTML)",
                    "GET /reviewer/queue": "Appeals awaiting review",
                    "GET /health": "Liveness + config",
                },
            }
        ),
        200,
    )


# --------------------------------------------------------------------------- #
# Error handlers — always return JSON
# --------------------------------------------------------------------------- #
@app.errorhandler(404)
def not_found(_e: object) -> tuple[Response, int]:
    return jsonify({"error": "not_found", "message": "No such resource."}), 404


@app.errorhandler(405)
def method_not_allowed(_e: object) -> tuple[Response, int]:
    return jsonify({"error": "method_not_allowed", "message": "Method not allowed."}), 405


@app.errorhandler(429)
def ratelimit_handler(e: object) -> tuple[Response, int]:
    desc = getattr(e, "description", config.RATE_LIMITS)
    return (
        jsonify(
            {
                "error": "rate_limit_exceeded",
                "message": f"Rate limit exceeded ({desc}). Slow down and retry later.",
                "limits": config.RATE_LIMITS,
            }
        ),
        429,
    )


@app.errorhandler(500)
def server_error(_e: object) -> tuple[Response, int]:
    logger.exception("Unhandled server error")
    return jsonify({"error": "server_error", "message": "Internal server error."}), 500


if __name__ == "__main__":
    # Threaded dev server; debug off so tracebacks never leak to clients.
    app.run(host="0.0.0.0", port=5000, debug=False)
