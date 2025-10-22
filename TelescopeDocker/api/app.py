from flask import Flask, request, jsonify, Response
from werkzeug.utils import secure_filename
import os
import time
import json
import threading
from flask import Response
from telescope_code.trainer import train_model_from_text
from telescope_code.predictor import load_model, predict_df, build_features_from_instances

app = Flask(__name__)

MODEL_ROOT = os.environ.get("MODEL_ROOT", "models")
os.makedirs(MODEL_ROOT, exist_ok=True)

_ACTIVE = {"model_dir": None, "models": None, "meta": None}
_SESSION_STATE = {}  # session_id -> {"prev_ra_offset": float, "prev_dec_offset": float}
_STATE_LOCK = threading.Lock()

# ---------- helpers for formatting ----------

def _wants_text() -> bool:
    """Return True if the client asked for text/plain (Accept header) or ?human=1."""
    if request.args.get("human") in {"1", "true", "yes"}:
        return True
    accept = (request.headers.get("Accept") or "").lower()
    return "text/plain" in accept

def _pretty_json(payload: dict) -> Response:
    """Pretty JSON so it's readable even without jq."""
    return Response(json.dumps(payload, indent=2, sort_keys=False) + "\n",
                    mimetype="application/json")

def _train_text(model_dir: str, meta: dict) -> Response:
    """Line-by-line summary for /train."""
    feats = meta.get("feature_cols", [])
    tgts  = meta.get("target_cols", [])
    rows  = meta.get("training_rows", "n/a")
    lines = [
        "Training complete ✅",
        f"Model dir : {model_dir}",
        f"Rows used : {rows}",
        f"Targets   : {', '.join(tgts) if tgts else '(none)'}",
        f"Features  : {len(feats)} total",
        "  " + ", ".join(feats),
        "",
        "Predict accepts either:",
        "  • DATE-OBS, LST, TPT-RA, TPT-DEC  (strings; server parses)",
        "  • years,months,days,hours,minutes,seconds,lst_hours,tpt_ra_deg,tpt_dec_deg (numeric)",
        "",
        "Tip: For JSON, omit Accept header or use `| jq` to pretty-print."
    ]
    return Response("\n".join(lines) + "\n", mimetype="text/plain")

def _predict_text(model_dir: str, meta: dict, parsed_info: dict,
                  preds: list[dict], session_mem: dict) -> Response:
    """Line-by-line summary for /predict."""
    lines = [
        "Prediction ✅",
        f"Model dir : {model_dir}",
        f"Parsed    : rows={parsed_info.get('rows')} "
        f"(strings={parsed_info.get('parsed_from_strings')}, numeric={parsed_info.get('accepted_numeric')})",
        "",
        "Results:"
    ]
    for i, p in enumerate(preds, 1):
        lines.append(f"  Row {i:>3}: ra_offset={p['ra_offset']:.6f}  dec_offset={p['dec_offset']:.6f}")
    lines += [
        "",
        "Session memory (used as prev_* for next call):",
        f"  prev_ra_offset={session_mem.get('prev_ra_offset', 0.0):.6f}",
        f"  prev_dec_offset={session_mem.get('prev_dec_offset', 0.0):.6f}"
    ]
    return Response("\n".join(lines) + "\n", mimetype="text/plain")

# ---------- routes ----------

@app.get("/health")
def health():
    loaded = _ACTIVE["models"] is not None
    payload = {
        "status": "ok",
        "model_loaded": loaded,
        "model_dir": _ACTIVE["model_dir"],
        "meta": _ACTIVE["meta"] if loaded else None,
        "notes": [
            "Train with columns: DATE-OBS, LST, TPT-RA, TPT-DEC, CRVAL1, CRVAL2 (WCS RA/DEC in degrees).",
            "Predict accepts either raw strings (DATE-OBS/LST/TPT-RA/TPT-DEC) or numeric features.",
            "Use Accept: text/plain or ?human=1 for readable output."
        ],
    }
    return _pretty_json(payload) if not _wants_text() else Response(
        "API OK\n" +
        (f"Active model: {_ACTIVE['model_dir']}\n" if loaded else "Active model: (none)\n"),
        mimetype="text/plain"
    )

@app.post("/train")
def train():
    if "file" not in request.files:
        err = {"error": "No file provided. Use multipart form field 'file'."}
        return _pretty_json(err) if not _wants_text() else Response(err["error"] + "\n", 400, mimetype="text/plain")

    f = request.files["file"]
    # read everything as text; trainer finds the header line automatically
    raw_text = f.read().decode("utf-8", errors="replace")

    model_root = request.form.get("model_root", MODEL_ROOT)
    tag = request.form.get("tag", None)

    model_dir, meta = train_model_from_text(raw_text, model_root=model_root, tag=tag)

    # load freshly trained model
    models, meta_loaded = load_model(model_dir)
    with _STATE_LOCK:
        _ACTIVE.update({"model_dir": model_dir, "models": models, "meta": meta_loaded})
        _SESSION_STATE.clear()

    payload = {"status": "trained", "model_dir": model_dir, "meta": meta}
    return _train_text(model_dir, meta) if _wants_text() else _pretty_json(payload)

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        err = {"error": "session_id is required"}
        return _pretty_json(err) if not _wants_text() else Response(err["error"] + "\n", 400, mimetype="text/plain")

    reset = bool(data.get("reset", False))
    instances = data.get("instances") or []
    if not isinstance(instances, list) or not instances:
        err = {"error": "Provide a non-empty 'instances' list"}
        return _pretty_json(err) if not _wants_text() else Response(err["error"] + "\n", 400, mimetype="text/plain")

    # allow overriding the active model
    model_dir = data.get("model_dir")
    if model_dir and model_dir != _ACTIVE["model_dir"]:
        models, meta = load_model(model_dir)
        if models is None:
            err = {"error": f"Could not load model_dir={model_dir}"}
            return _pretty_json(err) if not _wants_text() else Response(err["error"] + "\n", 400, mimetype="text/plain")
        with _STATE_LOCK:
            _ACTIVE.update({"model_dir": model_dir, "models": models, "meta": meta})
            _SESSION_STATE.pop(session_id, None)
    elif _ACTIVE["models"] is None:
        err = {"error": "No model loaded. Train via /train first."}
        return _pretty_json(err) if not _wants_text() else Response(err["error"] + "\n", 400, mimetype="text/plain")

    # normalize inputs → numeric feature DF
    feat_df, parse_info = build_features_from_instances(instances)

    # session memory
    with _STATE_LOCK:
        if reset or session_id not in _SESSION_STATE:
            _SESSION_STATE[session_id] = {"prev_ra_offset": 0.0, "prev_dec_offset": 0.0}

        preds, new_prev = predict_df(
            feat_df,
            session_state=_SESSION_STATE[session_id],
            models=_ACTIVE["models"],
            meta=_ACTIVE["meta"]
        )
        _SESSION_STATE[session_id] = new_prev

    payload = {
        "model_dir": _ACTIVE["model_dir"],
        "meta": _ACTIVE["meta"],
        "parsed": parse_info,
        "predictions": preds,
        "session_memory": _SESSION_STATE[session_id]
    }
    return _predict_text(_ACTIVE["model_dir"], _ACTIVE["meta"], parse_info, preds, _SESSION_STATE[session_id]) \
        if _wants_text() else _pretty_json(payload)

