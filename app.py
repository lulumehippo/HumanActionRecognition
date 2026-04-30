"""
app.py
------
Flask web server for the Human Action Recognition demo.

Usage
-----
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import uuid
import sys
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from predictor import predict, CLASSES       # noqa: E402

# ── App config ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB upload limit

UPLOAD_DIR  = ROOT / "uploads"
MODEL_PATH  = str(ROOT / "model" / "action_model.keras")
ALLOWED_EXT = {".avi", ".mp4", ".mov", ".mkv"}

UPLOAD_DIR.mkdir(exist_ok=True)


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", classes=CLASSES)


@app.route("/predict", methods=["POST"])
def predict_route():
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."}), 400

    f = request.files["video"]

    if f.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400

    if not _allowed(f.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported format. Allowed: {', '.join(ALLOWED_EXT)}"
        }), 400

    # Save to temp path
    tmp_name = f"{uuid.uuid4().hex}{Path(f.filename).suffix.lower()}"
    tmp_path = UPLOAD_DIR / tmp_name

    try:
        f.save(str(tmp_path))
        result = predict(str(tmp_path), model_path=MODEL_PATH)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()            # delete temp file immediately

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "online", "classes": CLASSES})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STARK INDUSTRIES — Human Action Recognition System")
    print(f"  Server  : http://127.0.0.1:5000")
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Classes : {', '.join(CLASSES)}")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
