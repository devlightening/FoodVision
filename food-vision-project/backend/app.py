from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from predict import predict_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = PROJECT_ROOT / "backend" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": 'Missing form-data file key "image".'}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    filename = Path(file.filename).name
    saved_path = UPLOADS_DIR / filename
    file.save(saved_path)

    try:
        result = predict_image(saved_path)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            saved_path.unlink(missing_ok=True)
        except Exception:
            pass

    return jsonify(
        {
            "class_name": result.class_name,
            "display_name_tr": result.display_name_tr,
            "confidence": round(result.confidence, 2),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

