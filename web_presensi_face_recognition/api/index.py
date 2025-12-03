# api/index.py
import os
import io
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import traceback

# universal imports for local & package
try:
    from .auto_crop import crop_face
    from .interface_model import FaceRecognizer
except Exception:
    from auto_crop import crop_face
    from interface_model import FaceRecognizer

# APP
app = Flask(__name__, static_folder="../frontend", static_url_path="/")
CORS(app)

# limit upload size (optional, set to a few MB)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

# Initialize model (lazy safe)
try:
    recognizer = FaceRecognizer()
    MODEL_LOADED = True
except Exception as e:
    print("[WARN] Gagal load model/mapper:", e)
    traceback.print_exc()
    recognizer = None
    MODEL_LOADED = False


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/presensi", methods=["POST", "OPTIONS"])
def presensi():
    if request.method == "OPTIONS":
        return ("", 204)

    if "image" not in request.files:
        return jsonify({"success": False, "message": "Field 'image' tidak ditemukan"}), 400

    file = request.files["image"]

    try:
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # crop wajah
        cropped = crop_face(pil_img)
        if cropped is None:
            return jsonify({"success": False, "message": "Wajah tidak terdeteksi."}), 200

        if not MODEL_LOADED or recognizer is None:
            return jsonify({"success": False, "message": "Model belum siap di server."}), 500

        pred = recognizer.predict_pil(cropped, topk=3)
        main = pred.get("main", {})

        confidence_threshold = 0.70
        prob = float(main.get("prob", 0.0))
        status_presensi = "HADIR" if prob >= confidence_threshold else "TOLAK - Yakinan Rendah"

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return jsonify({
            "success": True,
            "nim": main.get("nim"),
            "nama": main.get("nama"),
            "prob": prob,
            "timestamp": ts,
            "status": status_presensi,
            "topk": pred.get("topk", [])
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Terjadi error: {str(e)}"}), 500


# serve frontend index locally
@app.route("/", methods=["GET"])
def serve_frontend_local():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frontend_dir = os.path.join(root_dir, 'frontend')
    return send_from_directory(frontend_dir, "index.html")


if __name__ == "__main__":
    print("Running Flask dev server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
