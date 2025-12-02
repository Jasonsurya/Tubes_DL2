# api/index.py

import os
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
from PIL import Image
import io

# Import logika dari file helper yang SEJAJAR (di folder api/)
from .auto_crop import crop_face
from .interface_model import FaceRecognizer

app = Flask(__name__)

# Load model + Excel sekali di startup
recognizer = FaceRecognizer()


# ==========================
# CORS (Wajib untuk API agar diakses dari frontend)
# ==========================

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ==========================
# ROUTE PRESENSI (API)
# ==========================

@app.route("/api/presensi", methods=["POST", "OPTIONS"])
def presensi():
    if request.method == "OPTIONS":
        return ("", 204)

    if "image" not in request.files:
        return jsonify({"success": False, "message": "Field 'image' tidak ditemukan"}), 400

    file = request.files["image"]

    try:
        # Baca file ke PIL.Image
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # 1. Crop wajah
        cropped_face = crop_face(pil_img)
        if cropped_face is None:
            return jsonify({
                "success": False,
                "message": "Wajah tidak terdeteksi."
            }), 200

        # 2. Prediksi NIM
        pred = recognizer.predict_pil(cropped_face, topk=3)
        main = pred["main"]

        # 3. Timestamp
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return jsonify({
            "success": True,
            "nim": main["nim"],
            "nama": main["nama"],
            "prob": main["prob"],
            "timestamp": ts,
            "topk": pred["topk"]
        })

    except Exception as e:
        print("Error presensi:", e)
        return jsonify({
            "success": False,
            "message": f"Terjadi error: {str(e)}"
        }), 500

# ==========================
# ROUTE UTAMA (Diperlukan oleh Vercel)
# ==========================

# Vercel akan menjalankan file ini, jadi kita tidak perlu if __name__ == "__main__":
# Vercel akan otomatis melayani frontend/index.html berkat vercel.json