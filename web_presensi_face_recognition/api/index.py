# api/index.py

import os
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
from PIL import Image
import io
import torch 
from .auto_crop import crop_face
from .interface_model import FaceRecognizer

# --- Konfigurasi Path Absolut ---
# CURRENT_FILE_DIR adalah 'api/'
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR adalah folder di atas 'api/'
ROOT_DIR = os.path.realpath(os.path.join(CURRENT_FILE_DIR, '..'))

# --- Inisialisasi Flask & Model ---
app = Flask(__name__)
recognizer = FaceRecognizer()


# ==========================
# CORS (Wajib untuk API di Vercel)
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

        # 1. Crop wajah (auto_crop.py)
        cropped_face = crop_face(pil_img)
        if cropped_face is None:
            return jsonify({
                "success": False,
                "message": "Wajah tidak terdeteksi."
            }), 200

        # 2. Prediksi NIM (interface_model.py)
        pred = recognizer.predict_pil(cropped_face, topk=1)
        main = pred["main"]
        
        # Logika Keputusan (Contoh: Konfirmasi Hadir jika prob > 0.70)
        confidence_threshold = 0.70
        
        if float(main['prob']) >= confidence_threshold:
            status_presensi = "HADIR"
        else:
            status_presensi = "TOLAK - Yakinan Rendah"


        # 3. Timestamp
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return jsonify({
            "success": True,
            "nim": main["nim"],
            "nama": main["nama"],
            "prob": main["prob"],
            "timestamp": ts,
            "status": status_presensi
        })

    except Exception as e:
        print("Error presensi:", e)
        return jsonify({
            "success": False,
            "message": f"Terjadi error: {str(e)}"
        }), 500


# ==========================
# ROUTE UTAMA (Vercel akan mengabaikan ini dan memakai vercel.json)
# ==========================

@app.route("/", methods=["GET"])
def serve_root():
    return "Backend is running. Access frontend via Vercel root URL."