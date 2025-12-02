# backend/auto_crop.py

import cv2
import numpy as np
from PIL import Image

# Ukuran output wajah (harus sama dengan training: 224 x 224)
TARGET_SIZE = (224, 224)  # (width, height)

# Margin di sekitar wajah (misalnya 0.3 = 30% dari lebar/tinggi bounding box)
FACE_MARGIN_RATIO = 0.3

# Jika tidak ada wajah terdeteksi,
# apakah pakai center crop sebagai fallback?
USE_CENTER_FALLBACK = True

# Load Haar Cascade sekali saja
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError("Gagal memuat Haar Cascade untuk deteksi wajah.")


def _crop_with_margin(img_bgr, x, y, w, h, margin_ratio=FACE_MARGIN_RATIO):
    """
    img_bgr: numpy array BGR
    x, y, w, h: bounding box dari detectMultiScale
    """
    h_img, w_img = img_bgr.shape[:2]

    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w_img, x + w + mx)
    y2 = min(h_img, y + h + my)

    return img_bgr[y1:y2, x1:x2]


def _center_crop_to_target(img_bgr, target_size=TARGET_SIZE):
    """
    Jika tidak ada wajah terdeteksi, ambil crop di tengah.
    """
    h, w = img_bgr.shape[:2]
    tw, th = target_size

    # skala supaya sisi terkecil >= target
    scale = max(tw / w, th / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img_bgr, (new_w, new_h))

    # crop tengah
    x1 = (new_w - tw) // 2
    y1 = (new_h - th) // 2
    x2 = x1 + tw
    y2 = y1 + th

    return resized[y1:y2, x1:x2]


def crop_face(pil_image: Image.Image, use_center_fallback: bool = USE_CENTER_FALLBACK):
    """
    Terima gambar PIL (RGB), deteksi wajah, crop dengan margin, lalu resize ke TARGET_SIZE.
    Return:
      - PIL.Image (RGB) hasil crop & resize, jika berhasil
      - None jika gagal dan fallback dimatikan
    """

    # PIL (RGB) -> OpenCV (BGR)
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) > 0:
        # Pilih wajah terbesar
        faces_sorted = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = faces_sorted[0]

        face_crop = _crop_with_margin(img_bgr, x, y, w, h, margin_ratio=FACE_MARGIN_RATIO)
        face_resized = cv2.resize(face_crop, TARGET_SIZE)

        # BGR -> RGB -> PIL
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        return Image.fromarray(face_rgb)

    # Tidak ada wajah terdeteksi
    if use_center_fallback:
        center_cropped = _center_crop_to_target(img_bgr, TARGET_SIZE)
        face_rgb = cv2.cvtColor(center_cropped, cv2.COLOR_BGR2RGB)
        return Image.fromarray(face_rgb)

    return None
