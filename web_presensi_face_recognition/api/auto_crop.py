# api/auto_crop.py

import numpy as np
from facenet_pytorch import MTCNN 
from PIL import Image
import torch

# Inisialisasi detektor MTCNN sekali (Global)
# MTCNN harus diinisiasi di luar fungsi untuk menghindari lag
detector = MTCNN(keep_all=False, device='cpu', thresholds=[0.6, 0.7, 0.7]) 

def crop_face(pil_img):
    """Mendeteksi wajah utama dalam PIL image dan mengembalikan cropped image."""
    
    # 1. Deteksi wajah 
    boxes, _ = detector.detect(pil_img)
    
    if boxes is None or len(boxes) == 0:
        return None 

    # 2. Ambil kotak pertama 
    box = boxes[0]
    
    # Koordinat dalam format (x1, y1, x2, y2)
    x1, y1, x2, y2 = [int(c) for c in box]
    
    # Tambah margin kecil
    margin = 10
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(pil_img.width, x2 + margin)
    y2 = min(pil_img.height, y2 + margin)

    # 3. Potong dan kembalikan sebagai PIL image
    cropped_img = pil_img.crop((x1, y1, x2, y2))
    
    return cropped_img