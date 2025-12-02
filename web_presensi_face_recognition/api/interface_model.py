# backend/interface_model.py

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd


# ==========================
# KONFIGURASI PATH
# ==========================

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "models", "efficientnet_b1.pth")
EXCEL_PATH = os.path.join(BASE_DIR, "Nama_NIM_sorted_AZ.xlsx")

INPUT_SIZE = 224
DEVICE = "cpu" # "cuda" if torch.cuda.is_available() else "cpu" # Gunakan CPU secara default


# ==========================
# TRANSFORM INFERENCE
# ==========================

def get_inference_transform(input_size=INPUT_SIZE):
    """
    Transform untuk inference.
    Harus sama dengan transform validasi / testing saat training.
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ==========================
# LOAD MODEL
# ==========================

def load_model_and_classes(model_path=MODEL_PATH, device=DEVICE):
    """
    Load EfficientNet-B1 dan class_names dari checkpoint.
    Asumsi: checkpoint berupa dict dengan key:
      - "model_state_dict"
      - "class_names"
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" not in checkpoint or "class_names" not in checkpoint:
        raise ValueError(
            "Checkpoint tidak berisi 'model_state_dict' dan 'class_names'. "
            "Pastikan file .pth dari script training yang benar."
        )

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    # Buat arsitektur EfficientNet-B1 yang sama
    base_model = models.efficientnet_b1(weights=None)
    in_features = base_model.classifier[1].in_features
    base_model.classifier[1] = nn.Linear(in_features, num_classes)

    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.to(device)
    base_model.eval()

    return base_model, class_names


# ==========================
# LOAD MAPPING NIM -> NAMA
# ==========================

def load_nim_name_mapping(excel_path=EXCEL_PATH):
    """
    Excel diharapkan punya kolom:
      - 'NIM'
      - 'Nama'
    Return: dict {nim_str: nama_str}
    """
    if not os.path.exists(excel_path):
        print(f"[PERINGATAN] File Excel tidak ditemukan: {excel_path}")
        return {}

    df = pd.read_excel(excel_path)

    # Cari nama kolom yang kira-kira benar
    cols = [c.lower() for c in df.columns]

    try:
        nim_col = next(c for c in df.columns if "nim" in c.lower())
        nama_col = next(c for c in df.columns if "nama" in c.lower())
    except StopIteration:
        raise ValueError(
            f"Kolom 'NIM' / 'Nama' tidak ditemukan di Excel. Kolom yang ada: {df.columns}"
        )

    mapping = {}
    for _, row in df.iterrows():
        nim_val = str(row[nim_col]).strip()
        nama_val = str(row[nama_col]).strip()
        mapping[nim_val] = nama_val

    return mapping


# ==========================
# WRAPPER UNTUK BACKEND
# ==========================

class FaceRecognizer:
    """
    Wrapper supaya backend gampang:
    - Load model + class_names + Excel sekali di __init__
    - Punya method predict_pil() untuk gambar PIL (wajah sudah di-crop)
    """

    def __init__(self,
                 model_path: str = MODEL_PATH,
                 excel_path: str = EXCEL_PATH,
                 device: str = DEVICE):
        self.device = device
        print(f"[INFO] Loading model dari: {model_path}")
        self.model, self.class_names = load_model_and_classes(model_path, device)

        print(f"[INFO] Loading mapping NIM-Nama dari: {excel_path}")
        self.nim_to_name = load_nim_name_mapping(excel_path)

        self.transform = get_inference_transform(INPUT_SIZE)

    def predict_pil(self, pil_image: Image.Image, topk: int = 3):
        """
        Prediksi NIM dari gambar wajah (PIL) yang sudah di-crop.
        Return dict:
          {
            "main": { "nim": ..., "nama": ..., "prob": ... },
            "topk": [ {nim, nama, prob}, ... ]
          }
        """
        img = pil_image.convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(
                probs,
                k=min(topk, probs.shape[1]),
                dim=1
            )

        top_probs = top_probs.cpu().numpy().flatten()
        top_indices = top_indices.cpu().numpy().flatten()

        results = []
        for prob, idx in zip(top_probs, top_indices):
            nim = self.class_names[idx]
            nama = self.nim_to_name.get(nim, "(Nama tidak ditemukan di Excel)")
            results.append({
                "nim": nim,
                "nama": nama,
                "prob": float(prob)
            })

        return {
            "main": results[0],
            "topk": results
        }
