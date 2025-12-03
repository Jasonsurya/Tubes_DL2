# api/interface_model.py
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "efficientnet_b1.pth")
EXCEL_PATH = os.path.join(BASE_DIR, "Nama_NIM_sorted_AZ.xlsx")
INPUT_SIZE = 224
DEVICE = "cpu"  # change to "cuda" if available

def get_inference_transform(input_size=INPUT_SIZE):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def load_model_and_classes(model_path=MODEL_PATH, device=DEVICE):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" not in checkpoint or "class_names" not in checkpoint:
        raise ValueError("Checkpoint harus berisi 'model_state_dict' dan 'class_names'")

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    base_model = models.efficientnet_b1(weights=None)
    in_features = base_model.classifier[1].in_features
    base_model.classifier[1] = nn.Linear(in_features, num_classes)

    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.to(device)
    base_model.eval()

    return base_model, class_names

def load_nim_name_mapping(excel_path=EXCEL_PATH):
    if not os.path.exists(excel_path):
        print(f"[PERINGATAN] Excel tidak ditemukan di: {excel_path}")
        return {}

    df = pd.read_excel(excel_path)
    try:
        nim_col = next(c for c in df.columns if "nim" in c.lower())
        nama_col = next(c for c in df.columns if "nama" in c.lower())
    except StopIteration:
        raise ValueError(f"Kolom 'NIM' / 'Nama' tidak ditemukan. Kolom tersedia: {df.columns}")

    mapping = {}
    for _, row in df.iterrows():
        nim_val = str(row[nim_col]).strip()
        nama_val = str(row[nama_col]).strip()
        mapping[nim_val] = nama_val

    return mapping

class FaceRecognizer:
    def __init__(self, model_path=MODEL_PATH, excel_path=EXCEL_PATH, device=DEVICE):
        self.device = device
        self.model, self.class_names = load_model_and_classes(model_path, device)
        self.nim_to_name = load_nim_name_mapping(excel_path)
        self.transform = get_inference_transform(INPUT_SIZE)

    def predict_pil(self, pil_image: Image.Image, topk: int = 3):
        img = pil_image.convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            k = min(topk, probs.shape[1])
            top_probs, top_indices = torch.topk(probs, k=k, dim=1)

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
