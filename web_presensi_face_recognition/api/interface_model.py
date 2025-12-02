# api/interface_model.py

import os
import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model 
import pandas as pd
import numpy as np

# --- CONFIG ---
DEVICE = torch.device("cpu") # Vercel free tier adalah CPU
IMAGE_SIZE = 224

# Path: File berada di direktori yang sama (api/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'EfficientNet_B1_face_recognition_best.pth')
# ASUMSI: Nama file CSV adalah Nama_NIM_sorted_AZ.csv (sesuai file Anda)
EXCEL_PATH = os.path.join(CURRENT_DIR, 'Nama_NIM_sorted_AZ.xlsx - Sheet1.csv') 


class FaceRecognizer:
    def __init__(self):
        # 1. Load Data (NIM/Name Mapping) menggunakan Pandas
        self.df = pd.read_csv(EXCEL_PATH)
        self.df['NIM'] = self.df['NIM'].astype(str)
        # Membuat mapping NIM ke NAMA
        self.data_map = self.df.set_index('NIM').to_dict('index') 
        self.nim_list = list(self.data_map.keys())
        self.num_classes = len(self.nim_list)

        # 2. Load Model Architecture & Weights
        self.model = self._load_model()
        
        # Buat mapping terbalik (index model ke NIM)
        self.idx_to_nim = {i: nim for i, nim in enumerate(self.nim_list)}

        # 3. Define Transformations
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def _load_model(self):
        # Bangun Model (EfficientNet B1)
        model = create_model('efficientnet_b1', pretrained=False, num_classes=0)
        num_ftrs = model.num_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        # Load Bobot (dengan robust handling strict=False)
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False) 
        model.to(DEVICE).eval()
        return model


    def predict_pil(self, pil_img, topk=1):
        """Menjalankan prediksi pada single cropped PIL image."""
        
        input_tensor = self.transform(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Ambil top-k results
        top_probs, top_indices = torch.topk(probabilities, topk)
        
        # Format results
        results = []
        for i in range(topk):
            prob = top_probs[0][i].item()
            idx = top_indices[0][i].item()
            nim = self.idx_to_nim[idx]
            
            results.append({
                "nim": nim,
                "nama": self.data_map[nim]['Nama'], 
                "prob": f"{prob:.4f}"
            })
            
        return {
            "main": results[0], # Prediksi teratas
            "topk": results    # List prediksi top-k
        }