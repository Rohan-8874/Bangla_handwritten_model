# ============================================================
# 19. Load Best Model & Test on Your Image
# ============================================================

import os, json, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- Config ---
MODEL_PATH = "best_model.pth"  # Change to "final_model.pth" to use final model
WORKSPACE = "/home/rohan/Softograph/bangla_dataset_1_download"

# Try multiple locations for the model file
for candidate in [
    os.path.join(WORKSPACE, MODEL_PATH),
    os.path.join("/kaggle/working", MODEL_PATH),
    MODEL_PATH,
]:
    if os.path.exists(candidate):
        MODEL_PATH = candidate
        break

print(f"Loading model from: {MODEL_PATH}")
print(f"Model size: {os.path.getsize(MODEL_PATH) / 1024**2:.1f} MB")

# --- Load checkpoint ---
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
saved_char_to_idx = checkpoint['char_to_idx']
saved_idx_to_char = checkpoint['idx_to_char']
saved_num_classes = checkpoint['num_classes']

print(f"Model from epoch: {checkpoint.get('epoch', '?')}")
print(f"Val CER: {checkpoint.get('val_cer', '?')}")
print(f"Val WER: {checkpoint.get('val_wer', '?')}")
print(f"Vocabulary: {saved_num_classes} classes ({saved_num_classes - 1} chars + blank)")

# --- Rebuild model ---
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)

class CRNN(nn.Module):
    def __init__(self, num_classes, img_height=64, hidden_size=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            nn.Dropout(0.3),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes),
        )
    def forward(self, x):
        conv = self.cnn(x)
        conv = self.adaptive_pool(conv).squeeze(2).permute(0, 2, 1)
        output = self.rnn(conv).permute(1, 0, 2)
        return F.log_softmax(output, dim=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
infer_model = CRNN(num_classes=saved_num_classes, img_height=64, hidden_size=256).to(device)
infer_model.load_state_dict(checkpoint['model_state_dict'])
infer_model.eval()
print(f"\nModel loaded on {device} — ready for inference!")
print("="*50)