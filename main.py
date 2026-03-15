"""
main.py — Bangla Handwriting Recognition
=========================================
Unified script that:
  1. Loads the best trained CRNN model from checkpoint
  2. Preprocesses a test image (single word-crop or full-page with JSON annotation)
  3. Runs CTC decoding and prints / displays the predicted text
  4. Saves a .txt result file next to the image with the same name
  5. Batch mode: scans all image+JSON pairs, predicts every word, outputs CSV report

Usage:
    python main.py --image <path_to_image>    # Single image mode
    python main.py --batch                     # Batch mode on raw/raw/
    python main.py --batch --raw-dir <dir>     # Custom raw directory
"""

import os
import json
import csv
import glob
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ──────────────────────────────────────────────
# Bengali Font Setup for Matplotlib
# ──────────────────────────────────────────────
_BENGALI_FONT_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
    "/usr/share/fonts/truetype/lohit-bengali/Lohit-Bengali.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]

def _setup_bengali_font() -> fm.FontProperties:
    """Register and return a Bengali-capable FontProperties object."""
    for path in _BENGALI_FONT_PATHS:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            print(f"Bengali font loaded: {path}")
            return prop
    print("WARNING: No Bengali font found. Glyphs may not render correctly.")
    return fm.FontProperties()  # fallback

BENGALI_FONT = _setup_bengali_font()


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
WORKSPACE = "/home/rohan/Softograph/bangla_dataset_1_download"

# Default test image — override via --image CLI argument
DEFAULT_TEST_IMAGE = os.path.join(WORKSPACE, "image_12.png")

IMG_HEIGHT, IMG_WIDTH = 64, 256
HIDDEN_SIZE = 256


# ──────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)


class CRNN(nn.Module):
    def __init__(self, num_classes: int, img_height: int = 64, hidden_size: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),   nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1),nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0),nn.BatchNorm2d(512), nn.ReLU(True),
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


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────
def find_model(model_filename: str = "best_model.pth") -> str:
    """Locate the model checkpoint in common locations."""
    candidates = [
        os.path.join(WORKSPACE, model_filename),
        os.path.join("/kaggle/working", model_filename),
        model_filename,
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Model checkpoint '{model_filename}' not found in: {candidates}"
    )


def load_model(model_path: str, device: torch.device):
    """Load checkpoint and rebuild the CRNN model."""
    print(f"Loading model from : {model_path}")
    print(f"Model size         : {os.path.getsize(model_path) / 1024**2:.1f} MB")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    char_to_idx = checkpoint["char_to_idx"]
    idx_to_char = checkpoint["idx_to_char"]
    num_classes  = checkpoint["num_classes"]

    print(f"Epoch              : {checkpoint.get('epoch', '?')}")
    print(f"Val CER            : {checkpoint.get('val_cer', '?')}")
    print(f"Val WER            : {checkpoint.get('val_wer', '?')}")
    print(f"Vocabulary         : {num_classes} classes ({num_classes - 1} chars + blank)")

    model = CRNN(num_classes=num_classes, img_height=IMG_HEIGHT, hidden_size=HIDDEN_SIZE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    print(f"\nModel loaded on {device} — ready for inference!")
    print("=" * 55)
    return model, char_to_idx, idx_to_char


# ──────────────────────────────────────────────
# Preprocessing & CTC Decoding
# ──────────────────────────────────────────────
def preprocess_word_crop(img: Image.Image,
                          img_height: int = IMG_HEIGHT,
                          img_width: int = IMG_WIDTH) -> torch.Tensor:
    """Resize word image maintaining aspect ratio, pad to fixed size, and normalise."""
    if img.mode != "L":
        img = img.convert("L")
    w, h = img.size
    new_h = img_height
    new_w = max(1, int(w * new_h / h))
    if new_w > img_width:
        new_w = img_width
    img = img.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("L", (img_width, img_height), 255)
    padded.paste(img, (0, 0))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transform(padded).unsqueeze(0)          # (1, 1, H, W)


def ctc_decode(log_probs: torch.Tensor, idx_to_char: dict) -> list[str]:
    """Greedy CTC decoding — collapse repeated chars and remove blanks (index 0)."""
    _, max_idx = torch.max(log_probs, dim=2)       # (T, B)
    max_idx = max_idx.permute(1, 0).cpu().numpy()  # (B, T)
    decoded = []
    for seq in max_idx:
        chars, prev = [], -1
        for idx in seq:
            if idx != prev:
                if idx != 0:
                    chars.append(idx_to_char.get(int(idx), "?"))
                prev = idx
        decoded.append("".join(chars))
    return decoded


def predict_single_image(model: CRNN,
                          img: Image.Image,
                          device: torch.device,
                          idx_to_char: dict) -> str:
    """Return the predicted text string for a single word-crop PIL image."""
    tensor = preprocess_word_crop(img).to(device)
    with torch.no_grad():
        log_probs = model(tensor)
    return ctc_decode(log_probs, idx_to_char)[0]


# ──────────────────────────────────────────────
# Save Result to Text File
# ──────────────────────────────────────────────
def save_result_txt(image_path: str, predicted_text: str, checkpoint: dict):
    """
    Save prediction result to a .txt file with the same base name as the image.
    Format matches the existing project convention.
    """
    txt_path = os.path.splitext(image_path)[0] + ".txt"
    model_name = os.path.basename(checkpoint.get("model_path", "best_model.pth"))
    epoch     = checkpoint.get("epoch", "?")
    val_cer   = checkpoint.get("val_cer", "?")
    val_wer   = checkpoint.get("val_wer", "?")

    content = (
        f"Image: {os.path.basename(image_path)}\n"
        f"Predicted text: {predicted_text}\n"
        f"Model: {model_name} (epoch {epoch})\n"
        f"Val CER: {val_cer}\n"
        f"Val WER: {val_wer}\n"
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Result saved to    : {txt_path}")


# ──────────────────────────────────────────────
# Inference Entry Points
# ──────────────────────────────────────────────
def run_full_page(model, device, idx_to_char, image_path: str, json_path: str, checkpoint: dict = None):
    """Predict on a full page image that has a matching LabelMe JSON annotation."""
    print(f"Found annotation   : {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    full_img = Image.open(image_path).convert("L")
    words = annotation["shapes"]
    print(f"Words annotated    : {len(words)}\n")

    # Show full page
    fig_full, ax_full = plt.subplots(1, 1, figsize=(12, 16))
    ax_full.imshow(Image.open(image_path), cmap="gray")
    ax_full.set_title(f"Full Page: {os.path.basename(image_path)}", fontsize=14)
    ax_full.axis("off")
    plt.tight_layout()
    plt.show()

    # Predict each word crop
    n_words = len(words)
    cols = 4
    rows = min((n_words + cols - 1) // cols, 10)  # Max 10 rows displayed
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 2.5))
    if rows == 1:
        axes = axes.reshape(1, -1)

    correct, total = 0, 0
    for i, shape in enumerate(words):
        if i >= rows * cols:
            break
        gt_label = shape["label"].strip()
        pts = shape["points"]
        x1, y1 = int(pts[0][0]), int(pts[0][1])
        x2, y2 = int(pts[1][0]), int(pts[1][1])
        x1, x2 = max(0, min(x1, x2)), min(full_img.width,  max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(full_img.height, max(y1, y2))

        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        word_crop = full_img.crop((x1, y1, x2, y2))
        pred = predict_single_image(model, word_crop, device, idx_to_char)

        is_correct = pred == gt_label
        if is_correct:
            correct += 1
        total += 1

        r, c = i // cols, i % cols
        ax = axes[r, c]
        ax.imshow(word_crop, cmap="gray")
        color = "green" if is_correct else "red"
        ax.set_title(
            f"GT: {gt_label}\nPred: {pred}",
            fontsize=9, color=color,
            fontproperties=BENGALI_FONT,
        )
        ax.axis("off")

    # Hide unused axes
    for j in range(total, rows * cols):
        axes[j // cols, j % cols].axis("off")

    acc = correct / max(total, 1) * 100
    plt.suptitle(
        f"Predictions — {os.path.basename(image_path)} | "
        f"Accuracy: {correct}/{total} ({acc:.1f}%)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
    print(f"\nResults: {correct}/{total} words correct ({acc:.1f}% accuracy)")

    # Collect all predictions into one string and save
    all_preds = " ".join(
        predict_single_image(
            model,
            Image.open(image_path).convert("L").crop((
                max(0, min(int(s["points"][0][0]), int(s["points"][1][0]))),
                max(0, min(int(s["points"][0][1]), int(s["points"][1][1]))),
                min(Image.open(image_path).width,  max(int(s["points"][0][0]), int(s["points"][1][0]))),
                min(Image.open(image_path).height, max(int(s["points"][0][1]), int(s["points"][1][1]))),
            )),
            device,
            idx_to_char,
        )
        for s in words
        if (max(int(s["points"][0][0]), int(s["points"][1][0])) - min(int(s["points"][0][0]), int(s["points"][1][0]))) >= 5
        and (max(int(s["points"][0][1]), int(s["points"][1][1])) - min(int(s["points"][0][1]), int(s["points"][1][1]))) >= 5
    )
    if checkpoint:
        save_result_txt(image_path, all_preds, checkpoint)


def run_single_word(model, device, idx_to_char, image_path: str, checkpoint: dict = None):
    """Predict directly on a single word-crop image (no annotation)."""
    print("No JSON annotation found — treating as a single word-crop image.")
    img = Image.open(image_path)
    pred = predict_single_image(model, img, device, idx_to_char)

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.imshow(img, cmap="gray")
    ax.set_title(
        f"Prediction: {pred}",
        fontsize=16, color="blue", fontweight="bold",
        fontproperties=BENGALI_FONT,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    print(f"\n>>> Predicted text: {pred}")
    if checkpoint:
        save_result_txt(image_path, pred, checkpoint)


# ──────────────────────────────────────────────
# Batch Inference Mode
# ──────────────────────────────────────────────
def run_batch(model, device, idx_to_char, raw_dir: str, output_csv: str, checkpoint: dict = None):
    """
    Scan raw_dir for all .jpg + .json pairs, predict every annotated word,
    compare with ground truth, and save results to a CSV file.
    """
    # Discover all JSON annotation files
    json_files = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
    if not json_files:
        print(f"ERROR: No .json files found in {raw_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"  BATCH INFERENCE MODE")
    print(f"  Raw directory  : {raw_dir}")
    print(f"  Annotation files: {len(json_files)}")
    print(f"  Output CSV     : {output_csv}")
    print(f"{'=' * 60}\n")

    # Prepare CSV
    csv_rows = []
    total_words = 0
    total_correct = 0
    skipped_images = 0

    for file_idx, json_path in enumerate(json_files, 1):
        # Find matching image
        base = os.path.splitext(json_path)[0]
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = base + ext
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"  [{file_idx}/{len(json_files)}] SKIP {os.path.basename(json_path)} — no matching image")
            skipped_images += 1
            continue

        # Load annotation
        with open(json_path, "r", encoding="utf-8") as f:
            annotation = json.load(f)
        shapes = annotation.get("shapes", [])

        # Open full-page image
        try:
            full_img = Image.open(img_path).convert("L")
        except Exception as e:
            print(f"  [{file_idx}/{len(json_files)}] ERROR {os.path.basename(img_path)}: {e}")
            skipped_images += 1
            continue

        page_correct = 0
        page_total = 0
        img_basename = os.path.basename(img_path)

        for word_idx, shape in enumerate(shapes):
            gt_label = shape["label"].strip()
            pts = shape["points"]
            x1, y1 = int(pts[0][0]), int(pts[0][1])
            x2, y2 = int(pts[1][0]), int(pts[1][1])
            x1, x2 = max(0, min(x1, x2)), min(full_img.width, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(full_img.height, max(y1, y2))

            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            word_crop = full_img.crop((x1, y1, x2, y2))
            pred = predict_single_image(model, word_crop, device, idx_to_char)

            is_correct = pred == gt_label
            if is_correct:
                page_correct += 1
            page_total += 1

            csv_rows.append({
                "image": img_basename,
                "word_index": word_idx,
                "ground_truth": gt_label,
                "predicted": pred,
                "is_correct": is_correct,
            })

        total_words += page_total
        total_correct += page_correct
        page_acc = page_correct / max(page_total, 1) * 100
        print(
            f"  [{file_idx}/{len(json_files)}] {img_basename:<25s}  "
            f"{page_correct:>4d}/{page_total:<4d} correct  ({page_acc:.1f}%)"
        )

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "word_index", "ground_truth", "predicted", "is_correct"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Print summary
    overall_acc = total_correct / max(total_words, 1) * 100
    print(f"\n{'=' * 60}")
    print(f"  BATCH RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Images processed : {len(json_files) - skipped_images}")
    print(f"  Images skipped   : {skipped_images}")
    print(f"  Total words      : {total_words}")
    print(f"  Correct          : {total_correct}")
    print(f"  Overall accuracy : {overall_acc:.2f}%")
    print(f"  CSV saved to     : {output_csv}")
    print(f"{'=' * 60}")

    # Also save the .txt result alongside the CSV
    if checkpoint:
        txt_path = os.path.splitext(output_csv)[0] + "_summary.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Batch Inference Summary\n")
            f.write(f"=======================\n")
            f.write(f"Raw directory     : {raw_dir}\n")
            f.write(f"Images processed  : {len(json_files) - skipped_images}\n")
            f.write(f"Total words       : {total_words}\n")
            f.write(f"Correct           : {total_correct}\n")
            f.write(f"Overall accuracy  : {overall_acc:.2f}%\n")
            f.write(f"Model             : {os.path.basename(checkpoint.get('model_path', '?'))}\n")
            f.write(f"Epoch             : {checkpoint.get('epoch', '?')}\n")
            f.write(f"Val CER           : {checkpoint.get('val_cer', '?')}\n")
            f.write(f"Val WER           : {checkpoint.get('val_wer', '?')}\n")
        print(f"  Summary saved to : {txt_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Bangla Handwriting Recognition — Inference")
    parser.add_argument(
        "--image", "-i",
        default=DEFAULT_TEST_IMAGE,
        help="Path to the test image (word-crop PNG/JPG, or full-page with matching .json).",
    )
    parser.add_argument(
        "--model", "-m",
        default="best_model.pth",
        help="Model checkpoint filename (default: best_model.pth).",
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Enable batch mode: scan all images in --raw-dir and produce a CSV report.",
    )
    parser.add_argument(
        "--raw-dir",
        default=os.path.join(WORKSPACE, "raw", "raw"),
        help="Directory containing image+JSON pairs for batch mode (default: raw/raw/).",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(WORKSPACE, "batch_results.csv"),
        help="Output CSV path for batch results (default: batch_results.csv).",
    )
    args = parser.parse_args()

    # ── Device ──────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load Model ──────────────────────────
    model_path = find_model(args.model)
    model, _, idx_to_char = load_model(model_path, device)

    # Keep a reference to the raw checkpoint for saving metadata to .txt
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    checkpoint["model_path"] = model_path

    # ── Batch Mode ───────────────────────────
    if args.batch:
        run_batch(model, device, idx_to_char, args.raw_dir, args.output, checkpoint)
        return

    # ── Single Image Mode ────────────────────
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Test image not found: {args.image}")
    print(f"Testing image      : {args.image}")

    base = os.path.splitext(args.image)[0]
    json_path = base + ".json"

    if os.path.exists(json_path):
        run_full_page(model, device, idx_to_char, args.image, json_path, checkpoint)
    else:
        run_single_word(model, device, idx_to_char, args.image, checkpoint)


if __name__ == "__main__":
    main()
