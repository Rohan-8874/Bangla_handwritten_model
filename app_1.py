# ============================================================
# 20. Test on YOUR image — change TEST_IMAGE_PATH below
# ============================================================

# >>>>>>>>>> CHANGE THIS PATH TO YOUR IMAGE <<<<<<<<<<
# On Kaggle: upload your test image as a dataset or use a dataset image
TEST_IMAGE_PATH = "/kaggle/input/banglawriting/converted/converted/0_21_0.jpg"
# Local: "/home/rohan/Softograph/bangla_dataset_1_download/test_image.png"

IMG_HEIGHT, IMG_WIDTH = 64, 256

def ctc_decode(log_probs, idx_to_char):
    _, max_idx = torch.max(log_probs, dim=2)
    max_idx = max_idx.permute(1, 0).cpu().numpy()
    decoded = []
    for seq in max_idx:
        chars, prev = [], -1
        for idx in seq:
            if idx != prev:
                if idx != 0:
                    chars.append(idx_to_char.get(idx, '?'))
                prev = idx
        decoded.append(''.join(chars))
    return decoded

def preprocess_word_crop(img, img_height=64, img_width=256):
    """Resize word image, maintaining aspect ratio, pad to fixed size."""
    if img.mode != 'L':
        img = img.convert('L')
    w, h = img.size
    new_h = img_height
    new_w = max(1, int(w * new_h / h))
    if new_w > img_width:
        new_w = img_width
    img = img.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new('L', (img_width, img_height), 255)
    padded.paste(img, (0, 0))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transform(padded).unsqueeze(0)

def predict_single_image(model, img, device, idx_to_char):
    """Predict text from a single word-crop PIL image."""
    tensor = preprocess_word_crop(img).to(device)
    with torch.no_grad():
        log_probs = model(tensor)
    return ctc_decode(log_probs, idx_to_char)[0]

# --- Check if the image exists ---
assert os.path.exists(TEST_IMAGE_PATH), f"Image not found: {TEST_IMAGE_PATH}"
print(f"Testing image: {TEST_IMAGE_PATH}")

# --- Check if there's a matching JSON annotation ---
base = os.path.splitext(TEST_IMAGE_PATH)[0]
json_path = base + ".json"

if os.path.exists(json_path):
    # Full-page image with annotations — crop each word and predict
    print(f"Found annotation: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    
    full_img = Image.open(TEST_IMAGE_PATH).convert('L')
    words = annotation['shapes']
    print(f"Words annotated: {len(words)}\n")
    
    # Show full page
    fig_full, ax_full = plt.subplots(1, 1, figsize=(12, 16))
    ax_full.imshow(Image.open(TEST_IMAGE_PATH), cmap='gray')
    ax_full.set_title(f"Full Page: {os.path.basename(TEST_IMAGE_PATH)}", fontsize=14)
    ax_full.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Predict each word crop
    n_words = len(words)
    cols = 4
    rows = min((n_words + cols - 1) // cols, 10)  # Max 10 rows
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 2.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    correct, total = 0, 0
    for i, shape in enumerate(words):
        if i >= rows * cols:
            break
        gt_label = shape['label'].strip()
        pts = shape['points']
        x1, y1 = int(pts[0][0]), int(pts[0][1])
        x2, y2 = int(pts[1][0]), int(pts[1][1])
        x1, x2 = max(0, min(x1, x2)), min(full_img.width, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(full_img.height, max(y1, y2))
        
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        
        word_crop = full_img.crop((x1, y1, x2, y2))
        pred = predict_single_image(infer_model, word_crop, device, saved_idx_to_char)
        
        is_correct = pred == gt_label
        if is_correct:
            correct += 1
        total += 1
        
        r, c = i // cols, i % cols
        ax = axes[r, c]
        ax.imshow(word_crop, cmap='gray')
        color = 'green' if is_correct else 'red'
        ax.set_title(f"GT: {gt_label}\nPred: {pred}", fontsize=9, color=color)
        ax.axis('off')
    
    # Hide unused axes
    for i in range(total, rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis('off')
    
    plt.suptitle(f"Predictions on {os.path.basename(TEST_IMAGE_PATH)} — "
                 f"Accuracy: {correct}/{total} ({correct/max(total,1)*100:.1f}%)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\nResults: {correct}/{total} words correct ({correct/max(total,1)*100:.1f}% accuracy)")

else:
    # Single word crop image — predict directly
    print("No JSON annotation found — treating as a single word crop image")
    img = Image.open(TEST_IMAGE_PATH)
    pred = predict_single_image(infer_model, img, device, saved_idx_to_char)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Prediction: {pred}", fontsize=16, color='blue', fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"\n>>> Predicted text: {pred}")