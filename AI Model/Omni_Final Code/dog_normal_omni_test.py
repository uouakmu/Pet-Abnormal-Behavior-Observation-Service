"""
dog_normal_omni_test.py
────────────────────────────────────────────────────────────────────
강아지 정상 행동 분류 모델 테스트 (Behavior / Emotion / Sound / Patella)
- 체크포인트: dog_normal_omni_best.pth
- 테스트 데이터: files/work/dog_normal_dataset/test/
- 출력: 태스크별 정확도 + 클래스별 정확도 + Confusion Matrix PNG
────────────────────────────────────────────────────────────────────

사용법:
  python dog_normal_omni_test.py
  python dog_normal_omni_test.py --model_path /path/to/dog_normal_omni_best.pth
  python dog_normal_omni_test.py --model_path dog_normal_omni_best.pth --device cuda:0
"""

import os, gc, random, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
)
from collections import defaultdict
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ───────────────────────────────── CONFIG ─────────────────────────────────────
SEED = 42
random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)

WORK_DIR         = "files/work/dog_normal_dataset"
DEFAULT_CKPT     = "dog_normal_omni_best.pth"
AUDIO_MODEL_NAME = "facebook/wav2vec2-base"

SR            = 16000
MAX_AUDIO_LEN = SR * 5
IMG_SIZE      = 384
BATCH_SIZE    = 64
AUDIO_BATCH   = 16
NUM_WORKERS   = 4

# ──────────────────────── PATELLA FEATURE SCHEMA ──────────────────────────────
PATELLA_KP_LABELS = [
    "Iliac crest",
    "Femoral greater trochanter",
    "Femorotibial joint",
    "Lateral malleolus of the distal tibia",
    "Distal lateral aspect of the fifth metatarsus",
]
PATELLA_KP_SLOT = {
    f"{label}_{i}": idx * 2 + i
    for idx, label in enumerate(PATELLA_KP_LABELS)
    for i in range(2)
}
KP_DIM   = len(PATELLA_KP_LABELS) * 2 * 3   # 30
ANG_DIM  = 9
SENS_DIM = 18
MED_DIM  = 2
FEAT_DIM = KP_DIM + ANG_DIM + SENS_DIM + MED_DIM  # 59

# ───────────────────────────── ARG PARSE ──────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Dog Normal Omni — Test")
    p.add_argument("--model_path", type=str, default=DEFAULT_CKPT,
                   help=f"체크포인트 경로 (기본: {DEFAULT_CKPT})")
    p.add_argument("--device", type=str,
                   default="cuda:1" if torch.cuda.is_available() else "cpu",
                   help="추론 디바이스 (기본: cuda:1)")
    p.add_argument("--work_dir", type=str, default=WORK_DIR,
                   help=f"데이터셋 루트 (기본: {WORK_DIR})")
    return p.parse_args()

# ─────────────────────────────── TRANSFORM ────────────────────────────────────
TRANSFORM_TEST = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ──────────────────────── PATELLA FEATURE PARSING ─────────────────────────────
def _angle_deg(a, b, c) -> float:
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))

def _parse_patella_features(json_path: str) -> np.ndarray:
    """JSON → FEAT_DIM(59) float32 벡터 (학습 시와 동일한 로직)."""
    feat = np.zeros(FEAT_DIM, dtype=np.float32)
    if not os.path.exists(json_path):
        return feat
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Block A: raw keypoint
        label_count: dict = defaultdict(int)
        kp_xy: dict = {}
        for ann in data.get("annotation_info", []):
            label = ann["label"]
            occ   = label_count[label]
            label_count[label] += 1
            slot  = PATELLA_KP_SLOT.get(f"{label}_{occ}")
            if slot is None:
                continue
            x, y = float(ann["x"]), float(ann["y"])
            base  = slot * 3
            feat[base], feat[base+1], feat[base+2] = x, y, 1.0
            kp_xy[f"{label}_{occ}"] = (x, y)

        # Block B: joint angles
        ZERO = (0.0, 0.0)
        angles = []
        for side in [0, 1]:
            iliac = kp_xy.get(
                f"Iliac crest_{min(side, label_count.get('Iliac crest', 0) - 1)}", ZERO)
            troch = kp_xy.get(f"Femoral greater trochanter_{side}", ZERO)
            ftj   = kp_xy.get(f"Femorotibial joint_{side}", ZERO)
            mall  = kp_xy.get(f"Lateral malleolus of the distal tibia_{side}", ZERO)
            meta  = kp_xy.get(f"Distal lateral aspect of the fifth metatarsus_{side}", ZERO)
            hip   = _angle_deg(iliac, troch, ftj)  / 180.0
            knee  = _angle_deg(troch, ftj,   mall) / 180.0
            ankle = _angle_deg(ftj,   mall,  meta) / 180.0
            angles.append((hip, knee, ankle))
        feat[30], feat[32], feat[34] = angles[0]
        feat[31], feat[33], feat[35] = angles[1]
        for k, (l, r) in enumerate(zip(angles[0], angles[1])):
            feat[36+k] = abs(l - r) / (l + r + 1e-6)

        # Block C: sensor_values
        sensor_frames = data.get("sensor_values", [])
        for fi, frame in enumerate(sensor_frames[:3]):
            vals = np.array(frame, dtype=np.float32)
            cols = int(vals[4])
            rows = int(vals[5])
            grid_size = cols * rows
            grid_flat = vals[7 : 7 + grid_size] / 255.0
            if len(grid_flat) < grid_size:
                continue
            grid_2d = grid_flat.reshape(rows, cols)
            L_half  = grid_2d[:, : cols // 2]
            R_half  = grid_2d[:, cols // 2 :]
            L_sum   = L_half.sum()
            R_sum   = R_half.sum()
            base    = 39 + fi * 6
            feat[base+0] = grid_flat.mean()
            feat[base+1] = grid_flat.max()
            feat[base+2] = grid_flat.std()
            feat[base+3] = L_sum / (L_half.size + 1e-6)
            feat[base+4] = R_sum / (R_half.size + 1e-6)
            feat[base+5] = abs(L_sum - R_sum) / (L_sum + R_sum + 1e-6)

        # Block D: medical record
        MAX_GRADE = 4.0
        for rec in data.get("pet_medical_record_info", []):
            pos = rec.get("foot_position", "")
            val = float(rec.get("value", 0)) / MAX_GRADE
            if pos == "left":
                feat[57] = val
            elif pos == "right":
                feat[58] = val

    except Exception:
        pass
    return feat

# ─────────────────────────────── DATASETS ─────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, task_dir, class_list):
        self.label_to_id = {c: i for i, c in enumerate(class_list)}
        self.samples = []
        for cls in class_list:
            d = os.path.join(task_dir, cls)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(d, f), cls))
        print(f"  📂 {os.path.basename(task_dir)}: {len(self.samples)} test samples")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, c = self.samples[idx]
        return TRANSFORM_TEST(Image.open(p).convert("RGB")), self.label_to_id[c]


class AudioDataset(Dataset):
    def __init__(self, task_dir, class_list, feature_extractor):
        self.label_to_id = {c: i for i, c in enumerate(class_list)}
        self.fe = feature_extractor
        self.samples = []
        for cls in class_list:
            d = os.path.join(task_dir, cls)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.lower().endswith(('.wav', '.mp3', '.m4a')):
                    self.samples.append((os.path.join(d, f), cls))
        print(f"  📂 {os.path.basename(task_dir)}: {len(self.samples)} test samples")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, c = self.samples[idx]
        try:
            w, _ = librosa.load(p, sr=SR, mono=True)
        except Exception:
            w = np.zeros(MAX_AUDIO_LEN, dtype=np.float32)
        w = (w[:MAX_AUDIO_LEN] if len(w) > MAX_AUDIO_LEN
             else np.pad(w, (0, MAX_AUDIO_LEN - len(w))).astype(np.float32))
        inp = self.fe(w, sampling_rate=SR, return_tensors="pt")
        return {"input_values": inp.input_values.squeeze(0),
                "labels": torch.tensor(self.label_to_id[c], dtype=torch.long)}

def collate_audio(batch):
    return {"input_values": torch.stack([b["input_values"] for b in batch]),
            "labels":       torch.stack([b["labels"]       for b in batch])}


class PatellaDataset(Dataset):
    def __init__(self, task_dir, class_list):
        self.label_to_id = {c: i for i, c in enumerate(class_list)}
        self.samples = []
        for cls in class_list:
            d = os.path.join(task_dir, cls)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path  = os.path.join(d, f)
                    json_path = os.path.splitext(img_path)[0] + '.json'
                    self.samples.append((img_path, json_path, cls))
        print(f"  📂 patella ({os.path.basename(task_dir)}): {len(self.samples)} test samples")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, json_path, cls = self.samples[idx]
        img  = TRANSFORM_TEST(Image.open(img_path).convert("RGB"))
        feat = torch.from_numpy(_parse_patella_features(json_path))
        return img, feat, self.label_to_id[cls]

# ─────────────────────────────── MODELS ───────────────────────────────────────
def _efficientnet_backbone():
    b = efficientnet_v2_s(weights=None)
    feat = b.classifier[1].in_features
    b.classifier = nn.Identity()
    return b, feat

class ImageModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone, feat = _efficientnet_backbone()
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    def forward(self, x): return self.head(self.backbone(x))

class AudioModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            AUDIO_MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    def forward(self, input_values, labels=None):
        return self.model(input_values=input_values, labels=labels)

class PatellaModel(nn.Module):
    def __init__(self, num_classes, feat_dim=FEAT_DIM):
        super().__init__()
        self.backbone, img_feat = _efficientnet_backbone()
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        fused_feat = img_feat + 128   # 1408
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(fused_feat, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, img, feat):
        img_f = self.backbone(img)
        kp_f  = self.feat_branch(feat)
        return self.head(torch.cat([img_f, kp_f], dim=1))

# ──────────────────────────────── HELPERS ─────────────────────────────────────
def clear():
    gc.collect()
    torch.cuda.empty_cache()

def make_loader(ds, is_audio=False):
    return DataLoader(
        ds,
        batch_size=AUDIO_BATCH if is_audio else BATCH_SIZE,
        shuffle=False,
        num_workers=2 if is_audio else NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context="fork",
        collate_fn=collate_audio if is_audio else None,
        drop_last=False,
    )

# ─────────────────────────── EVALUATION ───────────────────────────────────────
def evaluate_image(model, loader, class_list, device):
    n = len(class_list)
    correct = np.zeros(n, dtype=int)
    total   = np.zeros(n, dtype=int)
    all_preds, all_labels = [], []

    model.to(device).eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  Evaluating", ncols=80, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast("cuda" if "cuda" in device else "cpu"):
                preds = model(imgs).argmax(1)
            for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                correct[l] += int(p == l); total[l] += 1
                all_preds.append(p); all_labels.append(l)
    model.cpu(); clear()

    per_class = np.where(total > 0, correct / total, 0.0)
    overall   = correct.sum() / total.sum() if total.sum() > 0 else 0.0
    return overall, per_class, np.array(all_preds), np.array(all_labels)

def evaluate_audio(model, loader, class_list, device):
    n = len(class_list)
    correct = np.zeros(n, dtype=int)
    total   = np.zeros(n, dtype=int)
    all_preds, all_labels = [], []

    model.to(device).eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", ncols=80, leave=False):
            inp    = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda" if "cuda" in device else "cpu"):
                preds = model(inp).logits.argmax(1)
            for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                correct[l] += int(p == l); total[l] += 1
                all_preds.append(p); all_labels.append(l)
    model.cpu(); clear()

    per_class = np.where(total > 0, correct / total, 0.0)
    overall   = correct.sum() / total.sum() if total.sum() > 0 else 0.0
    return overall, per_class, np.array(all_preds), np.array(all_labels)

def evaluate_patella(model, loader, class_list, device):
    n = len(class_list)
    correct = np.zeros(n, dtype=int)
    total   = np.zeros(n, dtype=int)
    all_preds, all_labels = [], []

    model.to(device).eval()
    with torch.no_grad():
        for imgs, kps, labels in tqdm(loader, desc="  Evaluating", ncols=80, leave=False):
            imgs, kps, labels = imgs.to(device), kps.to(device), labels.to(device)
            with torch.amp.autocast("cuda" if "cuda" in device else "cpu"):
                preds = model(imgs, kps).argmax(1)
            for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                correct[l] += int(p == l); total[l] += 1
                all_preds.append(p); all_labels.append(l)
    model.cpu(); clear()

    per_class = np.where(total > 0, correct / total, 0.0)
    overall   = correct.sum() / total.sum() if total.sum() > 0 else 0.0
    return overall, per_class, np.array(all_preds), np.array(all_labels)

# ─────────────────────────── CONFUSION MATRIX ─────────────────────────────────
def plot_confusion_matrix(preds, labels, class_list, title, save_path):
    n = len(class_list)
    cm = np.zeros((n, n), dtype=int)
    for p, l in zip(preds, labels):
        cm[l][p] += 1

    cm_norm = cm.astype(float)
    row_sum  = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sum > 0, cm_norm / row_sum, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(max(10, n * 0.9 + 2), max(7, n * 0.8 + 1)))

    for ax, data, fmt, ttl in zip(
        axes, [cm, cm_norm], [".0f", ".2f"],
        ["Count", "Recall (row-normalized)"]
    ):
        im = ax.imshow(data, interpolation="nearest",
                       cmap="Blues" if "Count" in ttl else "RdYlGn",
                       vmin=0, vmax=(None if "Count" in ttl else 1.0))
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(n)); ax.set_xticklabels(class_list, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(class_list, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{title} — {ttl}")
        thresh = data.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 {save_path}")

# ─────────────────────────────── MAIN TEST ────────────────────────────────────
def test():
    args = parse_args()
    device    = args.device
    ckpt_path = args.model_path
    work_dir  = args.work_dir

    print(f"\n🐶 Dog Normal Omni — Test")
    print(f"  Device    : {device}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Work Dir  : {work_dir}")

    # ── 1. 체크포인트 로드 ───────────────────────────────────────────────────────
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    DOG_BEHAVIOR_CLASSES = ckpt["dog_behavior_classes"]
    DOG_EMOTION_CLASSES  = ckpt["dog_emotion_classes"]
    DOG_SOUND_CLASSES    = ckpt["dog_sound_classes"]
    DOG_PATELLA_CLASSES  = ckpt["dog_patella_classes"]
    best_epoch = ckpt.get("best_epoch", "?")
    best_acc   = ckpt.get("best_acc",   0.0)

    print(f"\n  📋 Best Epoch: {best_epoch}  |  Best Val Avg: {best_acc*100:.1f}%")
    print(f"  Behavior classes ({len(DOG_BEHAVIOR_CLASSES)}): {DOG_BEHAVIOR_CLASSES}")
    print(f"  Emotion  classes ({len(DOG_EMOTION_CLASSES)}):  {DOG_EMOTION_CLASSES}")
    print(f"  Sound    classes ({len(DOG_SOUND_CLASSES)}):    {DOG_SOUND_CLASSES}")
    print(f"  Patella  classes ({len(DOG_PATELLA_CLASSES)}):  {DOG_PATELLA_CLASSES}")

    # ── 2. 모델 복원 ─────────────────────────────────────────────────────────────
    print("\n  🔧 모델 복원 중...")
    behavior_model = ImageModel(len(DOG_BEHAVIOR_CLASSES))
    emotion_model  = ImageModel(len(DOG_EMOTION_CLASSES))
    audio_model    = AudioModel(len(DOG_SOUND_CLASSES))
    patella_model  = PatellaModel(len(DOG_PATELLA_CLASSES), feat_dim=FEAT_DIM)

    behavior_model.load_state_dict(ckpt["behavior_model"])
    emotion_model.load_state_dict(ckpt["emotion_model"])
    audio_model.load_state_dict(ckpt["audio_model"])
    patella_model.load_state_dict(ckpt["patella_model"])
    print("  ✅ 모델 가중치 로드 완료")

    # ── 3. Feature Extractor (오디오) ────────────────────────────────────────────
    print(f"\n  🎙️ Feature Extractor 로드: {AUDIO_MODEL_NAME}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)

    # ── 4. 테스트 데이터셋 로드 ──────────────────────────────────────────────────
    test_root = os.path.join(work_dir, "test")
    print(f"\n  📦 Test 데이터 로드: {test_root}")

    bds = ImageDataset(os.path.join(test_root, "behavior"), DOG_BEHAVIOR_CLASSES)
    eds = ImageDataset(os.path.join(test_root, "emotion"),  DOG_EMOTION_CLASSES)
    sds = AudioDataset(os.path.join(test_root, "sound"),    DOG_SOUND_CLASSES, feature_extractor)
    pds = PatellaDataset(os.path.join(test_root, "patella"), DOG_PATELLA_CLASSES)

    bl = make_loader(bds, is_audio=False)
    el = make_loader(eds, is_audio=False)
    sl = make_loader(sds, is_audio=True)
    pl = make_loader(pds, is_audio=False)

    # ── 5. 평가 ──────────────────────────────────────────────────────────────────
    results = {}
    print("\n" + "="*55 + "\n  📊 TEST EVALUATION\n" + "="*55)

    # Behavior
    print("\n🐾 Behavior")
    b_acc, b_per, b_preds, b_labels = evaluate_image(
        behavior_model, bl, DOG_BEHAVIOR_CLASSES, device)
    results["behavior"] = {"acc": b_acc, "per_class": b_per,
                           "preds": b_preds, "labels": b_labels}
    print(f"  Overall: {b_acc*100:.2f}%")
    for cls, acc in zip(DOG_BEHAVIOR_CLASSES, b_per):
        print(f"    {cls:<40s}: {acc*100:.1f}%")

    # Emotion
    print("\n😊 Emotion")
    e_acc, e_per, e_preds, e_labels = evaluate_image(
        emotion_model, el, DOG_EMOTION_CLASSES, device)
    results["emotion"] = {"acc": e_acc, "per_class": e_per,
                          "preds": e_preds, "labels": e_labels}
    print(f"  Overall: {e_acc*100:.2f}%")
    for cls, acc in zip(DOG_EMOTION_CLASSES, e_per):
        print(f"    {cls:<40s}: {acc*100:.1f}%")

    # Sound
    print("\n🔊 Sound")
    s_acc, s_per, s_preds, s_labels = evaluate_audio(
        audio_model, sl, DOG_SOUND_CLASSES, device)
    results["sound"] = {"acc": s_acc, "per_class": s_per,
                        "preds": s_preds, "labels": s_labels}
    print(f"  Overall: {s_acc*100:.2f}%")
    for cls, acc in zip(DOG_SOUND_CLASSES, s_per):
        print(f"    {cls:<40s}: {acc*100:.1f}%")

    # Patella
    print("\n🦴 Patella")
    p_acc, p_per, p_preds, p_labels = evaluate_patella(
        patella_model, pl, DOG_PATELLA_CLASSES, device)
    results["patella"] = {"acc": p_acc, "per_class": p_per,
                          "preds": p_preds, "labels": p_labels}
    print(f"  Overall: {p_acc*100:.2f}%")
    for cls, acc in zip(DOG_PATELLA_CLASSES, p_per):
        print(f"    {cls:<40s}: {acc*100:.1f}%")

    # ── 6. 전체 요약 ─────────────────────────────────────────────────────────────
    avg = np.mean([r["acc"] for r in results.values()])
    print("\n" + "="*55)
    print(f"  ✅ Behavior : {results['behavior']['acc']*100:.2f}%")
    print(f"  ✅ Emotion  : {results['emotion']['acc']*100:.2f}%")
    print(f"  ✅ Sound    : {results['sound']['acc']*100:.2f}%")
    print(f"  ✅ Patella  : {results['patella']['acc']*100:.2f}%")
    print(f"  ─────────────────────")
    print(f"  📌 Test Avg : {avg*100:.2f}%")
    print(f"  📌 Best Val : {best_acc*100:.1f}%  (Epoch {best_epoch})")
    print("="*55)

    # ── 7. Confusion Matrix 저장 ─────────────────────────────────────────────────
    print("\n  📈 Confusion Matrix 저장 중...")
    class_map = {
        "behavior": DOG_BEHAVIOR_CLASSES,
        "emotion":  DOG_EMOTION_CLASSES,
        "sound":    DOG_SOUND_CLASSES,
        "patella":  DOG_PATELLA_CLASSES,
    }
    for task, (preds, labels) in [
        ("behavior", (b_preds, b_labels)),
        ("emotion",  (e_preds, e_labels)),
        ("sound",    (s_preds, s_labels)),
        ("patella",  (p_preds, p_labels)),
    ]:
        plot_confusion_matrix(
            preds, labels, class_map[task],
            f"Dog {task.capitalize()} (Test)",
            f"dog_normal_{task}_cm.png"
        )

    # ── 8. 결과 요약 PNG ─────────────────────────────────────────────────────────
    _save_summary_plot(results, class_map, avg, best_acc, best_epoch)

    print("\n🎉 테스트 완료!")


def _save_summary_plot(results, class_map, avg_acc, best_val_acc, best_epoch):
    """태스크별 per-class 막대 그래프 + 전체 요약"""
    n_tasks = len(results)
    fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 6))

    colors = {"behavior": "steelblue", "emotion": "seagreen",
              "sound": "tomato", "patella": "mediumpurple"}

    for ax, (task, res) in zip(axes, results.items()):
        classes = class_map[task]
        per_cls = res["per_class"]
        x = np.arange(len(classes))
        bars = ax.bar(x, per_cls * 100, color=colors.get(task, "gray"),
                      alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axhline(y=res["acc"] * 100, color="red", linestyle="--", linewidth=1.5,
                   label=f"Overall {res['acc']*100:.1f}%")
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 115); ax.set_ylabel("Accuracy (%)"); ax.set_title(f"Dog {task.capitalize()} — Test")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, per_cls):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val*100:.0f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle(
        f"Dog Normal Omni — Test Summary\n"
        f"Test Avg: {avg_acc*100:.2f}%  |  Best Val: {best_val_acc*100:.1f}% (Ep {best_epoch})",
        fontweight="bold", fontsize=12,
    )
    plt.tight_layout()
    plt.savefig("dog_normal_omni_test_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  💾 dog_normal_omni_test_summary.png")


if __name__ == "__main__":
    test()
