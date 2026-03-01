"""
dog_normal_omni_train.py
────────────────────────────────────────────────────────────────────
강아지 정상 행동 분류 모델 (Behavior / Emotion / Sound / Patella)
- Backbone: EfficientNet-V2-S (이미지), wav2vec2-base (오디오)
- Patella: EfficientNet + 키포인트 MLP 융합 모델 (5클래스: 1,2,3,4,normal)
  · annotation_info 해부학적 랜드마크 좌표 → 30차원 키포인트 벡터
  · 세션(날짜 폴더) 단위 stratified split → 데이터 리키지 방지

[FIX 목록]
  1. drop_last=is_train → BatchNorm1d 배치크기 1 에러 방지
  2. Dataset / DataLoader를 epoch 루프 밖에서 1회만 생성
  3. class_weight / criterion을 루프 밖에서 1회만 계산
  4. GradScaler skip 시 scheduler.step() 조건부 호출 (_scaler_step 헬퍼)
  5. NUM_WORKERS=8, prefetch_factor=2, fork context → deadlock 방지
  6. val accs ZeroDivisionError 방어
────────────────────────────────────────────────────────────────────
"""

import os, gc, random, shutil, json
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import librosa

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    get_cosine_schedule_with_warmup,
)
import torch.optim.lr_scheduler as lr_scheduler
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ───────────────────────────────── CONFIG ─────────────────────────────────────
SEED = 42
random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)

BEHAVIOR_ROOT = "files/1_Animal_Behavior"
EMOTION_ROOT  = "files/2_Animal_emotions"
SOUND_ROOT    = "files/3_Animal_Sound"
PATELLA_ROOT  = "files/6_Animal_Patella"
WORK_DIR      = "files/work/dog_normal_dataset"

DEVICE      = "cuda:1" if torch.cuda.is_available() else "cpu"
EPOCHS      = 100
BATCH_SIZE  = 64
NUM_WORKERS = 24
SR          = 16000
MAX_AUDIO_LEN = SR * 5

IMG_SIZE   = 384
IMG_RESIZE = 416

LR_BACKBONE = 2e-5
LR_HEAD     = 2e-4
LR_AUDIO    = 1e-5
FREEZE_EPOCHS = 5
LABEL_SMOOTHING = 0.1
AUDIO_MODEL_NAME = "facebook/wav2vec2-base"

print(f"🐶 Dog Normal Omni | Device: {DEVICE}")
FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)

# ──────────────────────────────── CLASSES ─────────────────────────────────────
DOG_BEHAVIOR_CLASSES = [
    "DOG_BODYLOWER", "DOG_BODYSCRATCH", "DOG_BODYSHAKE", "DOG_FEETUP",
    "DOG_FOOTUP",    "DOG_HEADING",     "DOG_LYING",     "DOG_MOUNTING",
    "DOG_SIT",       "DOG_TAILING",     "DOG_TAILLOW",   "DOG_TURN",
    "DOG_WALKRUN",
]  # 13클래스

DOG_EMOTION_CLASSES = [
    "dog_angry", "dog_anxious", "dog_confused",
    "dog_happy", "dog_relaxed", "dog_sad",
]  # 6클래스

DOG_SOUND_CLASSES = [
    "dog_bark",               # bark 계열 병합
    "dog_howling",
    "dog_respiratory_event",  # 호흡기 계열 4종 병합
    "dog_whining",
]  # 4클래스 (재구성 후)

DOG_PATELLA_CLASSES = ["1", "2", "3", "4", "normal"]  # 5클래스 (슬개골 이형성 등급)

# ──────────────────────── PATELLA FEATURE SCHEMA ──────────────────────────────
#
# JSON 구조 활용:
#   annotation_info  → 랜드마크 (x,y) 좌표     → raw KP 30d + 관절 각도 9d
#   sensor_values    → 압력판 프레임 데이터     → 통계 피처 18d
#
# 최종 입력 벡터 구성:
#   [0:30]  KP_RAW   raw (x,y,visible) × 5 landmark × 2 side
#   [30:39] KP_ANGLE hip/knee/ankle angle × left/right + asymmetry × 3
#   [39:57] SENSOR   per-frame (mean,max,std,L-col-mean,R-col-mean,asym) × 3 frames  [FIX]
#   [57:59] MEDICAL  left_foot_grade/4, right_foot_grade/4                            [NEW]
#   FEAT_DIM = 59

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
KP_DIM   = len(PATELLA_KP_LABELS) * 2 * 3   # 30 : raw keypoint block
ANG_DIM  = 9    # hip×2 + knee×2 + ankle×2 + asym×3
SENS_DIM = 18   # 6 stats × 3 frames
MED_DIM  = 2    # pet_medical_record_info: left_val, right_val (각 [0,1] 정규화)
FEAT_DIM = KP_DIM + ANG_DIM + SENS_DIM + MED_DIM  # 59

def _angle_deg(a, b, c) -> float:
    """꼭짓점 b의 각도 (도 단위). 좌표 없으면 0 반환."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))

def _parse_patella_features(json_path: str) -> np.ndarray:
    """
    JSON → FEAT_DIM(59) float32 벡터.

    [Block A: 0:30] raw keypoint (x, y, visible) × 10 slots
    [Block B: 30:39] joint angles
        [30,31] hip   angle L/R  : iliac→trochanter→femorotibial
        [32,33] knee  angle L/R  : trochanter→femorotibial→malleolus
        [34,35] ankle angle L/R  : femorotibial→malleolus→metatarsus
        [36]    hip asymmetry    : |L-R| / (L+R+ε)
        [37]    knee asymmetry   (핵심: 슬개골 이형성은 좌우 비대칭으로 드러남)
        [38]    ankle asymmetry
    [Block C: 39:57] sensor_values stats × 3 frames
        프레임 구조: [sync(2), proto(2), cols, rows, frame_idx, grid(cols×rows), ..., end(254)]
        grid data: vals[7 : 7 + cols×rows]  → row-major, 2D 분할로 올바른 L/R 계산
        per frame: [mean, max, std, L-col-mean, R-col-mean, asym]
    [Block D: 57:59] pet_medical_record_info
        [57] left  foot grade (0~4 → /4 정규화)
        [58] right foot grade (0~4 → /4 정규화)
    """
    feat = np.zeros(FEAT_DIM, dtype=np.float32)
    if not os.path.exists(json_path):
        return feat
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # ── Block A: raw keypoint ───────────────────────────────────────────
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

        # ── Block B: joint angles ───────────────────────────────────────────
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
        feat[30], feat[32], feat[34] = angles[0]   # L
        feat[31], feat[33], feat[35] = angles[1]   # R
        for k, (l, r) in enumerate(zip(angles[0], angles[1])):
            feat[36+k] = abs(l - r) / (l + r + 1e-6)

        # ── Block C: sensor_values ──────────────────────────────────────────
        # [FIX] 헤더 구조: [sync(0-1), proto(2-3), cols(4), rows(5), frame_idx(6), grid..., end(254)]
        #       올바른 grid data: vals[7 : 7 + cols×rows]
        #       잘못된 방식(이전): vals[6:] → frame_idx + grid + tail + end_marker(254) 포함
        #       end_marker(254/255≈0.996)가 R_sum을 오염시켜 비대칭값이 0.7~1.0으로 부풀려짐
        sensor_frames = data.get("sensor_values", [])
        for fi, frame in enumerate(sensor_frames[:3]):
            vals = np.array(frame, dtype=np.float32)
            cols = int(vals[4])   # 12
            rows = int(vals[5])   # 10
            grid_size = cols * rows   # 120
            # [FIX] grid 데이터만 추출 (frame_idx / tail / end_marker 제외)
            grid_flat = vals[7 : 7 + grid_size] / 255.0
            if len(grid_flat) < grid_size:
                continue   # 데이터 불완전 시 건너뜀
            # [FIX] 2D 재구성 후 열 기준 L/R 분할 (row-major 구조 반영)
            grid_2d = grid_flat.reshape(rows, cols)
            L_half  = grid_2d[:, : cols // 2]   # 왼쪽 6열
            R_half  = grid_2d[:, cols // 2 :]   # 오른쪽 6열
            L_sum   = L_half.sum()
            R_sum   = R_half.sum()
            base    = 39 + fi * 6
            feat[base+0] = grid_flat.mean()
            feat[base+1] = grid_flat.max()
            feat[base+2] = grid_flat.std()
            feat[base+3] = L_sum / (L_half.size + 1e-6)   # 셀당 평균 (정규화)
            feat[base+4] = R_sum / (R_half.size + 1e-6)
            feat[base+5] = abs(L_sum - R_sum) / (L_sum + R_sum + 1e-6)

        # ── Block D: pet_medical_record_info ────────────────────────────────
        # [NEW] 족별 임상 진단 등급 (0=정상, 1~4=이형성 등급) → /4 정규화
        #       left/right 중 해당 없으면 0 유지 (graceful degradation)
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

# ─────────────────────────────── AUGMENTATION ─────────────────────────────────
def augment_audio(waveform, p=0.7):
    if random.random() > p:
        return waveform
    waveform = librosa.effects.pitch_shift(waveform, sr=SR, n_steps=random.uniform(-2, 2))
    rate = random.uniform(0.85, 1.15)
    stretched = librosa.effects.time_stretch(waveform, rate=rate)
    stretched = stretched[:MAX_AUDIO_LEN] if len(stretched) > MAX_AUDIO_LEN \
                else np.pad(stretched, (0, MAX_AUDIO_LEN - len(stretched)))
    waveform = stretched * 0.99 + np.random.normal(0, 0.003, len(stretched))
    return waveform

# ─────────────────────────────── TRANSFORMS ───────────────────────────────────
TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((IMG_RESIZE, IMG_RESIZE)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
# [FIX] Patella 전용 transform: RandomHorizontalFlip 제외
#   → 이미지 좌우 반전 시 L/R 비대칭 피처(feat[36-38, 42-57])와 불일치 발생
#   → 슬개골 등급은 좌우 비대칭이 핵심 지표이므로 반전 자체가 정보 손상
TRANSFORM_PATELLA_TRAIN = transforms.Compose([
    transforms.Resize((IMG_RESIZE, IMG_RESIZE)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
TRANSFORM_VAL = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─────────────────────────────── DATASETS ─────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, task_dir, class_list, augment=False):
        self.label_to_id = {c: i for i, c in enumerate(class_list)}
        self.samples = []
        for cls in class_list:
            d = os.path.join(task_dir, cls)
            if not os.path.isdir(d): continue
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(d, f), cls))
        self.transform = TRANSFORM_TRAIN if augment else TRANSFORM_VAL
        print(f"  📊 {os.path.basename(task_dir)}: {len(self.samples)} samples, "
              f"{len(self.label_to_id)} classes")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, c = self.samples[idx]
        return self.transform(Image.open(p).convert("RGB")), self.label_to_id[c]


class AudioDataset(Dataset):
    def __init__(self, task_dir, class_list, augment=False):
        self.label_to_id = {c: i for i, c in enumerate(class_list)}
        self.id_to_label = {i: c for i, c in enumerate(class_list)}
        self.augment = augment
        self.samples = []
        for cls in class_list:
            d = os.path.join(task_dir, cls)
            if not os.path.isdir(d): continue
            for f in os.listdir(d):
                if f.lower().endswith(('.wav', '.mp3', '.m4a')):
                    self.samples.append((os.path.join(d, f), cls))
        print(f"  📊 {os.path.basename(task_dir)}: {len(self.samples)} samples, "
              f"{len(self.label_to_id)} classes, augment={augment}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, c = self.samples[idx]
        try:
            w, _ = librosa.load(p, sr=SR, mono=True)
        except Exception:
            w = np.zeros(MAX_AUDIO_LEN)
        if self.augment: w = augment_audio(w)
        w = w[:MAX_AUDIO_LEN] if len(w) > MAX_AUDIO_LEN else np.pad(w, (0, MAX_AUDIO_LEN - len(w)))
        inp = FEATURE_EXTRACTOR(w, sampling_rate=SR, return_tensors="pt")
        return {"input_values": inp.input_values.squeeze(0),
                "labels": torch.tensor(self.label_to_id[c], dtype=torch.long)}

def collate_audio(batch):
    return {"input_values": torch.stack([b["input_values"] for b in batch]),
            "labels":       torch.stack([b["labels"]       for b in batch])}


class PatellaDataset(Dataset):
    """
    슬개골 이형성 등급 분류 Dataset.
    이미지 + annotation_info 키포인트(KP_DIM=30차원) → (img_tensor, kp_tensor, label)
    """
    def __init__(self, task_dir, class_list, augment=False):
        self.label_to_id = {c: i for i, c in enumerate(class_list)}
        self.samples = []   # (img_path, json_path, cls)
        for cls in class_list:
            d = os.path.join(task_dir, cls)
            if not os.path.isdir(d): continue
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path  = os.path.join(d, f)
                    json_path = os.path.splitext(img_path)[0] + '.json'
                    self.samples.append((img_path, json_path, cls))
        self.transform = TRANSFORM_PATELLA_TRAIN if augment else TRANSFORM_VAL  # [FIX] HorizontalFlip 제외
        print(f"  📊 patella ({os.path.basename(task_dir)}): "
              f"{len(self.samples)} samples, {len(self.label_to_id)} classes")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, json_path, cls = self.samples[idx]
        img  = self.transform(Image.open(img_path).convert("RGB"))
        feat = torch.from_numpy(_parse_patella_features(json_path))  # (FEAT_DIM=59,)
        return img, feat, self.label_to_id[cls]


# ─────────────────────────────── MODELS ───────────────────────────────────────
def _efficientnet_backbone():
    b = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    feat = b.classifier[1].in_features  # 1280
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

class OrdinalLoss(nn.Module):
    """
    슬개골 등급(normal<1<2<3<4)은 순서형(ordinal) 데이터.
    인접 등급 오분류보다 2+ 등급 건너뛴 오분류에 더 큰 페널티 부여.
      penalty_weight = 1 + α × |pred - target|
    일반 CrossEntropy는 이 거리를 무시하므로 OrdinalLoss 사용.
    """
    def __init__(self, alpha: float = 0.5, weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha          = alpha
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        log_p  = F.log_softmax(logits, dim=-1)
        # label-smoothed NLL
        nll    = F.nll_loss(log_p, targets, weight=self.weight, reduction="none")
        smooth = -log_p.mean(dim=-1)
        ce     = (1 - self.label_smoothing) * nll + self.label_smoothing * smooth
        # ordinal distance penalty
        preds  = logits.argmax(dim=1)
        dist   = (preds - targets).abs().float()
        weight = 1.0 + self.alpha * dist
        return (ce * weight).mean()


class PatellaModel(nn.Module):
    """
    EfficientNet-V2-S 이미지 피처(1280)
      + 통합 피처 MLP(FEAT_DIM=59 → 128)
          · raw keypoint  30d : 랜드마크 (x,y,visible)
          · joint angles   9d : hip/knee/ankle × L/R + 비대칭 × 3
          · sensor stats  18d : 압력판 그리드(rows×cols) 2D 분할 통계 × 3 frames
          · medical record 2d : 좌/우 족 임상 등급 (0~4 → /4 정규화)
    → 융합 헤드 (1408 → 512 → num_classes)
    """
    def __init__(self, num_classes, feat_dim=FEAT_DIM):
        super().__init__()
        self.backbone, img_feat = _efficientnet_backbone()   # 1280

        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.GELU(),
        )

        fused_feat = img_feat + 128   # 1280 + 128 = 1408
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(fused_feat, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, img, feat):
        img_f  = self.backbone(img)                          # (B, 1280)
        kp_f   = self.feat_branch(feat)                      # (B, 128)
        fused  = torch.cat([img_f, kp_f], dim=1)             # (B, 1408)
        return self.head(fused)


# ─────────────────────────── DATA PREPARATION ─────────────────────────────────
def _task_ready(name, class_list=None,
                img_exts=('.jpg', '.png', '.jpeg', '.wav', '.mp3', '.m4a')):
    p = os.path.join(WORK_DIR, "train", name)
    if not os.path.isdir(p):
        return False
    search_dirs = []
    if class_list:
        for cls in class_list:
            d = os.path.join(p, cls)
            if os.path.isdir(d):
                search_dirs.append(d)
    else:
        for sub in os.listdir(p):
            d = os.path.join(p, sub)
            if os.path.isdir(d):
                search_dirs.append(d)
    total = sum(
        1 for d in search_dirs
        for f in os.listdir(d)
        if f.lower().endswith(img_exts)
    )
    return total > 0

def _patella_split_valid() -> bool:
    """모든 클래스가 train에 1개 이상 + val에 전체 1개 이상인지 검증."""
    img_exts = ('.jpg', '.png', '.jpeg')
    def _count(split, cls):
        d = os.path.join(WORK_DIR, split, "patella", cls)
        if not os.path.isdir(d): return 0
        return sum(1 for f in os.listdir(d) if f.lower().endswith(img_exts))
    for cls in DOG_PATELLA_CLASSES:
        if _count("train", cls) == 0:
            print(f"  ⚠️  patella split 불완전: train/{cls}에 샘플 없음 → 재분할")
            return False
    if sum(_count("val", cls) for cls in DOG_PATELLA_CLASSES) == 0:
        print("  ⚠️  patella split 불완전: val 전체 비어있음 → 재분할")
        return False
    return True


def collect_and_split(src_root, task_name, class_list, oversample_min=0):
    """클래스별 stratified split → WORK_DIR 복사."""
    rng = random.Random(SEED)
    class_files = defaultdict(list)
    for cls in class_list:
        d = os.path.join(src_root, cls)
        if not os.path.isdir(d): continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.wav', '.mp3', '.m4a')):
                    class_files[cls].append(os.path.join(root, f))

    for split in ["train", "val", "test"]:
        for cls in class_list:
            os.makedirs(os.path.join(WORK_DIR, split, task_name, cls), exist_ok=True)

    for cls, paths in class_files.items():
        rng.shuffle(paths)
        n = len(paths)
        n_val  = max(1, int(n * 0.1))
        n_test = max(1, int(n * 0.1))
        n_train = n - n_val - n_test
        splits = {"train": paths[:n_train],
                  "val":   paths[n_train:n_train+n_val],
                  "test":  paths[n_train+n_val:]}
        if oversample_min > 0 and len(splits["train"]) < oversample_min:
            splits["train"] = rng.choices(splits["train"], k=oversample_min)
        for sname, sfiles in splits.items():
            dst_dir = os.path.join(WORK_DIR, sname, task_name, cls)
            for i, src in enumerate(sfiles):
                dst = os.path.join(dst_dir, f"{cls}_{i:05d}{os.path.splitext(src)[1]}")
                if not os.path.exists(dst): shutil.copy2(src, dst)

    print(f"  ✅ {task_name} prepared → {WORK_DIR}")


def collect_and_split_patella(src_root):
    """
    구조: src_root/{class}/{date}/{direction}/{img, json}
    - 세션(날짜 폴더) 단위로 split → 동일 세션 이미지가 train/val에 혼재 방지
    - 이미지와 동명 JSON을 함께 복사
    """
    rng = random.Random(SEED)

    # class → list of sessions, 각 session = [(img_path, json_path), ...]
    class_sessions: dict = defaultdict(list)
    for cls in DOG_PATELLA_CLASSES:
        cls_dir = os.path.join(src_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for date_dir in sorted(os.listdir(cls_dir)):
            date_path = os.path.join(cls_dir, date_dir)
            if not os.path.isdir(date_path):
                continue
            session_files = []
            for dir_name in sorted(os.listdir(date_path)):
                dir_path = os.path.join(date_path, dir_name)
                if not os.path.isdir(dir_path):
                    continue
                for f in os.listdir(dir_path):
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        img_path  = os.path.join(dir_path, f)
                        json_path = os.path.splitext(img_path)[0] + '.json'
                        session_files.append((img_path, json_path))
            if session_files:
                class_sessions[cls].append(session_files)

    for split in ["train", "val", "test"]:
        for cls in DOG_PATELLA_CLASSES:
            os.makedirs(os.path.join(WORK_DIR, split, "patella", cls), exist_ok=True)

    for cls, sessions in class_sessions.items():
        rng.shuffle(sessions)
        n = len(sessions)
        # [FIX] n_train=0 방지: 세션 수별 분기로 train 최소 1세션 보장
        if n >= 4:
            n_val, n_test = max(1, int(n * 0.15)), max(1, int(n * 0.15))
            n_train = n - n_val - n_test
        elif n == 3:
            n_train, n_val, n_test = 2, 1, 0
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            n_train, n_val, n_test = 1, 0, 0
            print(f"  ⚠️  patella/{cls}: 세션 1개 → train에만 배정")

        split_sessions = {
            "train": sessions[:n_train],
            "val":   sessions[n_train:n_train + n_val],
            "test":  sessions[n_train + n_val:n_train + n_val + n_test],
        }
        counts = {k: sum(len(s) for s in v) for k, v in split_sessions.items()}
        print(f"  📂 patella/{cls}: 세션{n}개 → train {counts['train']}장/val {counts['val']}장/test {counts['test']}장")
        for sname, slist in split_sessions.items():
            all_pairs = [pair for session in slist for pair in session]
            dst_dir   = os.path.join(WORK_DIR, sname, "patella", cls)
            for i, (img_src, json_src) in enumerate(all_pairs):
                ext      = os.path.splitext(img_src)[1]
                img_dst  = os.path.join(dst_dir, f"{cls}_{i:05d}{ext}")
                json_dst = os.path.join(dst_dir, f"{cls}_{i:05d}.json")
                if not os.path.exists(img_dst):
                    shutil.copy2(img_src, img_dst)
                if os.path.exists(json_src) and not os.path.exists(json_dst):
                    shutil.copy2(json_src, json_dst)

    print(f"  ✅ patella prepared → {WORK_DIR}")


def prepare_datasets():
    if not _task_ready("behavior", DOG_BEHAVIOR_CLASSES):
        print("📦 Preparing behavior (dog)...")
        collect_and_split(BEHAVIOR_ROOT, "behavior", DOG_BEHAVIOR_CLASSES)
    else: print("✅ behavior ready")

    if not _task_ready("emotion", DOG_EMOTION_CLASSES):
        print("📦 Preparing emotion (dog)...")
        collect_and_split(EMOTION_ROOT, "emotion", DOG_EMOTION_CLASSES, oversample_min=400)
    else: print("✅ emotion ready")

    if not _task_ready("sound", DOG_SOUND_CLASSES):
        print("📦 Preparing sound (dog)...")
        collect_and_split(SOUND_ROOT, "sound", DOG_SOUND_CLASSES, oversample_min=100)
    else: print("✅ sound ready")

    if _patella_split_valid():
        print("✅ patella ready (split validated)")
    else:
        print("📦 Preparing patella (dog) — 재분할...")
        for split in ["train", "val", "test"]:
            p = os.path.join(WORK_DIR, split, "patella")
            if os.path.isdir(p): shutil.rmtree(p)
        collect_and_split_patella(PATELLA_ROOT)


# ──────────────────────────────── HELPERS ─────────────────────────────────────
def make_loader(ds, shuffle, is_audio=False, is_train=True):
    workers = 2 if is_audio else NUM_WORKERS
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        prefetch_factor=2 if workers > 0 else None,              # [FIX 5]
        multiprocessing_context="fork" if workers > 0 else None, # [FIX 5]
        collate_fn=collate_audio if is_audio else None,
        drop_last=is_train,   # [FIX 1]
    )

def get_class_weights(ds, class_list):
    """ImageDataset / AudioDataset용 (samples: list of (path, cls))"""
    labels = [ds.label_to_id[c] for _, c in ds.samples]
    w = compute_class_weight('balanced', classes=np.arange(len(class_list)), y=labels)
    return torch.tensor(w, dtype=torch.float).to(DEVICE)

def get_class_weights_patella(ds):
    """
    PatellaDataset용 (samples: list of (img_path, json_path, cls))

    [FIX] train split에 샘플이 없는 등급이 있을 경우 sklearn ValueError 방어.
    - compute_class_weight는 classes 인자의 모든 값이 y에 최소 1회 이상 등장해야 함.
    - 세션 수가 적은 등급(예: "4")은 train split에 이미지가 0개일 수 있음.
    - 해결: 실제 y에 존재하는 클래스만 balanced weight 계산 후,
             전체 5클래스 크기 tensor로 확장 (없는 클래스는 weight=1.0 으로 채움).
    """
    labels = [ds.label_to_id[c] for _, _, c in ds.samples]
    if not labels:
        return torch.ones(len(DOG_PATELLA_CLASSES), dtype=torch.float).to(DEVICE)

    present    = sorted(set(labels))                          # 실제 존재하는 label_id
    w_partial  = compute_class_weight('balanced',
                                      classes=np.array(present), y=np.array(labels))

    # 전체 클래스 크기(5)로 확장 — 없는 클래스는 weight=1.0 (중립)
    w_full = np.ones(len(DOG_PATELLA_CLASSES), dtype=np.float32)
    for cls_id, weight in zip(present, w_partial):
        w_full[cls_id] = float(weight)

    missing = [DOG_PATELLA_CLASSES[i] for i in range(len(DOG_PATELLA_CLASSES)) if i not in present]
    if missing:
        print(f"  ⚠️  patella train에 샘플 없는 등급: {missing} → weight=1.0 으로 설정")

    return torch.tensor(w_full, dtype=torch.float).to(DEVICE)

def clear(): gc.collect(); torch.cuda.empty_cache()

def _scaler_step(scaler, opt, sched):
    """GradScaler step 후 optimizer가 실제 실행됐을 때만 scheduler.step() [FIX 4]"""
    prev = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    if scaler.get_scale() == prev:   # overflow 없음 → optimizer 정상 실행
        sched.step()

def _save_history_plot(history, best_acc):
    """매 epoch 호출 — val acc 4개 + LR 곡선(WarmRestarts 재시작 시각화)"""
    fig, axes = plt.subplots(1, 5, figsize=(26, 4))
    for ax, key, title, color in zip(
            axes[:4],
            ["behavior_acc", "emotion_acc", "sound_acc", "patella_acc"],
            ["Behavior", "Emotion", "Sound", "Patella"],
            ["steelblue", "seagreen", "tomato", "mediumpurple"]):
        ax.plot([h[key] for h in history], color=color, linewidth=2)
        ax.set_title(f"Dog {title} Val Acc")
        ax.set_xlabel("Epoch"); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    # LR 곡선 (WarmRestarts 재시작 시점 점선 표시)
    ax = axes[4]
    lr_keys = [("emotion_lr","seagreen","Emotion"), ("sound_lr","tomato","Sound"), ("patella_lr","mediumpurple","Patella")]
    for key, color, label in lr_keys:
        if history and key in history[0]:
            ax.plot([h[key] for h in history], color=color, linewidth=1.5, label=label)
    for restart in [20, 40, 60, 80]:
        if restart < len(history):
            ax.axvline(x=restart, color='gray', linestyle=':', alpha=0.5)
    ax.set_title("LR (WarmRestarts T₀=20)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle(f"Dog Normal Omni | Best Avg {best_acc*100:.1f}%", fontweight="bold")
    plt.tight_layout()
    plt.savefig("dog_normal_omni_history.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────── TRAINING ─────────────────────────────────────
def train():
    prepare_datasets()

    # ── 모델 초기화 ──────────────────────────────────────────────────────────────
    behavior_model = ImageModel(len(DOG_BEHAVIOR_CLASSES))
    emotion_model  = ImageModel(len(DOG_EMOTION_CLASSES))
    audio_model    = AudioModel(len(DOG_SOUND_CLASSES))
    patella_model  = PatellaModel(len(DOG_PATELLA_CLASSES), feat_dim=FEAT_DIM)

    # ── Optimizers ───────────────────────────────────────────────────────────────
    def img_opt(m):
        return torch.optim.AdamW([
            {"params": m.backbone.parameters(), "lr": LR_BACKBONE, "weight_decay": 1e-4},
            {"params": m.head.parameters(),     "lr": LR_HEAD,     "weight_decay": 1e-4},
        ])

    def patella_opt_fn(m):
        return torch.optim.AdamW([
            {"params": m.backbone.parameters(),  "lr": LR_BACKBONE, "weight_decay": 1e-4},
            {"params": m.feat_branch.parameters(), "lr": LR_HEAD,     "weight_decay": 1e-4},
            {"params": m.head.parameters(),      "lr": LR_HEAD,     "weight_decay": 1e-4},
        ])

    behavior_opt = img_opt(behavior_model)
    emotion_opt  = img_opt(emotion_model)
    audio_opt    = torch.optim.AdamW(audio_model.parameters(), lr=LR_AUDIO, weight_decay=0.01)
    patella_opt  = patella_opt_fn(patella_model)

    # ── Dataset / DataLoader / criterion — 루프 밖 1회 생성 [FIX 2,3] ───────────
    print("\n📂 Building datasets & loaders (train)...")
    bds = ImageDataset(os.path.join(WORK_DIR,"train","behavior"), DOG_BEHAVIOR_CLASSES, augment=True)
    eds = ImageDataset(os.path.join(WORK_DIR,"train","emotion"),  DOG_EMOTION_CLASSES,  augment=True)
    sds = AudioDataset(os.path.join(WORK_DIR,"train","sound"),    DOG_SOUND_CLASSES,    augment=True)
    pds = PatellaDataset(os.path.join(WORK_DIR,"train","patella"),DOG_PATELLA_CLASSES,  augment=True)

    bl = make_loader(bds, shuffle=True, is_audio=False, is_train=True)
    el = make_loader(eds, shuffle=True, is_audio=False, is_train=True)
    sl = make_loader(sds, shuffle=True, is_audio=True,  is_train=True)
    pl = make_loader(pds, shuffle=True, is_audio=False, is_train=True)

    criterion_b = nn.CrossEntropyLoss(
        weight=get_class_weights(bds, DOG_BEHAVIOR_CLASSES), label_smoothing=LABEL_SMOOTHING)
    criterion_e = nn.CrossEntropyLoss(
        weight=get_class_weights(eds, DOG_EMOTION_CLASSES),  label_smoothing=LABEL_SMOOTHING)
    criterion_s = nn.CrossEntropyLoss(
        weight=get_class_weights(sds, DOG_SOUND_CLASSES))
    # [FIX] CrossEntropy → OrdinalLoss: 등급 순서 반영 (거리 오분류 페널티)
    criterion_p = OrdinalLoss(
        alpha=0.5,
        weight=get_class_weights_patella(pds),
        label_smoothing=LABEL_SMOOTHING)

    print("\n📂 Building datasets & loaders (val)...")
    bds_val = ImageDataset(os.path.join(WORK_DIR,"val","behavior"), DOG_BEHAVIOR_CLASSES, augment=False)
    eds_val = ImageDataset(os.path.join(WORK_DIR,"val","emotion"),  DOG_EMOTION_CLASSES,  augment=False)
    sds_val = AudioDataset(os.path.join(WORK_DIR,"val","sound"),    DOG_SOUND_CLASSES,    augment=False)
    pds_val = PatellaDataset(os.path.join(WORK_DIR,"val","patella"),DOG_PATELLA_CLASSES,  augment=False)

    bl_val = make_loader(bds_val, shuffle=False, is_audio=False, is_train=False)
    el_val = make_loader(eds_val, shuffle=False, is_audio=False, is_train=False)
    sl_val = make_loader(sds_val, shuffle=False, is_audio=True,  is_train=False)
    pl_val = make_loader(pds_val, shuffle=False, is_audio=False, is_train=False)

    # ── Scheduler ────────────────────────────────────────────────────────────────
    def img_sched(opt, n_samples):
        steps = max(1, (n_samples // BATCH_SIZE) * EPOCHS)
        return get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=max(1, steps // 50), num_training_steps=steps)

    # Behavior: ~96%에 수렴 → 단일 cosine warmup 유지 (재시작 불필요)
    behavior_sched = img_sched(behavior_opt, len(bds))

    # Emotion(73% plateau), Sound(~80%), Patella(90% 후 미하락) → WarmRestarts
    # T_0=20: 20, 40, 60, 80 epoch마다 LR 재시작
    emotion_sched = lr_scheduler.CosineAnnealingWarmRestarts(
        emotion_opt, T_0=20, T_mult=1, eta_min=1e-7
    )
    audio_sched = lr_scheduler.CosineAnnealingWarmRestarts(
        audio_opt, T_0=20, T_mult=1, eta_min=1e-8
    )
    patella_sched = lr_scheduler.CosineAnnealingWarmRestarts(
        patella_opt, T_0=20, T_mult=1, eta_min=1e-7
    )

    behavior_scaler = torch.amp.GradScaler("cuda")
    emotion_scaler  = torch.amp.GradScaler("cuda")
    audio_scaler    = torch.amp.GradScaler("cuda")
    patella_scaler  = torch.amp.GradScaler("cuda")

    # ── Freeze backbone 초기화 ──────────────────────────────────────────────────
    for m in [behavior_model, emotion_model, patella_model]:
        for p in m.backbone.parameters(): p.requires_grad = False

    best_acc, history = 0.0, []

    for epoch in range(EPOCHS):
        print(f"\n{'='*55}\nEpoch {epoch+1}/{EPOCHS}\n{'='*55}")

        # Gradual unfreeze
        if epoch == FREEZE_EPOCHS:
            for m in [behavior_model, emotion_model, patella_model]:
                for p in m.backbone.parameters(): p.requires_grad = True
            print(f"  🔓 Backbone unfrozen at epoch {epoch+1}")

        # ── 1. Behavior ──────────────────────────────────────────────────────────
        print("\n🐾 Training Behavior (dog)...")
        behavior_model.to(DEVICE).train()
        loss_b, corr_b, tot_b = 0, 0, 0
        for imgs, labels in tqdm(bl, desc="Behavior", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            behavior_opt.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = behavior_model(imgs)
                loss = criterion_b(logits, labels)
            behavior_scaler.scale(loss).backward()
            behavior_scaler.unscale_(behavior_opt)
            torch.nn.utils.clip_grad_norm_(behavior_model.parameters(), 1.0)
            _scaler_step(behavior_scaler, behavior_opt, behavior_sched)
            loss_b += loss.item()
            corr_b += (logits.argmax(1) == labels).sum().item()
            tot_b  += labels.size(0)
        print(f"  → Loss: {loss_b/len(bl):.4f} | Train Acc: {corr_b/tot_b*100:.1f}%")
        behavior_model.cpu(); clear()

        # ── 2. Emotion ───────────────────────────────────────────────────────────
        print("\n😊 Training Emotion (dog)...")
        emotion_model.to(DEVICE).train()
        loss_e, corr_e, tot_e = 0, 0, 0
        _n_emotion = len(el)
        for _i, (imgs, labels) in enumerate(tqdm(el, desc="Emotion", leave=False)):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            emotion_opt.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = emotion_model(imgs)
                loss = criterion_e(logits, labels)
            emotion_scaler.scale(loss).backward()
            emotion_scaler.unscale_(emotion_opt)
            torch.nn.utils.clip_grad_norm_(emotion_model.parameters(), 1.0)
            prev_s = emotion_scaler.get_scale()
            emotion_scaler.step(emotion_opt); emotion_scaler.update()
            # WarmRestarts: fractional epoch step
            if emotion_scaler.get_scale() == prev_s:
                emotion_sched.step(epoch + _i / _n_emotion)
            loss_e += loss.item()
            corr_e += (logits.argmax(1) == labels).sum().item()
            tot_e  += labels.size(0)
        _e_lr = emotion_opt.param_groups[1]['lr']
        print(f"  → Loss: {loss_e/len(el):.4f} | Train Acc: {corr_e/tot_e*100:.1f}% | LR: {_e_lr:.2e}")
        emotion_model.cpu(); clear()

        # ── 3. Sound ─────────────────────────────────────────────────────────────
        print("\n🔊 Training Sound (dog)...")
        audio_model.to(DEVICE).train()
        loss_s, corr_s, tot_s = 0, 0, 0
        _n_sound = len(sl)
        for _i, batch in enumerate(tqdm(sl, desc="Sound", leave=False)):
            inp, labels = batch["input_values"].to(DEVICE), batch["labels"].to(DEVICE)
            audio_opt.zero_grad()
            with torch.amp.autocast("cuda"):
                out = audio_model(inp)
                loss = criterion_s(out.logits, labels)
            audio_scaler.scale(loss).backward()
            audio_scaler.unscale_(audio_opt)
            torch.nn.utils.clip_grad_norm_(audio_model.parameters(), 1.0)
            prev_s = audio_scaler.get_scale()
            audio_scaler.step(audio_opt); audio_scaler.update()
            if audio_scaler.get_scale() == prev_s:
                audio_sched.step(epoch + _i / _n_sound)
            loss_s += loss.item()
            corr_s += (out.logits.argmax(1) == labels).sum().item()
            tot_s  += labels.size(0)
        _so_lr = audio_opt.param_groups[0]['lr']
        print(f"  → Loss: {loss_s/len(sl):.4f} | Train Acc: {corr_s/tot_s*100:.1f}% | LR: {_so_lr:.2e}")
        audio_model.cpu(); clear()

        # ── 4. Patella ───────────────────────────────────────────────────────────
        print("\n🦴 Training Patella (dog)...")
        patella_model.to(DEVICE).train()
        loss_p, corr_p, tot_p = 0, 0, 0
        _n_patella = len(pl)
        for _i, (imgs, kps, labels) in enumerate(tqdm(pl, desc="Patella", leave=False)):
            imgs, kps, labels = imgs.to(DEVICE), kps.to(DEVICE), labels.to(DEVICE)
            patella_opt.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = patella_model(imgs, kps)
                loss = criterion_p(logits, labels)
            patella_scaler.scale(loss).backward()
            patella_scaler.unscale_(patella_opt)
            torch.nn.utils.clip_grad_norm_(patella_model.parameters(), 1.0)
            prev_s = patella_scaler.get_scale()
            patella_scaler.step(patella_opt); patella_scaler.update()
            if patella_scaler.get_scale() == prev_s:
                patella_sched.step(epoch + _i / _n_patella)
            loss_p += loss.item()
            corr_p += (logits.argmax(1) == labels).sum().item()
            tot_p  += labels.size(0)
        _p_lr = patella_opt.param_groups[1]['lr']
        print(f"  → Loss: {loss_p/len(pl):.4f} | Train Acc: {corr_p/tot_p*100:.1f}% | LR: {_p_lr:.2e}")
        patella_model.cpu(); clear()

        # ── Validation ────────────────────────────────────────────────────────────
        print("\n🔍 Validation...")
        accs = {}

        # behavior / emotion
        for name, model, val_loader in [
            ("behavior", behavior_model, bl_val),
            ("emotion",  emotion_model,  el_val),
        ]:
            model.to(DEVICE).eval()
            c, t = 0, 0
            with torch.no_grad():
                for imgs, lbl in val_loader:
                    imgs, lbl = imgs.to(DEVICE), lbl.to(DEVICE)
                    c += (model(imgs).argmax(1) == lbl).sum().item()
                    t += lbl.size(0)
            accs[name] = c/t if t > 0 else 0.0   # [FIX 6]
            print(f"  {name:10s}: {accs[name]*100:.1f}%")
            model.cpu(); clear()

        # sound
        audio_model.to(DEVICE).eval()
        c, t = 0, 0
        with torch.no_grad():
            for batch in sl_val:
                inp = batch["input_values"].to(DEVICE)
                lbl = batch["labels"].to(DEVICE)
                c += (audio_model(inp).logits.argmax(1) == lbl).sum().item()
                t += lbl.size(0)
        accs["sound"] = c/t if t > 0 else 0.0
        print(f"  {'sound':10s}: {accs['sound']*100:.1f}%")
        audio_model.cpu(); clear()

        # patella
        patella_model.to(DEVICE).eval()
        c, t = 0, 0
        with torch.no_grad():
            for imgs, kps, lbl in pl_val:
                imgs, kps, lbl = imgs.to(DEVICE), kps.to(DEVICE), lbl.to(DEVICE)
                c += (patella_model(imgs, kps).argmax(1) == lbl).sum().item()
                t += lbl.size(0)
        accs["patella"] = c/t if t > 0 else 0.0
        print(f"  {'patella':10s}: {accs['patella']*100:.1f}%")
        patella_model.cpu(); clear()

        avg = sum(accs.values()) / len(accs)
        print(f"  Average: {avg*100:.1f}%")
        history.append({"epoch": epoch+1,
                        **{k+"_acc": v for k, v in accs.items()},
                        "avg_acc": avg,
                        "emotion_lr": emotion_opt.param_groups[1]["lr"],
                        "sound_lr":   audio_opt.param_groups[0]["lr"],
                        "patella_lr": patella_opt.param_groups[1]["lr"]})

        if avg > best_acc:
            best_acc = avg
            torch.save({
                "behavior_model": behavior_model.state_dict(),
                "emotion_model":  emotion_model.state_dict(),
                "audio_model":    audio_model.state_dict(),
                "patella_model":  patella_model.state_dict(),
                "dog_behavior_classes": DOG_BEHAVIOR_CLASSES,
                "dog_emotion_classes":  DOG_EMOTION_CLASSES,
                "dog_sound_classes":    DOG_SOUND_CLASSES,
                "dog_patella_classes":  DOG_PATELLA_CLASSES,
                "kp_dim": KP_DIM,
                "best_epoch": epoch+1, "best_acc": best_acc, "history": history,
            }, "dog_normal_omni_best.pth")
            print(f"  💾 Saved! (Avg {best_acc*100:.1f}%)")

        # 매 epoch 학습 곡선 덮어쓰기 저장
        _save_history_plot(history, best_acc)

    print(f"\n🎉 Done! Best Avg Acc: {best_acc*100:.1f}%")

if __name__ == "__main__":
    train()
