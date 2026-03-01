"""
cat_normal_omni.py
────────────────────────────────────────────────────────────────────
고양이 정상 행동 분류 모델 (Behavior / Emotion / Sound)
- Backbone: EfficientNet-V2-S (이미지), wav2vec2-base (오디오)
- Patella 없음 (고양이 전용 데이터 없음)
- Sound: cat_aggressive / cat_positive 2클래스 (샘플 극소, 오버샘플링 필수)
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
WORK_DIR      = "files/work/cat_normal_dataset"

DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"
EPOCHS      = 100
BATCH_SIZE  = 64          # EfficientNet 384px → VRAM 고려
NUM_WORKERS = 24
SR          = 16000
MAX_AUDIO_LEN = SR * 5

IMG_SIZE   = 384          # EfficientNet-V2-S 권장
IMG_RESIZE = 416

LR_BACKBONE = 2e-5
LR_HEAD     = 2e-4
LR_AUDIO    = 1e-5
FREEZE_EPOCHS = 5
LABEL_SMOOTHING = 0.1
AUDIO_MODEL_NAME = "facebook/wav2vec2-base"

# [FIX-SOUND] 오디오 전용 배치 크기
# BATCH_SIZE=64로 425샘플 처리 시 epoch당 배치 6개뿐 (drop_last로 25개 버림)
# → 16으로 낮춰 epoch당 26 배치로 4배 이상 gradient update 확보
AUDIO_BATCH_SIZE = 16

print(f"🐱 Cat Normal Omni | Device: {DEVICE}")
FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)

# ──────────────────────────────── CLASSES ─────────────────────────────────────
# 진단 결과 클래스 목록 고정 (collect 시 이 목록만 수집)
CAT_BEHAVIOR_CLASSES = [
    "CAT_ARCH", "CAT_ARMSTRETCH", "CAT_FOOTPUSH", "CAT_GETDOWN",
    "CAT_GROOMING", "CAT_HEADING", "CAT_LAYDOWN", "CAT_LYING",
    "CAT_ROLL", "CAT_SITDOWN", "CAT_TAILING", "CAT_WALKRUN",
]  # 12클래스, max/min≈1.7x (균형 양호)

CAT_EMOTION_CLASSES = [
    "cat_relaxed", "cat_happy", "cat_attentive", "cat_sad",
]  # 4클래스 — cat_sad 극소(136개) → 오버샘플링 필수

CAT_SOUND_CLASSES = [
    "cat_aggressive", "cat_positive",
]  # 2클래스, 합계 55개 → 극소, 오버샘플링으로 보완

# ─────────────────────────────── AUGMENTATION ─────────────────────────────────
def augment_audio(waveform, p=0.7):
    """
    [FIX-SOUND] librosa pitch_shift / time_stretch 완전 제거.

    근거:
      librosa FFT 계열 함수는 DataLoader multiprocessing fork worker 내에서
      내부 스레드 상태 충돌로 수치 불안정 유발.
      → val acc가 Ep7(59%)→Ep8(53%)처럼 급락하는 원인.

    대체: 모두 numpy 연산만 사용 (worker 내 완전 안전)
      1. Speed perturbation : 선형 보간으로 0.88~1.14배 속도 변환 (pitch 무변화)
      2. Gaussian noise      : SNR 약 25~30dB 수준의 약한 백색 잡음
      3. Amplitude scaling   : 음량 ±25% 랜덤 조절
    """
    if random.random() > p:
        return waveform
    # 1. Speed perturbation (librosa time_stretch 대체)
    speed   = random.uniform(0.88, 1.14)
    new_len = max(1, int(len(waveform) / speed))
    indices = np.linspace(0, len(waveform) - 1, new_len)
    waveform = np.interp(indices, np.arange(len(waveform)), waveform).astype(np.float32)
    # 길이 고정
    waveform = (waveform[:MAX_AUDIO_LEN] if len(waveform) > MAX_AUDIO_LEN
                else np.pad(waveform, (0, MAX_AUDIO_LEN - len(waveform))).astype(np.float32))
    # 2. Gaussian noise
    waveform = waveform + np.random.normal(0, 0.004, len(waveform)).astype(np.float32)
    # 3. Amplitude scaling
    waveform = waveform * random.uniform(0.75, 1.25)
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
        print(f"  📊 {os.path.basename(task_dir)}: {len(self.samples)} samples, {len(self.label_to_id)} classes")

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
        print(f"  📊 {os.path.basename(task_dir)}: {len(self.samples)} samples, {len(self.label_to_id)} classes, augment={augment}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, c = self.samples[idx]
        try:
            w, _ = librosa.load(p, sr=SR, mono=True)
        except:
            w = np.zeros(MAX_AUDIO_LEN)
        if self.augment: w = augment_audio(w)
        w = w[:MAX_AUDIO_LEN] if len(w) > MAX_AUDIO_LEN else np.pad(w, (0, MAX_AUDIO_LEN - len(w)))
        inp = FEATURE_EXTRACTOR(w, sampling_rate=SR, return_tensors="pt")
        return {"input_values": inp.input_values.squeeze(0),
                "labels": torch.tensor(self.label_to_id[c], dtype=torch.long)}

def collate_audio(batch):
    return {"input_values": torch.stack([b["input_values"] for b in batch]),
            "labels":       torch.stack([b["labels"]       for b in batch])}

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

# ─────────────────────────── DATA PREPARATION ─────────────────────────────────
def _task_ready(name, class_list=None,
                img_exts=('.jpg', '.png', '.jpeg', '.wav', '.mp3', '.m4a')):
    """
    [FIX] 폴더 존재 여부만 보던 기존 로직을 개선.
    class_list 가 주어지면 각 클래스 폴더 안의 실제 파일 수를 합산하여
    1개 이상일 때만 True 를 반환한다.
    (날짜 하위 폴더가 남아 있어 폴더만 존재하는 경우 False 처리)
    """
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

def collect_and_split(src_root, task_name, class_list, oversample_min=0):
    """클래스별 stratified split → WORK_DIR 복사. oversample_min>0 이면 train 오버샘플링."""
    rng = random.Random(SEED)
    class_files = defaultdict(list)
    for cls in class_list:
        d = os.path.join(src_root, cls)
        if not os.path.isdir(d): continue
        # [FIX] os.listdir() → os.walk() : 날짜/ID 하위 폴더가 있는 경우
        #       (1_Animal_Behavior 등) 재귀 탐색으로 이미지를 빠짐없이 수집.
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.wav', '.mp3', '.m4a')):
                    class_files[cls].append(os.path.join(root, f))

    for split in ["train","val","test"]:
        for cls in class_list:
            os.makedirs(os.path.join(WORK_DIR, split, task_name, cls), exist_ok=True)

    for cls, paths in class_files.items():
        rng.shuffle(paths)
        n = len(paths)
        n_val  = max(1, int(n * 0.1))
        n_test = max(1, int(n * 0.1))
        n_train = n - n_val - n_test

        splits = {"train": paths[:n_train], "val": paths[n_train:n_train+n_val], "test": paths[n_train+n_val:]}

        # 오버샘플링 (sound처럼 소수 클래스에 필요)
        if oversample_min > 0 and len(splits["train"]) < oversample_min:
            splits["train"] = rng.choices(splits["train"], k=oversample_min)

        for sname, sfiles in splits.items():
            dst_dir = os.path.join(WORK_DIR, sname, task_name, cls)
            for i, src in enumerate(sfiles):
                dst = os.path.join(dst_dir, f"{cls}_{i:05d}{os.path.splitext(src)[1]}")
                if not os.path.exists(dst): shutil.copy2(src, dst)

    print(f"  ✅ {task_name} prepared → {WORK_DIR}")

def prepare_datasets():
    if not _task_ready("behavior", CAT_BEHAVIOR_CLASSES):
        print("📦 Preparing behavior (cat)...")
        collect_and_split(BEHAVIOR_ROOT, "behavior", CAT_BEHAVIOR_CLASSES)
    else: print("✅ behavior ready")

    if not _task_ready("emotion", CAT_EMOTION_CLASSES):
        print("📦 Preparing emotion (cat)...")
        # cat_sad: 136개 → 오버샘플링으로 최소 400개 보장
        collect_and_split(EMOTION_ROOT, "emotion", CAT_EMOTION_CLASSES, oversample_min=400)
    else: print("✅ emotion ready")

    if not _task_ready("sound", CAT_SOUND_CLASSES):
        print("📦 Preparing sound (cat)...")
        # cat 음성 극소 (55개 합계) → 강한 오버샘플링
        # [FIX-SOUND] 150→300: AUDIO_BATCH_SIZE=16 기준 epoch당 37배치 확보
        collect_and_split(SOUND_ROOT, "sound", CAT_SOUND_CLASSES, oversample_min=300)
    else: print("✅ sound ready")

# ──────────────────────────────── HELPERS ─────────────────────────────────────
def make_loader(ds, shuffle, is_audio=False, is_train=True):
    # [FIX-SOUND] 오디오는 AUDIO_BATCH_SIZE(16) 사용: 25→26 배치/epoch
    batch   = AUDIO_BATCH_SIZE if is_audio else BATCH_SIZE
    workers = 2 if is_audio else NUM_WORKERS
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=workers, pin_memory=True,
                      persistent_workers=(workers > 0), prefetch_factor=2,
                      multiprocessing_context="fork" if workers > 0 else None,
                      collate_fn=collate_audio if is_audio else None,
                      drop_last=is_train)

def get_class_weights(ds, class_list):
    labels = [ds.label_to_id[c] for _, c in ds.samples]
    w = compute_class_weight('balanced', classes=np.arange(len(class_list)), y=labels)
    return torch.tensor(w, dtype=torch.float).to(DEVICE)

def clear(): gc.collect(); torch.cuda.empty_cache()

def _save_history_plot(history, best_acc):
    """매 epoch 호출 — val acc 3개 + LR 곡선(WarmRestarts 재시작 시각화)"""
    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    for ax, key, title, color in zip(axes[:3],
            ["behavior_acc","emotion_acc","sound_acc"],
            ["Behavior","Emotion","Sound"],["steelblue","seagreen","tomato"]):
        ax.plot([h[key] for h in history], color=color, linewidth=2)
        ax.set_title(f"Cat {title} Val Acc"); ax.set_ylim(0,1); ax.grid(True, alpha=0.3)
    # LR 곡선
    ax = axes[3]
    if history and "emotion_lr" in history[0]:
        ax.plot([h["emotion_lr"] for h in history], color="seagreen", linewidth=1.5, label="Emotion LR")
    if history and "sound_lr" in history[0]:
        ax.plot([h["sound_lr"]   for h in history], color="tomato",   linewidth=1.5, label="Sound LR")
    for restart in [20, 40, 60, 80]:
        if restart < len(history):
            ax.axvline(x=restart, color='gray', linestyle=':', alpha=0.5)
    ax.set_title("LR (WarmRestarts T₀=20)"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle(f"Cat Normal Omni | Best Avg {best_acc*100:.1f}%", fontweight="bold")
    plt.tight_layout()
    plt.savefig("cat_normal_omni_history.png", dpi=150, bbox_inches="tight")
    plt.close()

# ─────────────────────────────── TRAINING ─────────────────────────────────────
def train():
    prepare_datasets()

    # ── 모델 초기화 ──────────────────────────────────────────────────────────────
    behavior_model = ImageModel(len(CAT_BEHAVIOR_CLASSES))
    emotion_model  = ImageModel(len(CAT_EMOTION_CLASSES))
    audio_model    = AudioModel(len(CAT_SOUND_CLASSES))

    # ── Optimizers ───────────────────────────────────────────────────────────────
    def img_opt(m):
        return torch.optim.AdamW([
            {"params": m.backbone.parameters(), "lr": LR_BACKBONE, "weight_decay": 1e-4},
            {"params": m.head.parameters(),     "lr": LR_HEAD,     "weight_decay": 1e-4},
        ])

    behavior_opt = img_opt(behavior_model)
    emotion_opt  = img_opt(emotion_model)
    audio_opt    = torch.optim.AdamW(audio_model.parameters(), lr=LR_AUDIO, weight_decay=0.01)

    behavior_scaler = torch.amp.GradScaler("cuda")
    emotion_scaler  = torch.amp.GradScaler("cuda")
    audio_scaler    = torch.amp.GradScaler("cuda")

    # ── [FIX 3] Dataset / DataLoader / criterion을 루프 밖에서 1회만 생성 ───────
    # Worker 프로세스 생성/소멸 반복 제거 → persistent_workers 효과 실현
    print("\n📂 Building datasets & loaders (train)...")
    bds = ImageDataset(os.path.join(WORK_DIR,"train","behavior"), CAT_BEHAVIOR_CLASSES, augment=True)
    eds = ImageDataset(os.path.join(WORK_DIR,"train","emotion"),  CAT_EMOTION_CLASSES,  augment=True)
    sds = AudioDataset(os.path.join(WORK_DIR,"train","sound"),    CAT_SOUND_CLASSES,    augment=True)

    bl = make_loader(bds, shuffle=True,  is_audio=False, is_train=True)
    el = make_loader(eds, shuffle=True,  is_audio=False, is_train=True)
    sl = make_loader(sds, shuffle=True,  is_audio=True,  is_train=True)

    # criterion도 1회만 계산
    criterion_b = nn.CrossEntropyLoss(weight=get_class_weights(bds, CAT_BEHAVIOR_CLASSES), label_smoothing=LABEL_SMOOTHING)
    criterion_e = nn.CrossEntropyLoss(weight=get_class_weights(eds, CAT_EMOTION_CLASSES),  label_smoothing=LABEL_SMOOTHING)
    criterion_s = nn.CrossEntropyLoss(weight=get_class_weights(sds, CAT_SOUND_CLASSES))

    # ── Scheduler (cosine warmup) — Dataset 생성 이후 len(ds) 기반으로 계산 ───
    def img_sched(opt, n_samples):
        steps = (n_samples // BATCH_SIZE) * EPOCHS
        return get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(1, steps//50), num_training_steps=max(1, steps))

    # Behavior: 99%에 수렴 → 단일 cosine warmup 유지 (재시작 불필요)
    behavior_sched = img_sched(behavior_opt, len(bds))

    # Emotion(81% plateau) + Sound(~80%, 진동) → WarmRestarts
    # T_0=20: 20, 40, 60, 80 epoch마다 LR 재시작 → plateau 탈출
    # step(epoch + i/n_batches): 배치 단위 호출로 LR 곡선이 부드럽게 유지
    emotion_sched = lr_scheduler.CosineAnnealingWarmRestarts(
        emotion_opt, T_0=20, T_mult=1, eta_min=1e-7
    )
    audio_sched = lr_scheduler.CosineAnnealingWarmRestarts(
        audio_opt, T_0=20, T_mult=1, eta_min=1e-8
    )

    # [FIX 1] val 로더는 is_train=False → drop_last=False → ZeroDivisionError 방지
    print("\n📂 Building datasets & loaders (val)...")
    bds_val = ImageDataset(os.path.join(WORK_DIR,"val","behavior"), CAT_BEHAVIOR_CLASSES, augment=False)
    eds_val = ImageDataset(os.path.join(WORK_DIR,"val","emotion"),  CAT_EMOTION_CLASSES,  augment=False)
    sds_val = AudioDataset(os.path.join(WORK_DIR,"val","sound"),    CAT_SOUND_CLASSES,    augment=False)

    bl_val = make_loader(bds_val, shuffle=False, is_audio=False, is_train=False)
    el_val = make_loader(eds_val, shuffle=False, is_audio=False, is_train=False)
    sl_val = make_loader(sds_val, shuffle=False, is_audio=True,  is_train=False)

    # ── Freeze backbone 초기화 ──────────────────────────────────────────────────
    for m in [behavior_model, emotion_model]:
        for p in m.backbone.parameters(): p.requires_grad = False

    best_acc, history = 0.0, []

    for epoch in range(EPOCHS):
        print(f"\n{'='*55}\nEpoch {epoch+1}/{EPOCHS}\n{'='*55}")

        # Gradual unfreeze
        if epoch == FREEZE_EPOCHS:
            for m in [behavior_model, emotion_model]:
                for p in m.backbone.parameters(): p.requires_grad = True
            print(f"  🔓 Backbone unfrozen at epoch {epoch+1}")

        # ── 1. Behavior ──────────────────────────────────────────────────────────
        print("\n🐾 Training Behavior (cat)...")
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
            # [FIX 2] optimizer가 실제로 실행됐을 때만 scheduler.step()
            prev_scale = behavior_scaler.get_scale()
            behavior_scaler.step(behavior_opt); behavior_scaler.update()
            if behavior_scaler.get_scale() == prev_scale:
                behavior_sched.step()
            loss_b += loss.item(); corr_b += (logits.argmax(1)==labels).sum().item(); tot_b += labels.size(0)
        print(f"  → Loss: {loss_b/len(bl):.4f} | Train Acc: {corr_b/tot_b*100:.1f}%")
        behavior_model.cpu(); clear()

        # ── 2. Emotion ───────────────────────────────────────────────────────────
        print("\n😊 Training Emotion (cat)...")
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
            prev_scale = emotion_scaler.get_scale()
            emotion_scaler.step(emotion_opt); emotion_scaler.update()
            # WarmRestarts: fractional epoch step (부드러운 LR 곡선)
            if emotion_scaler.get_scale() == prev_scale:
                emotion_sched.step(epoch + _i / _n_emotion)
            loss_e += loss.item(); corr_e += (logits.argmax(1)==labels).sum().item(); tot_e += labels.size(0)
        _e_lr = emotion_opt.param_groups[1]['lr']
        print(f"  → Loss: {loss_e/len(el):.4f} | Train Acc: {corr_e/tot_e*100:.1f}% | LR: {_e_lr:.2e}")
        emotion_model.cpu(); clear()

        # ── 3. Sound ─────────────────────────────────────────────────────────────
        print("\n🔊 Training Sound (cat)...")
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
            prev_scale = audio_scaler.get_scale()
            audio_scaler.step(audio_opt); audio_scaler.update()
            # WarmRestarts: fractional epoch step
            if audio_scaler.get_scale() == prev_scale:
                audio_sched.step(epoch + _i / _n_sound)
            loss_s += loss.item(); corr_s += (out.logits.argmax(1)==labels).sum().item(); tot_s += labels.size(0)
        _s_lr = audio_opt.param_groups[0]['lr']
        print(f"  → Loss: {loss_s/len(sl):.4f} | Train Acc: {corr_s/tot_s*100:.1f}% | LR: {_s_lr:.2e}")
        audio_model.cpu(); clear()

        # ── Validation ────────────────────────────────────────────────────────────
        print("\n🔍 Validation...")
        accs = {}
        for name, model, val_loader, is_audio in [
            ("behavior", behavior_model, bl_val, False),
            ("emotion",  emotion_model,  el_val, False),
            ("sound",    audio_model,    sl_val, True),
        ]:
            model.to(DEVICE).eval()
            c, t = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    if is_audio:
                        inp = batch["input_values"].to(DEVICE)
                        lbl = batch["labels"].to(DEVICE)
                        p = model(inp).logits.argmax(1)
                    else:
                        imgs, lbl = batch[0].to(DEVICE), batch[1].to(DEVICE)
                        p = model(imgs).argmax(1)
                    c += (p==lbl).sum().item(); t += lbl.size(0)
            accs[name] = c/t if t > 0 else 0.0   # [FIX 1] 추가 방어: val 샘플 0개 대비
            print(f"  {name:10s}: {accs[name]*100:.1f}%")
            model.cpu(); clear()

        avg = sum(accs.values()) / len(accs)
        print(f"  Average: {avg*100:.1f}%")
        history.append({"epoch": epoch+1,
                         **{k+"_acc": v for k,v in accs.items()},
                         "avg_acc": avg,
                         "emotion_lr": emotion_opt.param_groups[1]["lr"],
                         "sound_lr":   audio_opt.param_groups[0]["lr"]})

        if avg > best_acc:
            best_acc = avg
            torch.save({
                "behavior_model": behavior_model.state_dict(),
                "emotion_model":  emotion_model.state_dict(),
                "audio_model":    audio_model.state_dict(),
                "cat_behavior_classes": CAT_BEHAVIOR_CLASSES,
                "cat_emotion_classes":  CAT_EMOTION_CLASSES,
                "cat_sound_classes":    CAT_SOUND_CLASSES,
                "best_epoch": epoch+1, "best_acc": best_acc, "history": history,
            }, "cat_normal_omni_best.pth")
            print(f"  💾 Saved! (Avg {best_acc*100:.1f}%)")

        # 매 epoch 학습 곡선 덮어쓰기 저장
        _save_history_plot(history, best_acc)

    print(f"\n🎉 Done! Best Avg Acc: {best_acc*100:.1f}%")

if __name__ == "__main__":
    train()
