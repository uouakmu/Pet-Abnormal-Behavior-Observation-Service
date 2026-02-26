import os
import random
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, get_linear_schedule_with_warmup
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image, ImageFile
import librosa
import numpy as np
from collections import defaultdict, Counter
from sklearn.utils.class_weight import compute_class_weight
import json
import gc

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# 0. 설정
# =========================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

BEHAVIOR_ROOT = "files/1_Animal_Behavior"
EMOTION_ROOT  = "files/2_Animal_emotions"
SOUND_ROOT    = "files/3_Animal_Sound"
PATELLA_ROOT  = "files/6_Animal_Patella"
WORK_DIR      = "files/work/omni_dataset"

BATCH_SIZE  = 64
EPOCHS      = 100
LR_VIDEO    = 5e-5
LR_AUDIO    = 1e-5
DEVICE      = "cuda:1" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 24
SR          = 16000
MAX_AUDIO_LEN = SR * 5

LOSS_WEIGHTS = {
    "behavior": 1.0,
    "emotion":  1.0,
    "sound":    0.6,
    "patella":  1.0,
}

AUDIO_MODEL_NAME = "facebook/wav2vec2-base"
FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)

print(f"🎯 Device: {DEVICE}")

# =========================
# 🔥 Audio Augmentation
# =========================
def augment_audio(waveform, p=0.5):
    if random.random() > p:
        return waveform

    n_steps = random.uniform(-2, 2)
    waveform = librosa.effects.pitch_shift(waveform, sr=SR, n_steps=n_steps)

    rate = random.uniform(0.9, 1.1)
    stretched = librosa.effects.time_stretch(waveform, rate=rate)
    if len(stretched) > MAX_AUDIO_LEN:
        stretched = stretched[:MAX_AUDIO_LEN]
    else:
        stretched = np.pad(stretched, (0, MAX_AUDIO_LEN - len(stretched)))
    waveform = stretched

    noise = np.random.normal(0, 0.003, len(waveform))
    waveform = waveform * 0.99 + noise

    return waveform

# =========================
# 1. Dataset Preparation
# =========================
def collect_samples(root, exts):
    samples = []
    for class_dir in sorted(os.listdir(root)):
        class_path = os.path.join(root, class_dir)
        if not os.path.isdir(class_path):
            continue

        for root_dir, _, files in os.walk(class_path):
            for filename in files:
                if any(filename.lower().endswith(ext) for ext in exts):
                    file_path = os.path.join(root_dir, filename)
                    samples.append((class_dir, file_path))

    print(f"  → {len(samples)} samples, {len(set(s[0] for s in samples))} classes")
    return samples

def collect_patella_samples(root):
    samples = []

    for grade in sorted(os.listdir(root)):
        grade_path = os.path.join(root, grade)
        if not os.path.isdir(grade_path):
            continue

        for date_dir in os.listdir(grade_path):
            date_path = os.path.join(grade_path, date_dir)
            if not os.path.isdir(date_path):
                continue

            for direction in ['Back', 'Front', 'Left', 'Right']:
                direction_path = os.path.join(date_path, direction)
                if not os.path.exists(direction_path):
                    continue

                for filename in os.listdir(direction_path):
                    if filename.lower().endswith('.jpg'):
                        img_path = os.path.join(direction_path, filename)
                        json_path = img_path.replace('.jpg', '.json')

                        if os.path.exists(json_path):
                            samples.append((grade, img_path, json_path))

    print(f"  → {len(samples)} samples, {len(set(s[0] for s in samples))} classes")
    return samples

def sample_balanced(samples):
    """샘플링 없이 전체 데이터 반환. 불균형은 학습 시 class_weight로 보정."""
    class_counts = defaultdict(int)
    for label, _ in samples:
        class_counts[label] += 1
    print(f"  📊 {len(class_counts)} classes, total {len(samples)} samples (all used)")
    for label, cnt in sorted(class_counts.items()):
        print(f"    {label}: {cnt}")
    return samples

def sample_balanced_audio(samples):
    """샘플링 없이 전체 오디오 데이터 반환. 불균형은 class_weight로 보정."""
    class_counts = defaultdict(int)
    for label, _ in samples:
        class_counts[label] += 1
    print(f"  📊 {len(class_counts)} classes, total {len(samples)} samples (all used)")
    for label, cnt in sorted(class_counts.items()):
        print(f"    {label}: {cnt}")
    return samples


def _dedup_samples(samples, is_patella=False):
    """
    파일 경로 기준 중복 제거.
    os.walk로 재귀 수집 시 동일 파일이 중복 등록되는 경우를 방지.
    → train/val/test 분리 전에 반드시 호출해 데이터 누수 원천 차단.
    """
    seen = set()
    deduped = []
    if is_patella:
        for label, img_path, json_path in samples:
            if img_path not in seen:
                seen.add(img_path)
                deduped.append((label, img_path, json_path))
    else:
        for label, path in samples:
            if path not in seen:
                seen.add(path)
                deduped.append((label, path))
    removed = len(samples) - len(deduped)
    if removed > 0:
        print(f"  ⚠️  중복 파일 {removed}개 제거 (총 {len(deduped)}개 사용)")
    return deduped


def split_and_copy(samples, task_name, is_patella=False, original_samples=None):
    """
    train/val/test 분리 후 WORK_DIR로 파일 복사.

    [데이터 누수 방지 수정]
    1. 모든 태스크: _dedup_samples()로 파일 경로 중복 제거 후 분리.
    2. sound 태스크: test 파일을 original_samples에서 먼저 확정하고,
       해당 경로들을 오버샘플 pool(samples)에서 사전 제거.
       → train/val ↔ test 겹침 완전 차단.
    """
    # ── 중복 경로 제거 ──
    samples = _dedup_samples(samples, is_patella=is_patella)

    random.shuffle(samples)
    class_samples = defaultdict(list)

    if is_patella:
        for label, img_path, json_path in samples:
            class_samples[label].append((img_path, json_path))
    else:
        for label, path in samples:
            class_samples[label].append(path)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(WORK_DIR, split, task_name), exist_ok=True)

    # ── sound: test 파일 먼저 확정 → train/val pool에서 제거 ──
    if original_samples is not None:
        orig_class = defaultdict(list)
        for label, path in _dedup_samples(original_samples):
            orig_class[label].append(path)

        test_items_by_label = {}
        excluded_paths = set()
        for label, paths in orig_class.items():
            n_test = max(10, len(paths) // 5)
            test_paths = paths[:n_test]
            test_items_by_label[label] = test_paths
            excluded_paths.update(test_paths)  # test 파일 집합 확정

        # 오버샘플 pool에서 test 파일 제거
        filtered_class_samples = defaultdict(list)
        for label, paths in class_samples.items():
            filtered_class_samples[label] = [p for p in paths if p not in excluded_paths]
        class_samples = filtered_class_samples
    else:
        test_items_by_label = None

    for label, items in class_samples.items():
        n = len(items)
        n_train = int(n * 0.8)
        n_val   = int(n * 0.1)

        if test_items_by_label is not None:
            train_items = items[:n_train]
            val_items   = items[n_train:n_train + n_val]
            test_items  = test_items_by_label.get(label, [])
        else:
            train_items = items[:n_train]
            val_items   = items[n_train:n_train + n_val]
            test_items  = items[n_train + n_val:]

        split_map = {"train": train_items, "val": val_items, "test": test_items}

        for split_name, split_items in split_map.items():
            dst_label_dir = os.path.join(WORK_DIR, split_name, task_name, label)
            os.makedirs(dst_label_dir, exist_ok=True)

            for item in tqdm(split_items, desc=f"{task_name}/{split_name}/{label}", leave=False):
                if is_patella:
                    img_path, json_path = item
                    dst_img  = os.path.join(dst_label_dir, f"{label}_{os.path.basename(img_path)}")
                    shutil.copy(img_path, dst_img)
                    dst_json = dst_img.replace('.jpg', '.json')
                    shutil.copy(json_path, dst_json)
                else:
                    dst_path = os.path.join(dst_label_dir, f"{label}_{os.path.basename(item)}")
                    shutil.copy(item, dst_path)


def _task_ready(task_name):
    """해당 task의 train 폴더가 존재하고 비어있지 않으면 True"""
    task_train = os.path.join(WORK_DIR, "train", task_name)
    return os.path.isdir(task_train) and len(os.listdir(task_train)) > 0


def prepare_dataset():
    need_behavior = not _task_ready("behavior")
    need_emotion  = not _task_ready("emotion")
    need_sound    = not _task_ready("sound")
    need_patella  = not _task_ready("patella")

    if not any([need_behavior, need_emotion, need_sound, need_patella]):
        print("✅ All file-copy tasks already prepared, skipping.")
        return

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(WORK_DIR, split), exist_ok=True)

    if need_behavior:
        print("\n📦 Collecting behavior (all samples)...")
        behavior_all = collect_samples(BEHAVIOR_ROOT, ['.jpg', '.png', '.jpeg'])
        behavior = sample_balanced(behavior_all)
        print("  📋 Splitting & Copying behavior...")
        split_and_copy(behavior, "behavior")
    else:
        print("✅ behavior already prepared, skipping.")

    if need_emotion:
        print("\n📦 Collecting emotion (all samples)...")
        emotion_all = collect_samples(EMOTION_ROOT, ['.jpg', '.png', '.jpeg'])
        emotion = sample_balanced(emotion_all)
        print("  📋 Splitting & Copying emotion...")
        split_and_copy(emotion, "emotion")
    else:
        print("✅ emotion already prepared, skipping.")

    if need_sound:
        print("\n📦 Collecting sound (all samples)...")
        sound_all = collect_samples(SOUND_ROOT, ['.wav', '.mp3', '.m4a'])
        sound = sample_balanced_audio(sound_all)
        print("  📋 Splitting & Copying sound...")
        split_and_copy(sound, "sound", original_samples=sound_all)
    else:
        print("✅ sound already prepared, skipping.")

    if need_patella:
        print("\n📦 Collecting patella luxation (all samples)...")
        patella_all = collect_patella_samples(PATELLA_ROOT)
        print("  ℹ️  Patella: Using all samples")
        print("  📋 Splitting & Copying patella...")
        split_and_copy(patella_all, "patella", is_patella=True)
    else:
        print("✅ patella already prepared, skipping.")

    print("\n✅ Dataset preparation complete.")


# =========================
# 2. Dataset Classes
# =========================
class ImageDataset(Dataset):
    def __init__(self, task_dir, augment=False):
        self.samples = []
        self.label_to_id = {}

        for label in sorted(os.listdir(task_dir)):
            label_dir = os.path.join(task_dir, label)
            if not os.path.isdir(label_dir):
                continue

            self.label_to_id[label] = len(self.label_to_id)

            for file in os.listdir(label_dir):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(label_dir, file), label))

        print(f"  📊 {os.path.basename(task_dir)}: {len(self.samples)} samples, {len(self.label_to_id)} classes")

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, self.label_to_id[label]


class PatellaDataset(Dataset):
    def __init__(self, task_dir, augment=False):
        self.samples = []
        self.label_to_id = {}

        for label in sorted(os.listdir(task_dir)):
            label_dir = os.path.join(task_dir, label)
            if not os.path.isdir(label_dir):
                continue

            self.label_to_id[label] = len(self.label_to_id)

            for file in os.listdir(label_dir):
                if file.lower().endswith('.jpg'):
                    img_path = os.path.join(label_dir, file)
                    json_path = img_path.replace('.jpg', '.json')

                    if os.path.exists(json_path):
                        self.samples.append((img_path, json_path, label))

        print(f"  📊 {os.path.basename(task_dir)}: {len(self.samples)} samples, {len(self.label_to_id)} classes")

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        keypoints = []
        for annotation in data.get('annotation_info', []):
            x = float(annotation.get('x', 0))
            y = float(annotation.get('y', 0))
            keypoints.extend([x, y])

        while len(keypoints) < 18:
            keypoints.append(0.0)

        keypoints = torch.tensor(keypoints[:18], dtype=torch.float32)

        return img, keypoints, self.label_to_id[label]


class AudioDataset(Dataset):
    def __init__(self, task_dir, augment=False):
        self.samples = []
        self.label_to_id = {}
        self.id_to_label = {}
        self.augment = augment
        next_id = 0

        for label in sorted(os.listdir(task_dir)):
            label_dir = os.path.join(task_dir, label)
            if not os.path.isdir(label_dir):
                continue

            self.label_to_id[label] = next_id
            self.id_to_label[next_id] = label
            next_id += 1

            for file in os.listdir(label_dir):
                if file.lower().endswith(('.wav', '.mp3', '.m4a')):
                    self.samples.append((os.path.join(label_dir, file), label))

        print(f"  📊 {os.path.basename(task_dir)}: {len(self.samples)} samples, {len(self.label_to_id)} classes, augment={augment}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            waveform, _ = librosa.load(path, sr=SR, mono=True)
        except Exception:
            waveform = np.zeros(MAX_AUDIO_LEN)

        if self.augment:
            waveform = augment_audio(waveform)

        if len(waveform) > MAX_AUDIO_LEN:
            waveform = waveform[:MAX_AUDIO_LEN]
        else:
            waveform = np.pad(waveform, (0, MAX_AUDIO_LEN - len(waveform)))

        inputs = FEATURE_EXTRACTOR(waveform, sampling_rate=SR, return_tensors="pt")
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(self.label_to_id[label], dtype=torch.long)
        }


def collate_fn_audio(batch):
    input_values = torch.stack([item["input_values"] for item in batch])
    labels       = torch.stack([item["labels"]       for item in batch])
    return {"input_values": input_values, "labels": labels}


# =========================
# 3. Individual Models
# =========================
def _efficientnet_b3_backbone():
    backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = backbone.classifier[1].in_features  # 1536
    backbone.classifier = nn.Identity()
    return backbone, in_features


class BehaviorModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone, in_features = _efficientnet_b3_backbone()
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class EmotionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone, in_features = _efficientnet_b3_backbone()
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class PatellaModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone, in_features = _efficientnet_b3_backbone()
        self.head = nn.Sequential(
            nn.Linear(in_features + 18, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, keypoints):
        feat = self.backbone(x)
        return self.head(torch.cat([feat, keypoints], dim=1))


class AudioModel(nn.Module):
    def __init__(self, num_classes, freeze_backbone=False):
        super().__init__()
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            AUDIO_MODEL_NAME,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        if freeze_backbone:
            for param in self.model.wav2vec2.parameters():
                param.requires_grad = False

    def forward(self, input_values, labels=None):
        return self.model(input_values=input_values, labels=labels)


# =========================
# 4. Helper Functions
# =========================
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()


def make_loader(dataset, shuffle, is_audio=False):
    """
    공통 DataLoader 생성 헬퍼.
    - persistent_workers=True: epoch마다 worker 재생성 오버헤드 제거
    - prefetch_factor=4: GPU 대기 시간 감소
    """
    workers = 2 if is_audio else NUM_WORKERS
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        prefetch_factor=4 if workers > 0 else None,
        collate_fn=collate_fn_audio if is_audio else None,
    )


# =========================
# 5. Sequential Training (메모리 효율적)
# =========================
def train():
    prepare_dataset()

    # label_to_id 미리 로드
    print("\n🔄 Pre-loading label mappings...")
    temp_b = ImageDataset(os.path.join(WORK_DIR, "train", "behavior"), augment=False)
    temp_e = ImageDataset(os.path.join(WORK_DIR, "train", "emotion"),  augment=False)
    temp_s = AudioDataset(os.path.join(WORK_DIR, "train", "sound"),    augment=False)
    temp_p = PatellaDataset(os.path.join(WORK_DIR, "train", "patella"), augment=False)

    behavior_label_to_id = temp_b.label_to_id
    emotion_label_to_id  = temp_e.label_to_id
    sound_label_to_id    = temp_s.label_to_id
    sound_id_to_label    = temp_s.id_to_label
    patella_label_to_id  = temp_p.label_to_id

    # Emotion class_weight
    emotion_labels_list = [temp_e.label_to_id[label] for _, label in temp_e.samples]
    emotion_class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(len(emotion_label_to_id)),
        y=emotion_labels_list
    )
    emotion_class_weights_tensor = torch.tensor(emotion_class_weights, dtype=torch.float)

    del temp_b, temp_e, temp_s, temp_p
    clear_memory()

    # 모델 초기화 (CPU에 먼저 생성)
    print("\n🔄 Initializing models...")
    behavior_model = BehaviorModel(len(behavior_label_to_id))
    emotion_model  = EmotionModel(len(emotion_label_to_id))
    patella_model  = PatellaModel(len(patella_label_to_id))
    audio_model    = AudioModel(len(sound_label_to_id), freeze_backbone=False)

    # Optimizers
    behavior_opt = torch.optim.AdamW(behavior_model.parameters(), lr=LR_VIDEO, weight_decay=0.01)
    emotion_opt  = torch.optim.AdamW(emotion_model.parameters(),  lr=LR_VIDEO, weight_decay=0.01)
    patella_opt  = torch.optim.AdamW(patella_model.parameters(),  lr=LR_VIDEO, weight_decay=0.01)
    audio_opt    = torch.optim.AdamW(audio_model.parameters(),    lr=LR_AUDIO, weight_decay=0.01)

    # Audio LR Warmup Scheduler
    _temp_sound = AudioDataset(os.path.join(WORK_DIR, "train", "sound"), augment=False)
    _approx_sound_steps = (len(_temp_sound) // BATCH_SIZE) * EPOCHS
    del _temp_sound
    audio_scheduler = get_linear_schedule_with_warmup(
        audio_opt,
        num_warmup_steps=100,
        num_training_steps=_approx_sound_steps
    )
    clear_memory()

    # Scalers
    video_scaler = torch.amp.GradScaler("cuda")
    audio_scaler = torch.amp.GradScaler("cuda")

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_emotion = nn.CrossEntropyLoss(
        weight=emotion_class_weights_tensor.to(DEVICE),
        label_smoothing=0.1
    )

    best_avg_acc = 0
    history = []

    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")

        loss_b, loss_e, loss_s, loss_p = 0, 0, 0, 0

        # ========== 1. Behavior ==========
        print(f"\n🐾 Training Behavior...")
        behavior_model.to(DEVICE)
        behavior_model.train()

        behavior_train  = ImageDataset(os.path.join(WORK_DIR, "train", "behavior"), augment=True)
        behavior_loader = make_loader(behavior_train, shuffle=True)

        for imgs, labels in tqdm(behavior_loader, desc="Behavior", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            behavior_opt.zero_grad()
            with torch.amp.autocast("cuda"):
                imgs, labels_a, labels_b, lam = mixup_data(imgs, labels)
                logits = behavior_model(imgs)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)

            video_scaler.scale(loss).backward()
            video_scaler.step(behavior_opt)
            video_scaler.update()

            loss_b += loss.item()

        loss_b /= len(behavior_loader)
        print(f"  → Avg Loss: {loss_b:.4f}")

        behavior_model.cpu()
        del behavior_train, behavior_loader
        clear_memory()

        # ========== 2. Emotion ==========
        print(f"\n😊 Training Emotion...")
        emotion_model.to(DEVICE)
        emotion_model.train()

        emotion_train  = ImageDataset(os.path.join(WORK_DIR, "train", "emotion"), augment=True)
        emotion_loader = make_loader(emotion_train, shuffle=True)

        for imgs, labels in tqdm(emotion_loader, desc="Emotion", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            emotion_opt.zero_grad()
            with torch.amp.autocast("cuda"):
                imgs, labels_a, labels_b, lam = mixup_data(imgs, labels)
                logits = emotion_model(imgs)
                loss = (lam * criterion_emotion(logits, labels_a)
                        + (1 - lam) * criterion_emotion(logits, labels_b))
                loss = loss * LOSS_WEIGHTS["emotion"]

            video_scaler.scale(loss).backward()
            video_scaler.step(emotion_opt)
            video_scaler.update()

            loss_e += loss.item()

        loss_e /= len(emotion_loader)
        print(f"  → Avg Loss: {loss_e:.4f}")

        emotion_model.cpu()
        del emotion_train, emotion_loader
        clear_memory()

        # ========== 3. Sound ==========
        print(f"\n🔊 Training Sound...")
        audio_model.to(DEVICE)
        audio_model.train()

        sound_train = AudioDataset(os.path.join(WORK_DIR, "train", "sound"), augment=True)

        sound_labels_list = [item[1] for item in sound_train.samples]
        sound_label_ids   = [sound_train.label_to_id[l] for l in sound_labels_list]
        class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(len(sound_train.label_to_id)),
            y=sound_label_ids
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

        sound_loader = make_loader(sound_train, shuffle=True, is_audio=True)

        for batch in tqdm(sound_loader, desc="Sound", leave=False):
            audios = batch["input_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            audio_opt.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = audio_model(input_values=audios, labels=labels)
                loss = outputs.loss * LOSS_WEIGHTS["sound"]
                per_sample_w = class_weights_tensor[labels]
                loss = loss * per_sample_w.mean()

            audio_scaler.scale(loss).backward()
            audio_scaler.unscale_(audio_opt)
            torch.nn.utils.clip_grad_norm_(audio_model.parameters(), 1.0)
            audio_scaler.step(audio_opt)
            audio_scaler.update()
            audio_scheduler.step()

            loss_s += loss.item()

        loss_s /= len(sound_loader)
        print(f"  → Avg Loss: {loss_s:.4f}")

        audio_model.cpu()
        del sound_train, sound_loader, class_weights_tensor
        clear_memory()

        # ========== 4. Patella ==========
        print(f"\n🦴 Training Patella...")
        patella_model.to(DEVICE)
        patella_model.train()

        patella_train  = PatellaDataset(os.path.join(WORK_DIR, "train", "patella"), augment=True)
        patella_loader = make_loader(patella_train, shuffle=True)

        for imgs, keypoints, labels in tqdm(patella_loader, desc="Patella", leave=False):
            imgs, keypoints, labels = imgs.to(DEVICE), keypoints.to(DEVICE), labels.to(DEVICE)

            patella_opt.zero_grad()
            with torch.amp.autocast("cuda"):
                imgs, labels_a, labels_b, lam = mixup_data(imgs, labels)
                logits = patella_model(imgs, keypoints)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)

            video_scaler.scale(loss).backward()
            video_scaler.step(patella_opt)
            video_scaler.update()

            loss_p += loss.item()

        loss_p /= len(patella_loader)
        print(f"  → Avg Loss: {loss_p:.4f}")

        patella_model.cpu()
        del patella_train, patella_loader
        clear_memory()

        # ========== Validation ==========
        print(f"\n🔍 Validation...")

        # Behavior Val
        behavior_model.to(DEVICE)
        behavior_model.eval()
        behavior_val        = ImageDataset(os.path.join(WORK_DIR, "val", "behavior"), augment=False)
        behavior_val_loader = make_loader(behavior_val, shuffle=False)

        correct_b, total_b = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(behavior_val_loader, desc="Val Behavior", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = behavior_model(imgs)
                pred = logits.argmax(-1)
                correct_b += (pred == labels).sum().item()
                total_b   += labels.size(0)
        acc_b = correct_b / total_b

        behavior_model.cpu()
        del behavior_val, behavior_val_loader
        clear_memory()

        # Emotion Val
        emotion_model.to(DEVICE)
        emotion_model.eval()
        emotion_val        = ImageDataset(os.path.join(WORK_DIR, "val", "emotion"), augment=False)
        emotion_val_loader = make_loader(emotion_val, shuffle=False)

        correct_e, total_e = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(emotion_val_loader, desc="Val Emotion", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = emotion_model(imgs)
                pred = logits.argmax(-1)
                correct_e += (pred == labels).sum().item()
                total_e   += labels.size(0)
        acc_e = correct_e / total_e

        emotion_model.cpu()
        del emotion_val, emotion_val_loader
        clear_memory()

        # Sound Val
        audio_model.to(DEVICE)
        audio_model.eval()
        sound_val        = AudioDataset(os.path.join(WORK_DIR, "val", "sound"), augment=False)
        sound_val_loader = make_loader(sound_val, shuffle=False, is_audio=True)

        correct_s, total_s = 0, 0
        with torch.no_grad():
            for batch in tqdm(sound_val_loader, desc="Val Sound", leave=False):
                audios = batch["input_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                outputs = audio_model(input_values=audios, labels=labels)
                pred = outputs.logits.argmax(-1)
                correct_s += (pred == labels).sum().item()
                total_s   += labels.size(0)
        acc_s = correct_s / total_s

        audio_model.cpu()
        del sound_val, sound_val_loader
        clear_memory()

        # Patella Val
        patella_model.to(DEVICE)
        patella_model.eval()
        patella_val        = PatellaDataset(os.path.join(WORK_DIR, "val", "patella"), augment=False)
        patella_val_loader = make_loader(patella_val, shuffle=False)

        correct_p, total_p = 0, 0
        with torch.no_grad():
            for imgs, keypoints, labels in tqdm(patella_val_loader, desc="Val Patella", leave=False):
                imgs, keypoints, labels = imgs.to(DEVICE), keypoints.to(DEVICE), labels.to(DEVICE)
                logits = patella_model(imgs, keypoints)
                pred = logits.argmax(-1)
                correct_p += (pred == labels).sum().item()
                total_p   += labels.size(0)
        acc_p = correct_p / total_p

        patella_model.cpu()
        del patella_val, patella_val_loader
        clear_memory()

        avg_acc = (acc_b + acc_e + acc_s + acc_p) / 4

        print(f"\n📊 Results:")
        print(f"  Behavior: Loss {loss_b:.4f} | Acc {acc_b:.4f} ({acc_b*100:.1f}%)")
        print(f"  Emotion:  Loss {loss_e:.4f} | Acc {acc_e:.4f} ({acc_e*100:.1f}%)")
        print(f"  Sound:    Loss {loss_s:.4f} | Acc {acc_s:.4f} ({acc_s*100:.1f}%)")
        print(f"  Patella:  Loss {loss_p:.4f} | Acc {acc_p:.4f} ({acc_p*100:.1f}%)")
        print(f"  Average Acc: {avg_acc:.4f} ({avg_acc*100:.1f}%)")

        history.append({
            'epoch' : epoch + 1,
            'loss_b': loss_b, 'loss_e': loss_e,
            'loss_s': loss_s, 'loss_p': loss_p,
            'acc_b' : acc_b,  'acc_e' : acc_e,
            'acc_s' : acc_s,  'acc_p' : acc_p,
            'acc_avg': avg_acc,
        })

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            torch.save({
                "behavior_model":       behavior_model.state_dict(),
                "emotion_model":        emotion_model.state_dict(),
                "audio_model":          audio_model.state_dict(),
                "patella_model":        patella_model.state_dict(),
                "behavior_label_to_id": behavior_label_to_id,
                "emotion_label_to_id":  emotion_label_to_id,
                "sound_label_to_id":    sound_label_to_id,
                "sound_id_to_label":    sound_id_to_label,
                "patella_label_to_id":  patella_label_to_id,
                "best_epoch":           epoch + 1,
                "best_acc":             best_avg_acc,
                "history":              history,
            }, "pet_normal_omni_best.pth")
            print(f"  💾 Saved new best model! (Acc: {best_avg_acc:.4f})")

    # 학습 곡선 시각화
    print("\n📈 Generating training history plot...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    tasks = [
        ('acc_b', 'loss_b', 'b-',     'Behavior'),
        ('acc_e', 'loss_e', 'r-',     'Emotion'),
        ('acc_s', 'loss_s', 'g-',     'Sound'),
        ('acc_p', 'loss_p', 'purple', 'Patella'),
    ]

    for i, (acc_key, loss_key, color, title) in enumerate(tasks):
        axes[0, i].plot([h[loss_key] for h in history], color=color, linewidth=2)
        axes[0, i].set_title(f'{title} Loss')
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].grid(True, alpha=0.3)

        axes[1, i].plot([h[acc_key] for h in history], color=color, linewidth=2)
        axes[1, i].set_title(f'{title} Accuracy')
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Accuracy')
        axes[1, i].set_ylim(0, 1)
        axes[1, i].grid(True, alpha=0.3)

    plt.suptitle('Pet Normal Omni Model Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pet_omni_sequential_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: pet_omni_sequential_history.png")

    print(f"\n🎉 Training Finished!")
    print(f"  Best Average Acc: {best_avg_acc:.4f} ({best_avg_acc*100:.1f}%)")


if __name__ == "__main__":
    train()
