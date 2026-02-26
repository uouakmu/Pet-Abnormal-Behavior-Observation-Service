import os
import gc
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===============================
# CONFIG
# ===============================

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS                = 50
BATCH_SIZE            = 32
NUM_WORKERS           = 24
LR                    = 1e-4
NUM_IMAGES_PER_SAMPLE = 5      # 사용자가 업로드하는 사진 수
LABEL_SMOOTHING       = 0.1

# train 80% / val 10% / test 10%
VAL_RATIO  = 0.1
TEST_RATIO = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# CLASS DEFINITIONS
# 규칙: dog_ 접두 → dog classes, cat_ 접두 → cat classes
# ─────────────────────────────────────────────────────────────────────────────

# ── 4_Animal_Skin ──
SKIN_CLASSES = [
    "cat_normal", "cat_결절,종괴", "cat_농포,여드름",
    "cat_비듬,각질,상피성잔고리", "dog_normal",
    "dog_결절,종괴", "dog_농포,여드름", "dog_미란,궤양",
    "dog_비듬,각질,상피성잔고리",
]

# ── 5_Animal_Eyes ──
EYES_CLASSES = [
    "cat_normal", "cat_각막궤양", "cat_각막부골편",
    "cat_결막염", "cat_비궤양성각막염", "cat_안검염",
    "dog_normal", "dog_결막염", "dog_궤양성각막질환_상",
    "dog_궤양성각막질환_하", "dog_백내장_비성숙", "dog_백내장_성숙",
    "dog_백내장_초기", "dog_비궤양성각막질환_상", "dog_비궤양성각막질환_하",
    "dog_색소침착성각막염", "dog_안검내반증", "dog_안검염",
    "dog_안검종양", "dog_유루증", "dog_핵경화",
]

# ─────────────────────────────────────────────────────────────────────────────
# 유사 클래스 그룹 정의 (Eyes 전용)
# 동일 질환 내 세분류는 Hierarchical Loss 가중치로 혼동 패널티를 줌
# ─────────────────────────────────────────────────────────────────────────────
EYES_SIMILAR_GROUPS = [
    ["dog_비궤양성각막질환_상", "dog_비궤양성각막질환_하"],
    ["dog_궤양성각막질환_상",   "dog_궤양성각막질환_하"],
    ["dog_백내장_초기", "dog_백내장_비성숙", "dog_백내장_성숙"],
]


# ===============================
# LOSS: Hierarchical-Aware CE
# ===============================

class HierarchicalWeightedLoss(nn.Module):
    """
    CrossEntropyLoss + Label Smoothing + 유사 클래스 혼동 페널티

    Args:
        class_names    : 학습 task에 해당하는 클래스 이름 리스트
        similar_groups : 유사 클래스 묶음 [[cls_a, cls_b], ...]
        class_weights  : 클래스 불균형 보정 weight 텐서
        smoothing      : label smoothing ε
        extra_penalty  : 같은 그룹 내 오분류 시 loss 배율
    """

    def __init__(
        self,
        class_names,
        similar_groups=None,
        class_weights=None,
        smoothing=LABEL_SMOOTHING,
        extra_penalty=1.5,
    ):
        super().__init__()
        self.smoothing     = smoothing
        self.extra_penalty = extra_penalty
        self.num_classes   = len(class_names)
        self.class_names   = class_names
        self.name_to_idx   = {n: i for i, n in enumerate(class_names)}

        # 유사 그룹 → (idx_i, idx_j) pair set
        self.penalty_pairs = set()
        if similar_groups:
            for group in similar_groups:
                idxs = [self.name_to_idx[n] for n in group if n in self.name_to_idx]
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        self.penalty_pairs.add((idxs[i], idxs[j]))
                        self.penalty_pairs.add((idxs[j], idxs[i]))

        self.register_buffer("weight", class_weights)

    def forward(self, logits, targets):
        B, C   = logits.shape
        device = logits.device

        # ── Label Smoothing ──
        log_prob    = F.log_softmax(logits, dim=-1)
        smooth_loss = -log_prob.mean(dim=-1)                                               # (B,)
        nll_loss    = F.nll_loss(log_prob, targets, weight=self.weight, reduction="none")  # (B,)
        base_loss   = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss       # (B,)

        # ── Hierarchical Penalty ──
        if self.penalty_pairs:
            pred_classes = logits.argmax(dim=-1)
            penalty_mask = torch.ones(B, device=device)
            for b in range(B):
                t = targets[b].item()
                p = pred_classes[b].item()
                if (t, p) in self.penalty_pairs:
                    penalty_mask[b] = self.extra_penalty
            base_loss = base_loss * penalty_mask

        return base_loss.mean()


# ===============================
# CLASS WEIGHT COMPUTATION
# ===============================

def compute_class_weights(sample_counts: dict, class_names: list) -> torch.Tensor:
    """Inverse-frequency 방식으로 클래스 가중치를 계산한다."""
    counts  = torch.tensor([sample_counts.get(n, 1) for n in class_names], dtype=torch.float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(class_names)   # normalize
    return weights


# ===============================
# MODEL DEFINITIONS
# ===============================

class AnomalyMultiBackbone(nn.Module):
    """
    이상 증상 Omni 모델
    ├── skin_backbone  → Skin 분류 (피부질환)
    └── eyes_backbone  → Eyes 분류 (안구질환)
    """

    def __init__(self, num_skin_classes: int, num_eyes_classes: int):
        super().__init__()

        # ── Skin Backbone (ResNet50 pretrained) ──────────────────────────────
        skin_base          = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        skin_feat_dim      = skin_base.fc.in_features   # 2048
        skin_base.fc       = nn.Identity()
        self.skin_backbone = skin_base
        self.skin_head = nn.Sequential(
            nn.Linear(skin_feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_skin_classes),
        )

        # ── Eyes Backbone (ResNet50 pretrained + SE attention) ───────────────
        eyes_base          = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        eyes_feat_dim      = eyes_base.fc.in_features
        eyes_base.fc       = nn.Identity()
        self.eyes_backbone = eyes_base
        self.eyes_se       = SqueezeExcitation(eyes_feat_dim, reduction=16)
        self.eyes_head = nn.Sequential(
            nn.Linear(eyes_feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_eyes_classes),
        )

    def forward(self, x: torch.Tensor, task: str = "skin") -> torch.Tensor:
        if task == "skin":
            return self.skin_head(self.skin_backbone(x))
        elif task == "eyes":
            feat = self.eyes_backbone(x)
            feat = self.eyes_se(feat)
            return self.eyes_head(feat)
        else:
            raise ValueError(f"Unknown task: {task!r}. Choose 'skin' or 'eyes'.")


class SqueezeExcitation(nn.Module):
    """1-D Squeeze-Excitation for feature vectors (after global avg pool)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


# ===============================
# INFERENCE: 5-Image Ensemble
# ===============================

def predict_anomaly(
    model: AnomalyMultiBackbone,
    images: list,
    task: str,
    pet_type: str,
    class_names: list,
    device=DEVICE,
) -> dict:
    """
    5장의 이미지를 입력받아 평균 softmax 확률로 최종 예측을 반환한다.

    Returns:
        {"predicted_class": str, "confidence": float, "top3": [(class_name, prob), ...]}
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model.eval()
    model.to(device)

    valid_idxs  = [i for i, n in enumerate(class_names) if n.startswith(pet_type + "_")]
    valid_names = [class_names[i] for i in valid_idxs]

    with torch.no_grad():
        probs_accum = torch.zeros(len(class_names), device=device)
        for img in images:
            tensor = transform(img).unsqueeze(0).to(device)
            logits = model(tensor, task=task)
            mask   = torch.full((len(class_names),), float("-inf"), device=device)
            mask[valid_idxs] = logits[0][valid_idxs]
            probs_accum += F.softmax(mask, dim=-1)
        probs_accum /= len(images)

    valid_probs = [(valid_names[i], probs_accum[valid_idxs[i]].item()) for i in range(len(valid_idxs))]
    valid_probs.sort(key=lambda x: x[1], reverse=True)

    return {
        "predicted_class": valid_probs[0][0],
        "confidence":      valid_probs[0][1],
        "top3":            valid_probs[:3],
    }


# ===============================
# DATA SPLIT UTILITY
# ===============================

def collect_and_split(
    root_dir: str,
    class_names: list,
    val_ratio: float  = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int         = SEED,
):
    """
    root_dir 하위 class 디렉토리에서 이미지를 수집하고
    클래스별 stratified split으로 train / val / test 를 반환한다.

    [데이터 누수 방지]
    - 파일 경로 중복 제거 (seen set)
    - 클래스별로 독립 shuffle 후 비율 분리
      → train / val / test 간 동일 파일 절대 미포함

    Returns:
        train_samples, val_samples, test_samples
        각 원소: (img_path: str, label_idx: int)
    """
    rng         = random.Random(seed)
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    class_files = defaultdict(list)
    seen_paths  = set()

    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        label_idx = name_to_idx[class_name]
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(class_dir, fname)
            if fpath in seen_paths:      # 중복 파일 제거
                continue
            seen_paths.add(fpath)
            class_files[label_idx].append(fpath)

    train_samples, val_samples, test_samples = [], [], []

    for label_idx, paths in class_files.items():
        rng.shuffle(paths)
        n       = len(paths)
        n_val   = max(1, int(n * val_ratio))
        n_test  = max(1, int(n * test_ratio))
        n_train = n - n_val - n_test

        # 샘플 수가 너무 적은 클래스 경고
        if n_train <= 0:
            print(f"  ⚠️  클래스 idx={label_idx}: 샘플 수({n})가 너무 적어 train이 0개입니다.")
            n_train, n_val, n_test = n, 0, 0

        train_samples.extend([(p, label_idx) for p in paths[:n_train]])
        val_samples.extend(  [(p, label_idx) for p in paths[n_train:n_train + n_val]])
        test_samples.extend( [(p, label_idx) for p in paths[n_train + n_val:]])

    print(f"  → train: {len(train_samples)} | val: {len(val_samples)} | test: {len(test_samples)}")
    return train_samples, val_samples, test_samples


def count_samples_from_split(samples: list, class_names: list) -> dict:
    """split된 samples에서 class_name별 개수를 반환 (class_weight 계산용)."""
    idx_to_name = {i: n for i, n in enumerate(class_names)}
    counts      = defaultdict(int)
    for _, label_idx in samples:
        counts[idx_to_name[label_idx]] += 1
    return dict(counts)


# ===============================
# DATASETS
# ===============================

class AnomalyDataset(Dataset):
    """
    collect_and_split() 결과를 받아 Dataset으로 래핑한다.

    samples  : [(img_path, label_idx), ...]
    is_train : True  → augmentation 적용
               False → resize only (val / test)
    """

    TRANSFORM_TRAIN = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    TRANSFORM_VAL = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, samples: list, is_train: bool = True):
        self.samples   = samples
        self.transform = self.TRANSFORM_TRAIN if is_train else self.TRANSFORM_VAL

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label


# ===============================
# TRAIN FUNCTION
# ===============================

def train(
    skin_root: str = "files/4_Animal_Skin",
    eyes_root: str = "files/5_Animal_Eyes",
):
    print(f"🎯 Device: {DEVICE}")

    skin_classes = SKIN_CLASSES
    eyes_classes = EYES_CLASSES

    # ── Train / Val / Test Split ───────────────────────────────────────────────
    # 클래스별 stratified split → 누수 없음
    print("\n📦 Splitting Skin dataset...")
    skin_train_samples, skin_val_samples, _ = collect_and_split(skin_root, skin_classes)

    print("\n📦 Splitting Eyes dataset...")
    eyes_train_samples, eyes_val_samples, _ = collect_and_split(eyes_root, eyes_classes)

    # ── 클래스 가중치: train split 기준으로만 계산 (val/test 정보 누수 방지) ──
    skin_train_counts = count_samples_from_split(skin_train_samples, skin_classes)
    eyes_train_counts = count_samples_from_split(eyes_train_samples, eyes_classes)

    skin_weights = compute_class_weights(skin_train_counts, skin_classes).to(DEVICE)
    eyes_weights = compute_class_weights(eyes_train_counts, eyes_classes).to(DEVICE)

    # ── Loss ───────────────────────────────────────────────────────────────────
    skin_criterion = HierarchicalWeightedLoss(
        class_names   = skin_classes,
        class_weights = skin_weights,
        smoothing     = LABEL_SMOOTHING,
    )
    eyes_criterion = HierarchicalWeightedLoss(
        class_names    = eyes_classes,
        similar_groups = EYES_SIMILAR_GROUPS,
        class_weights  = eyes_weights,
        smoothing      = LABEL_SMOOTHING,
        extra_penalty  = 1.5,
    )

    # ── 모델 / Optimizer / Scheduler ──────────────────────────────────────────
    model     = AnomalyMultiBackbone(len(skin_classes), len(eyes_classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler()

    # ── 학습 기록 & Best 추적 ──────────────────────────────────────────────────
    history      = []
    best_avg_acc = 0.0
    best_epoch   = 0

    # ── Training Loop ──────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        print(f"\n{'='*55}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*55}")

        # ──────────────────────────────────────────────────────────────────────
        # 1. Skin Training
        # ──────────────────────────────────────────────────────────────────────
        print("\n[Train 1/2] Skin")
        model.train()

        skin_train_ds     = AnomalyDataset(skin_train_samples, is_train=True)
        skin_train_loader = DataLoader(
            skin_train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0), prefetch_factor=4,
        )

        skin_loss_sum, skin_correct, skin_total = 0.0, 0, 0
        for images, labels in tqdm(skin_train_loader, desc=f"  Skin Train Ep{epoch+1:02d}", ncols=110, leave=True):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images, task="skin")
                loss    = skin_criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            skin_loss_sum += loss.item() * images.size(0)
            skin_correct  += (outputs.argmax(1) == labels).sum().item()
            skin_total    += images.size(0)

        skin_train_loss = skin_loss_sum / skin_total
        skin_train_acc  = skin_correct  / skin_total

        del skin_train_ds, skin_train_loader
        gc.collect(); torch.cuda.empty_cache()

        # ──────────────────────────────────────────────────────────────────────
        # 2. Eyes Training
        # ──────────────────────────────────────────────────────────────────────
        print("\n[Train 2/2] Eyes")

        eyes_train_ds     = AnomalyDataset(eyes_train_samples, is_train=True)
        eyes_train_loader = DataLoader(
            eyes_train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0), prefetch_factor=4,
        )

        eyes_loss_sum, eyes_correct, eyes_total = 0.0, 0, 0
        for images, labels in tqdm(eyes_train_loader, desc=f"  Eyes Train Ep{epoch+1:02d}", ncols=110, leave=True):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images, task="eyes")
                loss    = eyes_criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            eyes_loss_sum += loss.item() * images.size(0)
            eyes_correct  += (outputs.argmax(1) == labels).sum().item()
            eyes_total    += images.size(0)

        eyes_train_loss = eyes_loss_sum / eyes_total
        eyes_train_acc  = eyes_correct  / eyes_total

        del eyes_train_ds, eyes_train_loader
        gc.collect(); torch.cuda.empty_cache()

        # LR Scheduler Step
        scheduler.step()

        # ──────────────────────────────────────────────────────────────────────
        # 3. Validation  ← [수정] 추가: val acc 기준으로 best model 저장
        # ──────────────────────────────────────────────────────────────────────
        print("\n[Val] Skin & Eyes")
        model.eval()

        # Skin Val
        skin_val_ds     = AnomalyDataset(skin_val_samples, is_train=False)
        skin_val_loader = DataLoader(
            skin_val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS // 2, pin_memory=True,
            persistent_workers=(NUM_WORKERS // 2 > 0), prefetch_factor=4,
        )

        skin_val_loss_sum, skin_val_correct, skin_val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(skin_val_loader, desc="  Skin Val  ", ncols=110, leave=False):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    outputs = model(images, task="skin")
                    loss    = skin_criterion(outputs, labels)
                skin_val_loss_sum += loss.item() * images.size(0)
                skin_val_correct  += (outputs.argmax(1) == labels).sum().item()
                skin_val_total    += images.size(0)

        skin_val_loss = skin_val_loss_sum / skin_val_total
        skin_val_acc  = skin_val_correct  / skin_val_total

        del skin_val_ds, skin_val_loader
        gc.collect(); torch.cuda.empty_cache()

        # Eyes Val
        eyes_val_ds     = AnomalyDataset(eyes_val_samples, is_train=False)
        eyes_val_loader = DataLoader(
            eyes_val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS // 2, pin_memory=True,
            persistent_workers=(NUM_WORKERS // 2 > 0), prefetch_factor=4,
        )

        eyes_val_loss_sum, eyes_val_correct, eyes_val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(eyes_val_loader, desc="  Eyes Val  ", ncols=110, leave=False):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    outputs = model(images, task="eyes")
                    loss    = eyes_criterion(outputs, labels)
                eyes_val_loss_sum += loss.item() * images.size(0)
                eyes_val_correct  += (outputs.argmax(1) == labels).sum().item()
                eyes_val_total    += images.size(0)

        eyes_val_loss = eyes_val_loss_sum / eyes_val_total
        eyes_val_acc  = eyes_val_correct  / eyes_val_total

        del eyes_val_ds, eyes_val_loader
        gc.collect(); torch.cuda.empty_cache()

        # ── 결과 출력 ──────────────────────────────────────────────────────────
        avg_val_acc = (skin_val_acc + eyes_val_acc) / 2

        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"  Skin │ Train  Loss: {skin_train_loss:.4f}  Acc: {skin_train_acc*100:.2f}%"
              f"  │  Val Loss: {skin_val_loss:.4f}  Acc: {skin_val_acc*100:.2f}%")
        print(f"  Eyes │ Train  Loss: {eyes_train_loss:.4f}  Acc: {eyes_train_acc*100:.2f}%"
              f"  │  Val Loss: {eyes_val_loss:.4f}  Acc: {eyes_val_acc*100:.2f}%")
        print(f"  Avg Val Acc: {avg_val_acc*100:.2f}%")

        # ── History 기록 ────────────────────────────────────────────────────────
        history.append({
            'epoch'          : epoch + 1,
            'skin_train_loss': skin_train_loss,
            'skin_train_acc' : skin_train_acc,
            'skin_val_loss'  : skin_val_loss,
            'skin_val_acc'   : skin_val_acc,
            'eyes_train_loss': eyes_train_loss,
            'eyes_train_acc' : eyes_train_acc,
            'eyes_val_loss'  : eyes_val_loss,
            'eyes_val_acc'   : eyes_val_acc,
            'avg_val_acc'    : avg_val_acc,
        })

        # ── Best Model 저장: val acc 기준 ─────────────────────────────────────
        # [수정] 기존: train acc 기준 → 과적합 모델이 저장될 위험
        #        변경: val acc 기준  → 실제 일반화 성능이 가장 좋은 모델 저장
        if avg_val_acc > best_avg_acc:
            best_avg_acc = avg_val_acc
            best_epoch   = epoch + 1
            torch.save(
                {
                    "model"        : model.state_dict(),
                    "epoch"        : epoch + 1,
                    "best_avg_acc" : best_avg_acc,
                    "skin_classes" : SKIN_CLASSES,
                    "eyes_classes" : EYES_CLASSES,
                    "history"      : history,
                },
                "pet_abnormal_omni_best.pth",
            )
            print(f"  💾 Saved best model! (Epoch {best_epoch} | Val Avg Acc: {best_avg_acc*100:.2f}%)")

    print(f"\n🏆 Training Finished.")
    print(f"   Best Epoch: {best_epoch} | Best Val Avg Acc: {best_avg_acc*100:.2f}%")

    # ── 학습 곡선 시각화 ──────────────────────────────────────────────────────
    print("\n📈 Generating training history plot...")

    epochs_x        = [h['epoch']           for h in history]
    skin_tr_losses  = [h['skin_train_loss']  for h in history]
    skin_val_losses = [h['skin_val_loss']    for h in history]
    eyes_tr_losses  = [h['eyes_train_loss']  for h in history]
    eyes_val_losses = [h['eyes_val_loss']    for h in history]
    skin_tr_accs    = [h['skin_train_acc']   for h in history]
    skin_val_accs   = [h['skin_val_acc']     for h in history]
    eyes_tr_accs    = [h['eyes_train_acc']   for h in history]
    eyes_val_accs   = [h['eyes_val_acc']     for h in history]
    avg_val_accs    = [h['avg_val_acc']      for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # ─ Loss ─
    axes[0].plot(epochs_x, skin_tr_losses,  'b-',  linewidth=2, label='Skin Train Loss')
    axes[0].plot(epochs_x, skin_val_losses, 'b--', linewidth=2, label='Skin Val Loss')
    axes[0].plot(epochs_x, eyes_tr_losses,  'r-',  linewidth=2, label='Eyes Train Loss')
    axes[0].plot(epochs_x, eyes_val_losses, 'r--', linewidth=2, label='Eyes Val Loss')
    axes[0].axvline(best_epoch, color='gray', linestyle=':', alpha=0.7, label=f'Best Epoch {best_epoch}')
    axes[0].set_title('Loss');    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend();             axes[0].grid(True, alpha=0.3)

    # ─ Accuracy ─
    axes[1].plot(epochs_x, skin_tr_accs,  'b-',  linewidth=2, label='Skin Train Acc')
    axes[1].plot(epochs_x, skin_val_accs, 'b--', linewidth=2, label='Skin Val Acc')
    axes[1].plot(epochs_x, eyes_tr_accs,  'r-',  linewidth=2, label='Eyes Train Acc')
    axes[1].plot(epochs_x, eyes_val_accs, 'r--', linewidth=2, label='Eyes Val Acc')
    axes[1].axvline(best_epoch, color='gray', linestyle=':', alpha=0.7, label=f'Best Epoch {best_epoch}')
    axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1);        axes[1].legend();            axes[1].grid(True, alpha=0.3)

    # ─ Avg Val Accuracy ─
    axes[2].plot(epochs_x, avg_val_accs, 'g-', linewidth=2, label='Avg Val Acc')
    axes[2].axvline(best_epoch, color='gray', linestyle=':', alpha=0.7, label=f'Best Epoch {best_epoch}')
    axes[2].axhline(best_avg_acc, color='green', linestyle='--', alpha=0.6,
                    label=f'Best Val Acc {best_avg_acc*100:.1f}%')
    axes[2].set_title('Avg Val Accuracy'); axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim(0, 1);                axes[2].legend();             axes[2].grid(True, alpha=0.3)

    plt.suptitle('Anomaly Model Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('anomaly_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: anomaly_training_history.png")


# ===============================
# ENTRY POINT
# ===============================

if __name__ == "__main__":
    train()
