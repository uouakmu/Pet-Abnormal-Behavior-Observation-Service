import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===============================
# CONFIG
# ===============================

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

EPOCHS = 50
BATCH_SIZE = 32
NUM_WORKERS = 24
LR = 1e-4
NUM_IMAGES_PER_SAMPLE = 5          # 사용자가 업로드하는 사진 수
LABEL_SMOOTHING = 0.1

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
    "dog_안검종양", "dog_유루증", "dog_핵경화"
]

# ─────────────────────────────────────────────────────────────────────────────
# 유사 클래스 그룹 정의 (Eyes 전용)
# 동일 질환 내 세분류는 Hierarchical Loss 가중치로 혼동 패널티를 줌
# ─────────────────────────────────────────────────────────────────────────────
EYES_SIMILAR_GROUPS = [
    ["dog_비궤양성각막질환_상", "dog_비궤양성각막질환_하"],
    ["dog_궤양성각막질환_상", "dog_궤양성각막질환_하"],
    ["dog_백내장_초기", "dog_백내장_비성숙", "dog_백내장_성숙"],
]


# ===============================
# LOSS: Hierarchical-Aware CE
# ===============================
# 같은 질환 그룹 내 오분류에 extra_penalty 를 곱해
# 모델이 상/하, 초기/성숙 구분을 더 열심히 학습하게 만든다.

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
        self.smoothing      = smoothing
        self.extra_penalty  = extra_penalty
        self.num_classes    = len(class_names)
        self.class_names    = class_names
        self.name_to_idx    = {n: i for i, n in enumerate(class_names)}

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
        """
        logits  : (B, C)
        targets : (B,)  long
        """
        B, C = logits.shape
        device = logits.device

        # ── Label Smoothing ──
        log_prob = F.log_softmax(logits, dim=-1)
        smooth_loss = -log_prob.mean(dim=-1)                              # (B,)
        nll_loss    = F.nll_loss(log_prob, targets, weight=self.weight, reduction="none")  # (B,)
        base_loss   = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss  # (B,)

        # ── Hierarchical Penalty ──
        if self.penalty_pairs:
            pred_classes = logits.argmax(dim=-1)          # (B,)
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
    """
    Inverse-frequency 방식으로 클래스 가중치를 계산한다.
    sample_counts: {class_name: n_samples}
    """
    counts = torch.tensor(
        [sample_counts.get(n, 1) for n in class_names], dtype=torch.float
    )
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

    각 backbone 은 ResNet50 (ImageNet pretrained) 을 기반으로 하며,
    마지막 fc 를 task-specific head 로 교체한다.

    Eyes 의 경우 유사 클래스 혼동을 줄이기 위해:
      1) Dropout + 더 깊은 head
      2) Feature Attention (Channel Squeeze-Excitation)
    을 추가한다.
    """

    def __init__(self, num_skin_classes: int, num_eyes_classes: int):
        super().__init__()

        # ── Skin Backbone (ResNet50 pretrained) ──────────────────────────────
        skin_base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        skin_feat_dim = skin_base.fc.in_features          # 2048
        skin_base.fc = nn.Identity()
        self.skin_backbone = skin_base
        self.skin_head = nn.Sequential(
            nn.Linear(skin_feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_skin_classes),
        )

        # ── Eyes Backbone (ResNet50 pretrained + SE attention) ───────────────
        eyes_base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        eyes_feat_dim = eyes_base.fc.in_features
        eyes_base.fc = nn.Identity()
        self.eyes_backbone = eyes_base

        # Squeeze-Excitation: 채널 중요도 재보정 → 미세한 병변 구분력 향상
        self.eyes_se = SqueezeExcitation(eyes_feat_dim, reduction=16)

        # 더 깊은 classifier head
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
        """
        x    : (B, 3, 224, 224)  — 단일 이미지 또는 앙상블 후 평균 logit 용
        task : "skin" | "eyes"
        """
        if task == "skin":
            feat = self.skin_backbone(x)
            return self.skin_head(feat)

        elif task == "eyes":
            feat = self.eyes_backbone(x)           # (B, 2048)
            feat = self.eyes_se(feat)              # channel attention
            return self.eyes_head(feat)

        else:
            raise ValueError(f"Unknown task: {task!r}. Choose 'skin' or 'eyes'.")


class SqueezeExcitation(nn.Module):
    """
    1-D Squeeze-Excitation for feature vectors (after global avg pool).
    feat : (B, C)
    """

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
    images: list,           # list of PIL.Image (5장)
    task: str,              # "skin" | "eyes"
    pet_type: str,          # "dog" | "cat"
    class_names: list,
    device=DEVICE,
) -> dict:
    """
    5장의 이미지를 입력받아 평균 softmax 확률로 최종 예측을 반환한다.

    Returns:
        {
            "predicted_class": str,
            "confidence": float,
            "top3": [(class_name, prob), ...]
        }
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    model.eval()
    model.to(device)

    # 반려동물 종에 맞는 class index만 선택
    valid_idxs = [
        i for i, n in enumerate(class_names) if n.startswith(pet_type + "_")
    ]
    valid_names = [class_names[i] for i in valid_idxs]

    with torch.no_grad():
        probs_accum = torch.zeros(len(class_names), device=device)

        for img in images:
            tensor = transform(img).unsqueeze(0).to(device)   # (1, 3, 224, 224)
            logits = model(tensor, task=task)                  # (1, C)

            # 해당 pet_type 외 class 마스킹 (−inf → softmax ≈ 0)
            mask = torch.full((len(class_names),), float("-inf"), device=device)
            mask[valid_idxs] = logits[0][valid_idxs]

            probs = F.softmax(mask, dim=-1)
            probs_accum += probs

        probs_accum /= len(images)    # 평균 앙상블

    # 유효 class 중 top-k
    valid_probs = [(valid_names[i], probs_accum[valid_idxs[i]].item())
                   for i in range(len(valid_idxs))]
    valid_probs.sort(key=lambda x: x[1], reverse=True)

    return {
        "predicted_class": valid_probs[0][0],
        "confidence":      valid_probs[0][1],
        "top3":            valid_probs[:3],
    }


# ===============================
# DATASETS
# ===============================

class AnomalyDataset(Dataset):
    """
    데이터셋 구조:
        root_dir/
            dog_결막염/  img001.jpg ...
            cat_normal/  img001.jpg ...
            ...

    task      : "skin" | "eyes"
    pet_type  : "dog" | "cat" | "all"
    """

    TRANSFORM = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    TRANSFORM_VAL = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def __init__(
        self,
        root_dir: str,
        class_names: list,
        task: str,
        is_train: bool = True,
    ):
        self.class_names = class_names
        self.task        = task
        self.transform   = self.TRANSFORM if is_train else self.TRANSFORM_VAL
        self.name_to_idx = {n: i for i, n in enumerate(class_names)}

        self.samples = []   # [(img_path, label_idx), ...]

        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label_idx = self.name_to_idx[class_name]
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label

    @staticmethod
    def get_sample_counts(root_dir: str, class_names: list) -> dict:
        counts = {}
        for cn in class_names:
            d = os.path.join(root_dir, cn)
            if os.path.isdir(d):
                counts[cn] = len([
                    f for f in os.listdir(d)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
            else:
                counts[cn] = 1
        return counts


# ===============================
# TRAIN FUNCTION
# ===============================

def train(
    skin_root: str = "files/4_Animal_Skin",
    eyes_root: str = "files/5_Animal_Eyes",
):
    # ── 클래스 정의 ────────────────────────────────────────────────────────────
    skin_classes = SKIN_CLASSES
    eyes_classes = EYES_CLASSES

    num_skin  = len(skin_classes)
    num_eyes  = len(eyes_classes)

    # ── 모델 초기화 ────────────────────────────────────────────────────────────
    model = AnomalyMultiBackbone(num_skin, num_eyes)

    # ── 클래스 가중치 (불균형 보정) ────────────────────────────────────────────
    skin_counts  = AnomalyDataset.get_sample_counts(skin_root, skin_classes)
    eyes_counts  = AnomalyDataset.get_sample_counts(eyes_root, eyes_classes)

    skin_weights = compute_class_weights(skin_counts, skin_classes).to(DEVICE)
    eyes_weights = compute_class_weights(eyes_counts, eyes_classes).to(DEVICE)

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

    # ── Optimizer & Scheduler ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    scaler = GradScaler()

    # ── 학습 기록 & Best 추적 ──────────────────────────────────────
    history      = []   # {epoch, skin_loss, skin_acc, eyes_loss, eyes_acc, avg_acc}
    best_avg_acc = 0.0
    best_epoch   = 0

    # ── Training Loop ──────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        print(f"\n========= Epoch {epoch + 1}/{EPOCHS} =========\n")

        # ──────────────────────────────────────────────────────────────────────
        # 1️⃣  Skin Training
        # ──────────────────────────────────────────────────────────────────────
        print("[1/2] Skin Training")
        model.to(DEVICE)
        model.train()

        skin_dataset = AnomalyDataset(skin_root, skin_classes, task="skin", is_train=True)
        skin_loader  = DataLoader(
            skin_dataset,
            batch_size  = BATCH_SIZE,
            shuffle     = True,
            num_workers = NUM_WORKERS,
            pin_memory  = True,
        )

        skin_loss_sum, skin_correct, skin_total = 0.0, 0, 0

        skin_pbar = tqdm(skin_loader, desc=f"  [Skin ] Epoch {epoch+1:02d}/{EPOCHS}", ncols=110, leave=True)
        for images, labels in skin_pbar:
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

            skin_pbar.set_postfix(
                loss=f"{skin_loss_sum / skin_total:.4f}",
                acc=f"{100 * skin_correct / skin_total:.2f}%"
            )

        del skin_loader, skin_dataset
        gc.collect()
        torch.cuda.empty_cache()

        # ──────────────────────────────────────────────────────────────────────
        # 2️⃣  Eyes Training
        # ──────────────────────────────────────────────────────────────────────
        print("[2/2] Eyes Training")

        eyes_dataset = AnomalyDataset(eyes_root, eyes_classes, task="eyes", is_train=True)
        eyes_loader  = DataLoader(
            eyes_dataset,
            batch_size  = BATCH_SIZE,
            shuffle     = True,
            num_workers = NUM_WORKERS,
            pin_memory  = True,
        )

        eyes_loss_sum, eyes_correct, eyes_total = 0.0, 0, 0

        eyes_pbar = tqdm(eyes_loader, desc=f"  [Eyes ] Epoch {epoch+1:02d}/{EPOCHS}", ncols=110, leave=True)
        for images, labels in eyes_pbar:
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

            eyes_pbar.set_postfix(
                loss=f"{eyes_loss_sum / eyes_total:.4f}",
                acc=f"{100 * eyes_correct / eyes_total:.2f}%"
            )

        del eyes_loader, eyes_dataset
        gc.collect()
        torch.cuda.empty_cache()

        # ── LR Scheduler Step ────────────────────────────────────────────────
        scheduler.step()

        # ── History 기록 ──────────────────────────────────────────
        skin_epoch_loss = skin_loss_sum / skin_total
        skin_epoch_acc  = skin_correct  / skin_total
        eyes_epoch_loss = eyes_loss_sum / eyes_total
        eyes_epoch_acc  = eyes_correct  / eyes_total
        avg_acc         = (skin_epoch_acc + eyes_epoch_acc) / 2

        history.append({
            'epoch'     : epoch + 1,
            'skin_loss' : skin_epoch_loss,
            'skin_acc'  : skin_epoch_acc,
            'eyes_loss' : eyes_epoch_loss,
            'eyes_acc'  : eyes_epoch_acc,
            'avg_acc'   : avg_acc,
        })

        print(f"  Skin | Loss: {skin_epoch_loss:.4f} | Acc: {skin_epoch_acc*100:.2f}%")
        print(f"  Eyes | Loss: {eyes_epoch_loss:.4f} | Acc: {eyes_epoch_acc*100:.2f}%")
        print(f"  Avg Acc: {avg_acc*100:.2f}%")

        # ── Best Model 저장 (avg acc 기준) ───────────────────────────────
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_epoch   = epoch + 1
            torch.save(
                {
                    "model"           : model.state_dict(),
                    "epoch"           : epoch + 1,
                    "best_avg_acc"    : best_avg_acc,
                    "skin_classes"    : SKIN_CLASSES,
                    "eyes_classes"    : EYES_CLASSES,
                    "history"         : history,
                },
                "pet_abnormal_omni_best.pth",
            )
            print(f"  💾 Saved best model! (Epoch {best_epoch} | Avg Acc: {best_avg_acc*100:.2f}%)")


    print(f"\n🏆 Training Finished. Best Epoch: {best_epoch} | Best Avg Acc: {best_avg_acc*100:.2f}%")

    # ── 학습 곡선 시각화 ───────────────────────────────────────────
    print("→3️⃣  Generating training history plot...")
    import matplotlib.pyplot as plt

    epochs_x     = [h['epoch']     for h in history]
    skin_losses  = [h['skin_loss'] for h in history]
    eyes_losses  = [h['eyes_loss'] for h in history]
    skin_accs    = [h['skin_acc']  for h in history]
    eyes_accs    = [h['eyes_acc']  for h in history]
    avg_accs     = [h['avg_acc']   for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ─ Loss ─
    axes[0].plot(epochs_x, skin_losses, 'b-',  linewidth=2, label='Skin Loss')
    axes[0].plot(epochs_x, eyes_losses, 'r-',  linewidth=2, label='Eyes Loss')
    axes[0].axvline(best_epoch, color='gray', linestyle='--', alpha=0.6, label=f'Best Epoch {best_epoch}')
    axes[0].set_title('Training Loss');  axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # ─ Accuracy ─
    axes[1].plot(epochs_x, skin_accs, 'b-',  linewidth=2, label='Skin Acc')
    axes[1].plot(epochs_x, eyes_accs, 'r-',  linewidth=2, label='Eyes Acc')
    axes[1].axvline(best_epoch, color='gray', linestyle='--', alpha=0.6, label=f'Best Epoch {best_epoch}')
    axes[1].set_title('Training Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # ─ Avg Accuracy ─
    axes[2].plot(epochs_x, avg_accs, 'g-', linewidth=2, label='Avg Acc')
    axes[2].axvline(best_epoch, color='gray', linestyle='--', alpha=0.6, label=f'Best Epoch {best_epoch}')
    axes[2].axhline(best_avg_acc, color='green', linestyle=':', alpha=0.6, label=f'Best Acc {best_avg_acc*100:.1f}%')
    axes[2].set_title('Average Accuracy');  axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim(0, 1); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Anomaly Model Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('anomaly_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: pet_abnormal_omni.png")


# ===============================
# ENTRY POINT
# ===============================

if __name__ == "__main__":
    train()