"""
cat_abnormal_omni.py
────────────────────────────────────────────────────────────────────
고양이 이상 증상 분류 모델 (Skin / Eyes)
- Skin backbone: EfficientNet-V2-S (기본) / ViT-B/16 (USE_VIT_SKIN=True)
- Eyes backbone: EfficientNet-V2-S
- 클래스: cat 전용 (dog 클래스 완전 분리)
────────────────────────────────────────────────────────────────────
"""

import os, gc, random, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision.models import (
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    vit_b_16, ViT_B_16_Weights,
)
from transformers import get_cosine_schedule_with_warmup
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ───────────────────────────────── CONFIG ─────────────────────────────────────
SEED = 42
random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)

SKIN_ROOT = "files/4_Animal_Skin"
EYES_ROOT = "files/5_Animal_Eyes"
WORK_DIR  = "files/work/cat_abnormal_dataset"

DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
EPOCHS        = 100
BATCH_SIZE    = 32
NUM_WORKERS   = 12       # [FIX] 24→8: deadlock 방지
LABEL_SMOOTHING = 0.1
FREEZE_EPOCHS = 5

# ── Skin backbone 선택 ──────────────────────────────────────────────────────────
# True  → ViT-B/16  (미세 질감 패턴에 강함, 학습 느림)
# False → EfficientNet-V2-S (기본, 빠름)
USE_VIT_SKIN = False

# 해상도: EfficientNet=384, ViT=384 (ViT-B/16은 384로 학습 가능)
IMG_SIZE   = 384
IMG_RESIZE = 416

LR_BACKBONE = 2e-5
LR_HEAD     = 2e-4
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

print(f"🐱 Cat Abnormal Omni | Device: {DEVICE} | Skin backbone: {'ViT-B/16' if USE_VIT_SKIN else 'EfficientNet-V2-S'}")

# ──────────────────────────────── CLASSES ─────────────────────────────────────
CAT_SKIN_CLASSES = [
    "cat_normal",
    "cat_결절,종괴",
    "cat_농포,여드름",
    "cat_비듬,각질,상피성잔고리",
]  # 4클래스 — 각 5,000개 균형

CAT_EYES_CLASSES = [
    "cat_normal",
    "cat_각막궤양",
    "cat_각막부골편",
    "cat_결막염",
    "cat_비궤양성각막염",
    "cat_안검염",
]  # 6클래스

# Eyes 유사 클래스 페널티
EYES_SIMILAR_GROUPS = [
    ["cat_각막궤양", "cat_비궤양성각막염"],   # 각막 유형 혼동
]

# ─────────────────────────────── LOSS ──────────────────────────────────────────
class WeightedSmoothLoss(nn.Module):
    """CrossEntropy + Label Smoothing + 유사 클래스 페널티."""
    def __init__(self, class_names, similar_groups=None, class_weights=None,
                 smoothing=LABEL_SMOOTHING, penalty=1.5):
        super().__init__()
        self.smoothing = smoothing
        self.penalty   = penalty
        self.name_to_idx = {n: i for i, n in enumerate(class_names)}
        self.penalty_pairs = set()
        if similar_groups:
            for g in similar_groups:
                idxs = [self.name_to_idx[n] for n in g if n in self.name_to_idx]
                for i in range(len(idxs)):
                    for j in range(i+1, len(idxs)):
                        self.penalty_pairs.add((idxs[i], idxs[j]))
                        self.penalty_pairs.add((idxs[j], idxs[i]))
        self.register_buffer("weight", class_weights)

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=-1)
        nll   = F.nll_loss(log_p, targets, weight=self.weight, reduction="none")
        smooth = -log_p.mean(dim=-1)
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth
        if self.penalty_pairs:
            preds = logits.argmax(1)
            mask  = torch.ones(len(targets), device=logits.device)
            for b in range(len(targets)):
                if (targets[b].item(), preds[b].item()) in self.penalty_pairs:
                    mask[b] = self.penalty
            loss = loss * mask
        return loss.mean()

# ─────────────────────────────── BACKBONES ────────────────────────────────────
def _efficientnet_backbone():
    b = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    feat = b.classifier[1].in_features  # 1280
    b.classifier = nn.Identity()
    return b, feat

def _vit_backbone():
    """
    ViT-B/16 (SWAG fine-tuned, 384×384 권장).
    피부 미세 질감 분류에 self-attention이 유리.
    """
    b = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    feat = b.heads.head.in_features  # 768
    b.heads = nn.Identity()
    return b, feat

# ─────────────────────────────── MODELS ───────────────────────────────────────
class SkinModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        if USE_VIT_SKIN:
            self.backbone, feat = _vit_backbone()
        else:
            self.backbone, feat = _efficientnet_backbone()
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.head(self.backbone(x))

class EyesModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone, feat = _efficientnet_backbone()
        # SE attention
        self.se = nn.Sequential(
            nn.Linear(feat, feat//16, bias=False), nn.ReLU(),
            nn.Linear(feat//16, feat, bias=False), nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        f = self.backbone(x)
        return self.head(f * self.se(f))

# ─────────────────────────────── TRANSFORMS ───────────────────────────────────
# Skin: 피부색·질감이 진단 핵심 → ColorJitter 최소화, Sharpness 강조
SKIN_TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((IMG_RESIZE, IMG_RESIZE)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
# Eyes: 충혈·혼탁 등 색 변화가 진단 단서 → ColorJitter 유지
EYES_TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((IMG_RESIZE, IMG_RESIZE)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5), p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
TRANSFORM_VAL = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─────────────────────────────── DATASET ──────────────────────────────────────
class AnomalyDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l = self.samples[idx]
        return self.transform(Image.open(p).convert("RGB")), l

# ─────────────────────────── DATA PREPARATION ─────────────────────────────────
def _task_ready(name):
    p = os.path.join(WORK_DIR, "train", name)
    return os.path.isdir(p) and len(os.listdir(p)) > 0

def collect_copy_split(src_root, task_name, class_list):
    rng = random.Random(SEED)
    name_to_idx = {n: i for i, n in enumerate(class_list)}
    class_files = defaultdict(list)
    seen = set()
    for cls in class_list:
        d = os.path.join(src_root, cls)
        if not os.path.isdir(d): continue
        for f in os.listdir(d):
            if not f.lower().endswith((".jpg",".jpeg",".png")): continue
            fp = os.path.join(d, f)
            if fp in seen: continue
            seen.add(fp)
            class_files[name_to_idx[cls]].append(fp)
    for split in ["train","val","test"]:
        for cls in class_list:
            os.makedirs(os.path.join(WORK_DIR, split, task_name, cls), exist_ok=True)
    train_s, val_s = [], []
    for idx, paths in class_files.items():
        cls = class_list[idx]
        rng.shuffle(paths)
        n = len(paths)
        nv = max(1, int(n * VAL_RATIO))
        nt = max(1, int(n * TEST_RATIO))
        splits = {"train": paths[:n-nv-nt], "val": paths[n-nv-nt:n-nt], "test": paths[n-nt:]}
        for sname, spaths in splits.items():
            dst = os.path.join(WORK_DIR, sname, task_name, cls)
            for src in spaths:
                d = os.path.join(dst, os.path.basename(src))
                if not os.path.exists(d): shutil.copy2(src, d)
            if sname == "train": train_s.extend((os.path.join(dst, os.path.basename(s)), idx) for s in spaths)
            elif sname == "val": val_s.extend((os.path.join(dst, os.path.basename(s)), idx) for s in spaths)
    print(f"  ✅ {task_name}: train={len(train_s)} val={len(val_s)}")
    return train_s, val_s

def load_from_dir(task_dir, class_list):
    name_to_idx = {n: i for i, n in enumerate(class_list)}
    samples = []
    for cls in sorted(os.listdir(task_dir)):
        d = os.path.join(task_dir, cls)
        if not os.path.isdir(d) or cls not in name_to_idx: continue
        for f in os.listdir(d):
            if f.lower().endswith((".jpg",".jpeg",".png")):
                samples.append((os.path.join(d, f), name_to_idx[cls]))
    return samples

def class_weights_from_samples(samples, n_classes):
    labels = [l for _, l in samples]
    w = compute_class_weight('balanced', classes=np.arange(n_classes), y=labels)
    return torch.tensor(w, dtype=torch.float).to(DEVICE)

# ──────────────────────────────── HELPERS ─────────────────────────────────────
def make_loader(samples, transform, shuffle, drop_last=False):
    ds = AnomalyDataset(samples, transform)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True,
                      persistent_workers=True,
                      prefetch_factor=2,                                      # [FIX] 4→2
                      multiprocessing_context="fork",                         # [FIX] deadlock 방지
                      drop_last=drop_last)                                    # [FIX] 마지막 배치 BatchNorm 방어

def clear(): gc.collect(); torch.cuda.empty_cache()

def _save_history_plot(history, best_acc):
    """매 epoch 호출 → 학습 중단 시에도 마지막 epoch까지의 곡선 확인 가능"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot([h["skin_val"] for h in history], 'b-', linewidth=2, label="Skin Val")
    ax1.set_ylim(0, 1); ax1.set_title("Skin Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot([h["eyes_val"] for h in history], 'r-', linewidth=2, label="Eyes Val")
    ax2.set_ylim(0, 1); ax2.set_title("Eyes Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.suptitle(f"Cat Abnormal Omni | Best Avg {best_acc*100:.1f}%", fontweight="bold")
    plt.tight_layout()
    plt.savefig("cat_abnormal_omni_history.png", dpi=150)
    plt.close()

def val_loop(model, loader):
    model.eval()
    c, t = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast("cuda"):
                out = model(imgs)
            c += (out.argmax(1)==labels).sum().item(); t += labels.size(0)
    return c/t if t > 0 else 0.0   # [FIX] ZeroDivisionError 방어

# ─────────────────────────────── TRAINING ─────────────────────────────────────
def train():
    # ── Dataset ──────────────────────────────────────────────────────────────────
    if _task_ready("skin"):
        print("✅ skin ready"); skin_train = load_from_dir(os.path.join(WORK_DIR,"train","skin"), CAT_SKIN_CLASSES)
        skin_val   = load_from_dir(os.path.join(WORK_DIR,"val","skin"), CAT_SKIN_CLASSES)
    else:
        print("📦 Preparing skin (cat)...")
        skin_train, skin_val = collect_copy_split(SKIN_ROOT, "skin", CAT_SKIN_CLASSES)

    if _task_ready("eyes"):
        print("✅ eyes ready"); eyes_train = load_from_dir(os.path.join(WORK_DIR,"train","eyes"), CAT_EYES_CLASSES)
        eyes_val   = load_from_dir(os.path.join(WORK_DIR,"val","eyes"), CAT_EYES_CLASSES)
    else:
        print("📦 Preparing eyes (cat)...")
        eyes_train, eyes_val = collect_copy_split(EYES_ROOT, "eyes", CAT_EYES_CLASSES)

    # ── Loss ──────────────────────────────────────────────────────────────────────
    skin_w   = class_weights_from_samples(skin_train, len(CAT_SKIN_CLASSES))
    eyes_w   = class_weights_from_samples(eyes_train, len(CAT_EYES_CLASSES))
    skin_crit = WeightedSmoothLoss(CAT_SKIN_CLASSES, class_weights=skin_w)
    eyes_crit = WeightedSmoothLoss(CAT_EYES_CLASSES, similar_groups=EYES_SIMILAR_GROUPS, class_weights=eyes_w)

    # ── Models ─────────────────────────────────────────────────────────────────────
    skin_model = SkinModel(len(CAT_SKIN_CLASSES)).to(DEVICE)
    eyes_model = EyesModel(len(CAT_EYES_CLASSES)).to(DEVICE)

    # Freeze backbone
    for m in [skin_model, eyes_model]:
        for p in m.backbone.parameters(): p.requires_grad = False
    print(f"  🔒 Backbone frozen for {FREEZE_EPOCHS} epochs")

    # ── Optimizers ─────────────────────────────────────────────────────────────────
    def make_opt(m):
        return torch.optim.AdamW([
            {"params": m.backbone.parameters(), "lr": LR_BACKBONE, "weight_decay": 1e-4},
            {"params": [p for n, p in m.named_parameters() if "backbone" not in n],
             "lr": LR_HEAD, "weight_decay": 1e-4},
        ])
    skin_opt = make_opt(skin_model)
    eyes_opt = make_opt(eyes_model)

    def make_sched(opt, n):
        steps = (n // BATCH_SIZE) * EPOCHS
        return get_cosine_schedule_with_warmup(opt, steps // 25, steps)
    skin_sched = make_sched(skin_opt, len(skin_train))
    eyes_sched = make_sched(eyes_opt, len(eyes_train))

    skin_scaler = torch.amp.GradScaler("cuda")
    eyes_scaler = torch.amp.GradScaler("cuda")

    best_acc, history = 0.0, []

    for epoch in range(EPOCHS):
        print(f"\n{'='*55}\nEpoch {epoch+1}/{EPOCHS}\n{'='*55}")
        if epoch == FREEZE_EPOCHS:
            for m in [skin_model, eyes_model]:
                for p in m.backbone.parameters(): p.requires_grad = True
            print(f"  🔓 Backbone unfrozen")

        def train_task(name, model, samples, crit, transform, opt, scaler, sched):
            model.train()
            loader = make_loader(samples, transform, shuffle=True, drop_last=True)  # [FIX] drop_last
            ls, c, t = 0, 0, 0
            for imgs, labels in tqdm(loader, desc=f"  {name}", ncols=100, leave=True):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                opt.zero_grad()
                with torch.amp.autocast("cuda"):
                    out  = model(imgs)
                    loss = crit(out, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                prev_scale = scaler.get_scale()                    # [FIX] GradScaler skip 처리
                scaler.step(opt); scaler.update()
                if scaler.get_scale() == prev_scale:
                    sched.step()
                ls += loss.item(); c += (out.argmax(1)==labels).sum().item(); t += labels.size(0)
            del loader
            return ls / (t / BATCH_SIZE + 1), c / t

        print("\n[1/2] Skin")
        s_loss, s_tacc = train_task("Skin", skin_model, skin_train, skin_crit,
                                    SKIN_TRANSFORM_TRAIN, skin_opt, skin_scaler, skin_sched)
        clear()

        print("\n[2/2] Eyes")
        e_loss, e_tacc = train_task("Eyes", eyes_model, eyes_train, eyes_crit,
                                    EYES_TRANSFORM_TRAIN, eyes_opt, eyes_scaler, eyes_sched)
        clear()

        # Validation
        skin_vacc = val_loop(skin_model, make_loader(skin_val, TRANSFORM_VAL, shuffle=False, drop_last=False)); clear()
        eyes_vacc = val_loop(eyes_model, make_loader(eyes_val, TRANSFORM_VAL, shuffle=False, drop_last=False)); clear()
        avg = (skin_vacc + eyes_vacc) / 2

        print(f"\n📊 Ep{epoch+1} | Skin Train {s_tacc*100:.1f}% Val {skin_vacc*100:.1f}% "
              f"| Eyes Train {e_tacc*100:.1f}% Val {eyes_vacc*100:.1f}% | Avg {avg*100:.1f}%")

        history.append({"epoch": epoch+1, "skin_val": skin_vacc, "eyes_val": eyes_vacc, "avg": avg})
        if avg > best_acc:
            best_acc = avg
            torch.save({
                "skin_model": skin_model.state_dict(),
                "eyes_model": eyes_model.state_dict(),
                "cat_skin_classes": CAT_SKIN_CLASSES,
                "cat_eyes_classes": CAT_EYES_CLASSES,
                "use_vit_skin": USE_VIT_SKIN,
                "best_epoch": epoch+1, "best_acc": best_acc, "history": history,
            }, "cat_abnormal_omni_best.pth")
            print(f"  💾 Saved! Avg {best_acc*100:.1f}%")

        # 매 epoch 학습 곡선 덮어쓰기 저장
        _save_history_plot(history, best_acc)

    print(f"\n🎉 Done! Best Avg: {best_acc*100:.1f}%")

if __name__ == "__main__":
    train()
