"""
cat_normal_omni_test.py
────────────────────────────────────────────────────────────────────
고양이 정상 행동 분류 모델 테스트 (Behavior / Emotion / Sound)
- 체크포인트: cat_normal_omni_best.pth
- 테스트 데이터: files/work/cat_normal_dataset/test/
- 출력: 태스크별 정확도 + 클래스별 정확도 + Confusion Matrix PNG
────────────────────────────────────────────────────────────────────

사용법:
  python cat_normal_omni_test.py
  python cat_normal_omni_test.py --model_path /path/to/cat_normal_omni_best.pth
  python cat_normal_omni_test.py --model_path cat_normal_omni_best.pth --device cuda:1
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
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ───────────────────────────────── CONFIG ─────────────────────────────────────
SEED = 42
random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)

WORK_DIR         = "files/work/cat_normal_dataset"
DEFAULT_CKPT     = "cat_normal_omni_best.pth"
AUDIO_MODEL_NAME = "facebook/wav2vec2-base"

SR            = 16000
MAX_AUDIO_LEN = SR * 5
IMG_SIZE      = 384
BATCH_SIZE    = 64
AUDIO_BATCH   = 16
NUM_WORKERS   = 4

# ───────────────────────────── ARG PARSE ──────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Cat Normal Omni — Test")
    p.add_argument("--model_path", type=str, default=DEFAULT_CKPT,
                   help=f"체크포인트 경로 (기본: {DEFAULT_CKPT})")
    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu",
                   help="추론 디바이스 (기본: cuda:0)")
    p.add_argument("--work_dir", type=str, default=WORK_DIR,
                   help=f"데이터셋 루트 (기본: {WORK_DIR})")
    return p.parse_args()

# ─────────────────────────────── TRANSFORM ────────────────────────────────────
TRANSFORM_TEST = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─────────────────────────────── DATASETS ─────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, task_dir, class_list):
        self.label_to_id = {c: i for i, c in enumerate(class_list)}
        self.id_to_label = {i: c for i, c in enumerate(class_list)}
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
        self.id_to_label = {i: c for i, c in enumerate(class_list)}
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

# ─────────────────────────────── MODELS ───────────────────────────────────────
def _efficientnet_backbone():
    b = efficientnet_v2_s(weights=None)   # 가중치는 체크포인트에서 로드
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
    """이미지 모델 평가 → (overall_acc, per_class_acc, all_preds, all_labels)"""
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
                correct[l] += int(p == l)
                total[l]   += 1
                all_preds.append(p); all_labels.append(l)
    model.cpu(); clear()

    per_class = np.where(total > 0, correct / total, 0.0)
    overall   = correct.sum() / total.sum() if total.sum() > 0 else 0.0
    return overall, per_class, np.array(all_preds), np.array(all_labels)

def evaluate_audio(model, loader, class_list, device):
    """오디오 모델 평가 → (overall_acc, per_class_acc, all_preds, all_labels)"""
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
                correct[l] += int(p == l)
                total[l]   += 1
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

    # 행 합계로 정규화 (recall 기반)
    cm_norm = cm.astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm_norm / row_sum, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(max(10, n * 0.9 + 2), max(7, n * 0.8 + 1)))

    for ax, data, fmt, ttl in zip(
        axes,
        [cm, cm_norm],
        [".0f", ".2f"],
        ["Count", "Recall (row-normalized)"]
    ):
        im = ax.imshow(data, interpolation="nearest",
                       cmap="Blues" if ttl == "Count" else "RdYlGn",
                       vmin=0, vmax=(None if ttl == "Count" else 1.0))
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
    print(f"  💾 Confusion matrix saved → {save_path}")

# ─────────────────────────────── MAIN TEST ────────────────────────────────────
def test():
    args = parse_args()
    device   = args.device
    ckpt_path = args.model_path
    work_dir  = args.work_dir

    print(f"\n🐱 Cat Normal Omni — Test")
    print(f"  Device    : {device}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Work Dir  : {work_dir}")

    # ── 1. 체크포인트 로드 ───────────────────────────────────────────────────────
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    CAT_BEHAVIOR_CLASSES = ckpt["cat_behavior_classes"]
    CAT_EMOTION_CLASSES  = ckpt["cat_emotion_classes"]
    CAT_SOUND_CLASSES    = ckpt["cat_sound_classes"]
    best_epoch = ckpt.get("best_epoch", "?")
    best_acc   = ckpt.get("best_acc",   0.0)

    print(f"\n  📋 Best Epoch: {best_epoch}  |  Best Val Avg: {best_acc*100:.1f}%")
    print(f"  Behavior classes ({len(CAT_BEHAVIOR_CLASSES)}): {CAT_BEHAVIOR_CLASSES}")
    print(f"  Emotion  classes ({len(CAT_EMOTION_CLASSES)}):  {CAT_EMOTION_CLASSES}")
    print(f"  Sound    classes ({len(CAT_SOUND_CLASSES)}):    {CAT_SOUND_CLASSES}")

    # ── 2. 모델 복원 ─────────────────────────────────────────────────────────────
    print("\n  🔧 모델 복원 중...")
    behavior_model = ImageModel(len(CAT_BEHAVIOR_CLASSES))
    emotion_model  = ImageModel(len(CAT_EMOTION_CLASSES))
    audio_model    = AudioModel(len(CAT_SOUND_CLASSES))

    behavior_model.load_state_dict(ckpt["behavior_model"])
    emotion_model.load_state_dict(ckpt["emotion_model"])
    audio_model.load_state_dict(ckpt["audio_model"])
    print("  ✅ 모델 가중치 로드 완료")

    # ── 3. Feature Extractor (오디오) ────────────────────────────────────────────
    print(f"\n  🎙️ Feature Extractor 로드: {AUDIO_MODEL_NAME}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)

    # ── 4. 테스트 데이터셋 로드 ──────────────────────────────────────────────────
    test_root = os.path.join(work_dir, "test")
    print(f"\n  📦 Test 데이터 로드: {test_root}")

    bds = ImageDataset(os.path.join(test_root, "behavior"), CAT_BEHAVIOR_CLASSES)
    eds = ImageDataset(os.path.join(test_root, "emotion"),  CAT_EMOTION_CLASSES)
    sds = AudioDataset(os.path.join(test_root, "sound"),    CAT_SOUND_CLASSES, feature_extractor)

    bl = make_loader(bds, is_audio=False)
    el = make_loader(eds, is_audio=False)
    sl = make_loader(sds, is_audio=True)

    # ── 5. 평가 ──────────────────────────────────────────────────────────────────
    results = {}
    print("\n" + "="*55 + "\n  📊 TEST EVALUATION\n" + "="*55)

    # Behavior
    print("\n🐾 Behavior")
    b_acc, b_per, b_preds, b_labels = evaluate_image(
        behavior_model, bl, CAT_BEHAVIOR_CLASSES, device)
    results["behavior"] = {"acc": b_acc, "per_class": b_per, "preds": b_preds, "labels": b_labels}
    print(f"  Overall: {b_acc*100:.2f}%")
    for cls, acc in zip(CAT_BEHAVIOR_CLASSES, b_per):
        print(f"    {cls:<35s}: {acc*100:.1f}%")

    # Emotion
    print("\n😊 Emotion")
    e_acc, e_per, e_preds, e_labels = evaluate_image(
        emotion_model, el, CAT_EMOTION_CLASSES, device)
    results["emotion"] = {"acc": e_acc, "per_class": e_per, "preds": e_preds, "labels": e_labels}
    print(f"  Overall: {e_acc*100:.2f}%")
    for cls, acc in zip(CAT_EMOTION_CLASSES, e_per):
        print(f"    {cls:<35s}: {acc*100:.1f}%")

    # Sound
    print("\n🔊 Sound")
    s_acc, s_per, s_preds, s_labels = evaluate_audio(
        audio_model, sl, CAT_SOUND_CLASSES, device)
    results["sound"] = {"acc": s_acc, "per_class": s_per, "preds": s_preds, "labels": s_labels}
    print(f"  Overall: {s_acc*100:.2f}%")
    for cls, acc in zip(CAT_SOUND_CLASSES, s_per):
        print(f"    {cls:<35s}: {acc*100:.1f}%")

    # ── 6. 전체 요약 ─────────────────────────────────────────────────────────────
    avg = np.mean([r["acc"] for r in results.values()])
    print("\n" + "="*55)
    print(f"  ✅ Behavior : {results['behavior']['acc']*100:.2f}%")
    print(f"  ✅ Emotion  : {results['emotion']['acc']*100:.2f}%")
    print(f"  ✅ Sound    : {results['sound']['acc']*100:.2f}%")
    print(f"  ─────────────────────")
    print(f"  📌 Test Avg : {avg*100:.2f}%")
    print(f"  📌 Best Val : {best_acc*100:.1f}%  (Epoch {best_epoch})")
    print("="*55)

    # ── 7. Confusion Matrix 저장 ─────────────────────────────────────────────────
    print("\n  📈 Confusion Matrix 저장 중...")
    plot_confusion_matrix(b_preds, b_labels, CAT_BEHAVIOR_CLASSES,
                          "Cat Behavior (Test)", "cat_normal_behavior_cm.png")
    plot_confusion_matrix(e_preds, e_labels, CAT_EMOTION_CLASSES,
                          "Cat Emotion (Test)", "cat_normal_emotion_cm.png")
    plot_confusion_matrix(s_preds, s_labels, CAT_SOUND_CLASSES,
                          "Cat Sound (Test)", "cat_normal_sound_cm.png")

    # ── 8. 결과 요약 PNG ─────────────────────────────────────────────────────────
    _save_summary_plot(results,
                       {"behavior": CAT_BEHAVIOR_CLASSES,
                        "emotion":  CAT_EMOTION_CLASSES,
                        "sound":    CAT_SOUND_CLASSES},
                       avg, best_acc, best_epoch)

    print("\n🎉 테스트 완료!")


def _save_summary_plot(results, class_map, avg_acc, best_val_acc, best_epoch):
    """태스크별 per-class 막대 그래프 + 전체 요약"""
    n_tasks = len(results)
    fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 6))

    colors = {"behavior": "steelblue", "emotion": "seagreen", "sound": "tomato"}

    for ax, (task, res) in zip(axes, results.items()):
        classes  = class_map[task]
        per_cls  = res["per_class"]
        x = np.arange(len(classes))
        bars = ax.bar(x, per_cls * 100, color=colors.get(task, "gray"), alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axhline(y=res["acc"] * 100, color="red", linestyle="--", linewidth=1.5,
                   label=f"Overall {res['acc']*100:.1f}%")
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 110); ax.set_ylabel("Accuracy (%)"); ax.set_title(f"Cat {task.capitalize()} — Test")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, per_cls):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val*100:.0f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle(
        f"Cat Normal Omni — Test Summary\n"
        f"Test Avg: {avg_acc*100:.2f}%  |  Best Val: {best_val_acc*100:.1f}% (Ep {best_epoch})",
        fontweight="bold", fontsize=12,
    )
    plt.tight_layout()
    plt.savefig("cat_normal_omni_test_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  💾 Summary plot saved → cat_normal_omni_test_summary.png")


if __name__ == "__main__":
    test()
