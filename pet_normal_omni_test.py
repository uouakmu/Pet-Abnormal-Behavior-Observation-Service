"""
pet_normal_omni_test.py

학습된 pet_normal_omni_best.pth 를 로드하여
WORK_DIR/test/ 하위에 이미 분리된 test split에 대한 최종 성능을 평가한다.

출력 항목:
  - Behavior / Emotion / Sound / Patella 전체 Accuracy
  - 태스크별 클래스별 Precision / Recall / F1 (classification_report)
  - 태스크별 Confusion Matrix 히트맵 PNG
  - 태스크별 F1 바차트 PNG
  - 요약 결과 JSON (test_results.json)

사용법:
  python pet_normal_omni_test.py \
      --ckpt   pet_normal_omni_best.pth \
      --work_dir files/work/omni_dataset \
      --output_dir test_results
"""

import os
import gc
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import torchvision.transforms as transforms
import librosa
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# CONFIG  (train 파일과 동일)
# =========================

DEVICE        = "cuda:1" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 64
NUM_WORKERS   = 8
SR            = 16000
MAX_AUDIO_LEN = SR * 5
AUDIO_MODEL_NAME = "facebook/wav2vec2-base"

print(f"🎯 Device: {DEVICE}")

FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)


# =========================
# Dataset Classes  (train과 동일, augment=False 고정)
# =========================

class ImageDataset(Dataset):
    """Behavior / Emotion test용 이미지 Dataset."""

    def __init__(self, task_dir: str, label_to_id: dict):
        """
        Args:
            task_dir    : WORK_DIR/test/<task> 경로
            label_to_id : checkpoint에서 복원한 label → id 매핑
        """
        self.samples     = []
        self.label_to_id = label_to_id

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for label in sorted(os.listdir(task_dir)):
            label_dir = os.path.join(task_dir, label)
            if not os.path.isdir(label_dir):
                continue
            if label not in label_to_id:
                print(f"  ⚠️  '{label}' 이 label_to_id 에 없습니다. 건너뜁니다.")
                continue
            for file in os.listdir(label_dir):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(label_dir, file), label))

        print(f"  📊 {os.path.basename(task_dir)}: {len(self.samples)} test samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.label_to_id[label]


class PatellaDataset(Dataset):
    """Patella test용 Dataset."""

    def __init__(self, task_dir: str, label_to_id: dict):
        self.samples     = []
        self.label_to_id = label_to_id

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for label in sorted(os.listdir(task_dir)):
            label_dir = os.path.join(task_dir, label)
            if not os.path.isdir(label_dir):
                continue
            if label not in label_to_id:
                print(f"  ⚠️  '{label}' 이 label_to_id 에 없습니다. 건너뜁니다.")
                continue
            for file in os.listdir(label_dir):
                if file.lower().endswith('.jpg'):
                    img_path  = os.path.join(label_dir, file)
                    json_path = img_path.replace('.jpg', '.json')
                    if os.path.exists(json_path):
                        self.samples.append((img_path, json_path, label))

        print(f"  📊 {os.path.basename(task_dir)}: {len(self.samples)} test samples")

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
            keypoints.extend([float(annotation.get('x', 0)),
                               float(annotation.get('y', 0))])
        while len(keypoints) < 18:
            keypoints.append(0.0)
        keypoints = torch.tensor(keypoints[:18], dtype=torch.float32)

        return img, keypoints, self.label_to_id[label]


class AudioDataset(Dataset):
    """Sound test용 Dataset. Augmentation 없음."""

    def __init__(self, task_dir: str, label_to_id: dict):
        self.samples     = []
        self.label_to_id = label_to_id

        for label in sorted(os.listdir(task_dir)):
            label_dir = os.path.join(task_dir, label)
            if not os.path.isdir(label_dir):
                continue
            if label not in label_to_id:
                print(f"  ⚠️  '{label}' 이 label_to_id 에 없습니다. 건너뜁니다.")
                continue
            for file in os.listdir(label_dir):
                if file.lower().endswith(('.wav', '.mp3', '.m4a')):
                    self.samples.append((os.path.join(label_dir, file), label))

        print(f"  📊 {os.path.basename(task_dir)}: {len(self.samples)} test samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            waveform, _ = librosa.load(path, sr=SR, mono=True)
        except Exception:
            waveform = np.zeros(MAX_AUDIO_LEN)

        if len(waveform) > MAX_AUDIO_LEN:
            waveform = waveform[:MAX_AUDIO_LEN]
        else:
            waveform = np.pad(waveform, (0, MAX_AUDIO_LEN - len(waveform)))

        inputs = FEATURE_EXTRACTOR(waveform, sampling_rate=SR, return_tensors="pt")
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels":       torch.tensor(self.label_to_id[label], dtype=torch.long)
        }


def collate_fn_audio(batch):
    return {
        "input_values": torch.stack([b["input_values"] for b in batch]),
        "labels":       torch.stack([b["labels"]       for b in batch]),
    }


# =========================
# Model Definitions  (train과 동일)
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
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))

    def forward(self, x):
        return self.head(self.backbone(x))


class EmotionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone, in_features = _efficientnet_b3_backbone()
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))

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
        return self.head(torch.cat([self.backbone(x), keypoints], dim=1))


class AudioModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            AUDIO_MODEL_NAME,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, input_values, labels=None):
        return self.model(input_values=input_values, labels=labels)


# =========================
# Evaluation Utilities
# =========================

def make_loader(dataset, is_audio=False):
    workers = 2 if is_audio else NUM_WORKERS
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        prefetch_factor=4 if workers > 0 else None,
        collate_fn=collate_fn_audio if is_audio else None,
    )


def plot_confusion_matrix(cm, class_names, title, save_path):
    n        = len(class_names)
    figsize  = (max(10, n * 0.8), max(8, n * 0.7))
    fontsize = max(6, 12 - n // 5)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticklabels(class_names, fontsize=fontsize)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=fontsize,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {save_path}")


def save_f1_bar(report_dict, class_names, acc, task_name, save_path):
    f1_scores = [report_dict.get(cn, {}).get("f1-score", 0.0) for cn in class_names]
    colors    = ["steelblue" if s >= 0.7 else "tomato" if s < 0.4 else "orange"
                 for s in f1_scores]

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.8), 5))
    bars = ax.bar(range(len(class_names)), f1_scores, color=colors, edgecolor="white", alpha=0.88)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"{task_name.upper()} — Per-class F1  (Acc: {acc*100:.2f}%)",
                 fontsize=12, fontweight="bold")
    ax.axhline(acc, color="black", linestyle="--", alpha=0.6,
               label=f"Overall Acc {acc*100:.2f}%")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for bar, score in zip(bars, f1_scores):
        if score > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{score:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {save_path}")


def evaluate_image_task(model, dataset, label_to_id, task_name, output_dir):
    """Behavior / Emotion 공통 평가 루프."""
    if len(dataset) == 0:
        print(f"  ⚠️  {task_name} test samples 가 없습니다.")
        return {}

    loader = make_loader(dataset)
    id_to_label = {v: k for k, v in label_to_id.items()}

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"  [{task_name}] Inference", ncols=110, leave=False):
            imgs   = imgs.to(DEVICE)
            with autocast(DEVICE):
                logits = model(imgs)
            preds = logits.argmax(-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    del loader
    gc.collect(); torch.cuda.empty_cache()

    return _build_results(all_labels, all_preds, label_to_id, id_to_label, task_name, output_dir)


def evaluate_patella_task(model, dataset, label_to_id, output_dir):
    """Patella 평가 루프 (keypoint 포함)."""
    if len(dataset) == 0:
        print(f"  ⚠️  patella test samples 가 없습니다.")
        return {}

    loader = make_loader(dataset)
    id_to_label = {v: k for k, v in label_to_id.items()}

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, keypoints, labels in tqdm(loader, desc="  [Patella] Inference", ncols=110, leave=False):
            imgs, keypoints = imgs.to(DEVICE), keypoints.to(DEVICE)
            with autocast(DEVICE):
                logits = model(imgs, keypoints)
            preds = logits.argmax(-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    del loader
    gc.collect(); torch.cuda.empty_cache()

    return _build_results(all_labels, all_preds, label_to_id, id_to_label, "patella", output_dir)


def evaluate_sound_task(model, dataset, label_to_id, output_dir):
    """Sound 평가 루프 (Wav2Vec2)."""
    if len(dataset) == 0:
        print(f"  ⚠️  sound test samples 가 없습니다.")
        return {}

    loader = make_loader(dataset, is_audio=True)
    id_to_label = {v: k for k, v in label_to_id.items()}

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="  [Sound] Inference", ncols=110, leave=False):
            audios = batch["input_values"].to(DEVICE)
            labels = batch["labels"]
            with autocast(DEVICE):
                outputs = model(input_values=audios)
            preds = outputs.logits.argmax(-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    del loader
    gc.collect(); torch.cuda.empty_cache()

    return _build_results(all_labels, all_preds, label_to_id, id_to_label, "sound", output_dir)


def _build_results(all_labels, all_preds, label_to_id, id_to_label, task_name, output_dir):
    """추론 결과로부터 acc / report / confusion matrix 를 계산하고 시각화를 저장한다."""
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n  [{task_name.upper()}] Overall Accuracy: {acc*100:.2f}%")

    present_ids   = sorted(set(all_labels))
    present_names = [id_to_label[i] for i in present_ids]

    report_str = classification_report(
        all_labels, all_preds,
        labels=present_ids, target_names=present_names,
        digits=4, zero_division=0,
    )
    report_dict = classification_report(
        all_labels, all_preds,
        labels=present_ids, target_names=present_names,
        digits=4, zero_division=0, output_dict=True,
    )
    print(f"\n{report_str}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=present_ids)
    plot_confusion_matrix(
        cm, present_names,
        title     = f"{task_name.upper()} Confusion Matrix  (Acc: {acc*100:.2f}%)",
        save_path = os.path.join(output_dir, f"confusion_matrix_{task_name}.png"),
    )

    # F1 바차트
    save_f1_bar(
        report_dict, present_names, acc, task_name,
        save_path = os.path.join(output_dir, f"f1_bar_{task_name}.png"),
    )

    return {
        "task"             : task_name,
        "num_test_samples" : len(all_labels),
        "accuracy"         : round(acc, 6),
        "report"           : report_dict,
        "confusion_matrix" : cm.tolist(),
        "class_names"      : present_names,
    }


# =========================
# Main Test Function
# =========================

def test(
    ckpt_path : str = "pet_normal_omni_best.pth",
    work_dir  : str = "files/work/omni_dataset",
    output_dir: str = "test_results",
):
    """
    Best checkpoint 로드 후 WORK_DIR/test/ 하위의 각 태스크 test set을 평가한다.

    Args:
        ckpt_path  : 학습 중 저장된 .pth 파일 경로
        work_dir   : train 시 사용한 WORK_DIR (test 폴더가 그 하위에 존재)
        output_dir : 결과 파일 저장 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📂 Checkpoint : {ckpt_path}")
    print(f"📂 Work Dir   : {work_dir}")

    # ── Checkpoint 로드 ────────────────────────────────────────────────────────
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    behavior_label_to_id = ckpt["behavior_label_to_id"]
    emotion_label_to_id  = ckpt["emotion_label_to_id"]
    sound_label_to_id    = ckpt["sound_label_to_id"]
    patella_label_to_id  = ckpt["patella_label_to_id"]

    best_epoch = ckpt.get("best_epoch", "?")
    best_acc   = ckpt.get("best_acc",   None)
    print(f"📌 Checkpoint info  →  best epoch: {best_epoch}"
          + (f"  |  best val avg acc: {best_acc*100:.2f}%" if best_acc else ""))

    # ── 모델 복원 ──────────────────────────────────────────────────────────────
    print("\n🔄 Loading models...")
    behavior_model = BehaviorModel(len(behavior_label_to_id))
    emotion_model  = EmotionModel(len(emotion_label_to_id))
    patella_model  = PatellaModel(len(patella_label_to_id))
    audio_model    = AudioModel(len(sound_label_to_id))

    behavior_model.load_state_dict(ckpt["behavior_model"])
    emotion_model.load_state_dict( ckpt["emotion_model"])
    patella_model.load_state_dict( ckpt["patella_model"])
    audio_model.load_state_dict(   ckpt["audio_model"])
    print("✅ All models loaded.")

    test_dir = os.path.join(work_dir, "test")

    # ── 1. Behavior ────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}\n[Test] BEHAVIOR")
    behavior_model.to(DEVICE)
    behavior_ds = ImageDataset(os.path.join(test_dir, "behavior"), behavior_label_to_id)
    behavior_results = evaluate_image_task(
        behavior_model, behavior_ds, behavior_label_to_id, "behavior", output_dir
    )
    behavior_model.cpu()
    del behavior_ds
    gc.collect(); torch.cuda.empty_cache()

    # ── 2. Emotion ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}\n[Test] EMOTION")
    emotion_model.to(DEVICE)
    emotion_ds = ImageDataset(os.path.join(test_dir, "emotion"), emotion_label_to_id)
    emotion_results = evaluate_image_task(
        emotion_model, emotion_ds, emotion_label_to_id, "emotion", output_dir
    )
    emotion_model.cpu()
    del emotion_ds
    gc.collect(); torch.cuda.empty_cache()

    # ── 3. Sound ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}\n[Test] SOUND")
    audio_model.to(DEVICE)
    sound_ds = AudioDataset(os.path.join(test_dir, "sound"), sound_label_to_id)
    sound_results = evaluate_sound_task(
        audio_model, sound_ds, sound_label_to_id, output_dir
    )
    audio_model.cpu()
    del sound_ds
    gc.collect(); torch.cuda.empty_cache()

    # ── 4. Patella ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}\n[Test] PATELLA")
    patella_model.to(DEVICE)
    patella_ds = PatellaDataset(os.path.join(test_dir, "patella"), patella_label_to_id)
    patella_results = evaluate_patella_task(
        patella_model, patella_ds, patella_label_to_id, output_dir
    )
    patella_model.cpu()
    del patella_ds
    gc.collect(); torch.cuda.empty_cache()

    # ── 최종 요약 ──────────────────────────────────────────────────────────────
    acc_b = behavior_results.get("accuracy", 0.0)
    acc_e = emotion_results.get("accuracy",  0.0)
    acc_s = sound_results.get("accuracy",    0.0)
    acc_p = patella_results.get("accuracy",  0.0)
    avg   = (acc_b + acc_e + acc_s + acc_p) / 4

    print(f"\n{'='*55}")
    print(f"🏆  Final Test Results")
    print(f"{'='*55}")
    print(f"  Behavior Acc : {acc_b*100:.2f}%")
    print(f"  Emotion  Acc : {acc_e*100:.2f}%")
    print(f"  Sound    Acc : {acc_s*100:.2f}%")
    print(f"  Patella  Acc : {acc_p*100:.2f}%")
    print(f"  Avg      Acc : {avg*100:.2f}%")
    print(f"{'='*55}")

    # ── 요약 바차트 ────────────────────────────────────────────────────────────
    _save_task_summary_bar(
        {"behavior": acc_b, "emotion": acc_e, "sound": acc_s, "patella": acc_p},
        avg, output_dir
    )

    # ── JSON 저장 ──────────────────────────────────────────────────────────────
    summary = {
        "checkpoint"      : ckpt_path,
        "best_val_epoch"  : best_epoch,
        "best_val_avg_acc": best_acc,
        "test_behavior_acc": round(acc_b, 6),
        "test_emotion_acc" : round(acc_e, 6),
        "test_sound_acc"   : round(acc_s, 6),
        "test_patella_acc" : round(acc_p, 6),
        "test_avg_acc"     : round(avg,   6),
        "behavior_detail"  : behavior_results,
        "emotion_detail"   : emotion_results,
        "sound_detail"     : sound_results,
        "patella_detail"   : patella_results,
    }

    json_path = os.path.join(output_dir, "test_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 Saved: {json_path}")

    return summary


def _save_task_summary_bar(acc_dict: dict, avg_acc: float, output_dir: str):
    """태스크별 Accuracy 바차트를 저장한다."""
    tasks  = list(acc_dict.keys())
    accs   = [acc_dict[t] for t in tasks]
    colors = ["steelblue" if a >= 0.7 else "tomato" if a < 0.4 else "orange" for a in accs]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(tasks, accs, color=colors, edgecolor="white", alpha=0.88)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Test Accuracy per Task  (Avg: {avg_acc*100:.2f}%)",
                 fontsize=13, fontweight="bold")
    ax.axhline(avg_acc, color="black", linestyle="--", alpha=0.6,
               label=f"Avg Acc {avg_acc*100:.2f}%")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc*100:.2f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "test_task_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {save_path}")


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pet Normal Omni model test evaluation")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="pet_normal_omni_best.pth",
        help="Path to the best model checkpoint (.pth)",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="files/work/omni_dataset",
        help="WORK_DIR used during training (contains test/ subfolder)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_results",
        help="Directory to save evaluation outputs",
    )
    args = parser.parse_args()

    test(
        ckpt_path  = args.ckpt,
        work_dir   = args.work_dir,
        output_dir = args.output_dir,
    )
