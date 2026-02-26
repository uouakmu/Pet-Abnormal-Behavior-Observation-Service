import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image, ImageFile
import librosa
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# 0. 설정
# =========================
WORK_DIR      = "files/work/omni_dataset"
MODEL_PATH    = "pet_normal_omni_best.pth"
DEVICE        = "cuda:1" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 64
NUM_WORKERS   = 8
SR            = 16000
MAX_AUDIO_LEN = SR * 5
AUDIO_MODEL_NAME = "facebook/wav2vec2-base"

FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)

print(f"🎯 Device: {DEVICE}")

# =========================
# 1. Dataset Classes  (train 코드와 동일 구조)
# =========================
class ImageDataset(Dataset):
    def __init__(self, task_dir):
        self.samples     = []
        self.label_to_id = {}

        for label in sorted(os.listdir(task_dir)):
            label_dir = os.path.join(task_dir, label)
            if not os.path.isdir(label_dir):
                continue
            self.label_to_id[label] = len(self.label_to_id)
            for file in os.listdir(label_dir):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(label_dir, file), label))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
        print(f"  📂 {os.path.basename(task_dir)}: "
              f"{len(self.samples)} samples, {len(self.label_to_id)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.label_to_id[label]


class PatellaDataset(Dataset):
    def __init__(self, task_dir):
        self.samples     = []
        self.label_to_id = {}

        for label in sorted(os.listdir(task_dir)):
            label_dir = os.path.join(task_dir, label)
            if not os.path.isdir(label_dir):
                continue
            self.label_to_id[label] = len(self.label_to_id)
            for file in os.listdir(label_dir):
                if file.lower().endswith('.jpg'):
                    img_path  = os.path.join(label_dir, file)
                    json_path = img_path.replace('.jpg', '.json')
                    if os.path.exists(json_path):
                        self.samples.append((img_path, json_path, label))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
        print(f"  📂 {os.path.basename(task_dir)}: "
              f"{len(self.samples)} samples, {len(self.label_to_id)} classes")

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
            keypoints.extend([
                float(annotation.get('x', 0)),
                float(annotation.get('y', 0)),
            ])
        while len(keypoints) < 18:
            keypoints.append(0.0)
        keypoints = torch.tensor(keypoints[:18], dtype=torch.float32)

        return img, keypoints, self.label_to_id[label]


class AudioDataset(Dataset):
    def __init__(self, task_dir):
        self.samples     = []
        self.label_to_id = {}
        self.id_to_label = {}
        next_id = 0

        for label in sorted(os.listdir(task_dir)):
            label_dir = os.path.join(task_dir, label)
            if not os.path.isdir(label_dir):
                continue
            self.label_to_id[label]  = next_id
            self.id_to_label[next_id] = label
            next_id += 1
            for file in os.listdir(label_dir):
                if file.lower().endswith(('.wav', '.mp3', '.m4a')):
                    self.samples.append((os.path.join(label_dir, file), label))

        print(f"  📂 {os.path.basename(task_dir)}: "
              f"{len(self.samples)} samples, {len(self.label_to_id)} classes")

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
            "labels": torch.tensor(self.label_to_id[label], dtype=torch.long),
        }


def collate_fn_audio(batch):
    return {
        "input_values": torch.stack([b["input_values"] for b in batch]),
        "labels":       torch.stack([b["labels"]       for b in batch]),
    }


# =========================
# 2. Model Definitions  (train 코드와 동일)
# =========================
def _efficientnet_b3_backbone():
    backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = backbone.classifier[1].in_features
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
            nn.Linear(256, num_classes),
        )

    def forward(self, x, keypoints):
        return self.head(torch.cat([self.backbone(x), keypoints], dim=1))


class AudioModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            AUDIO_MODEL_NAME,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, input_values, labels=None):
        return self.model(input_values=input_values, labels=labels)


# =========================
# 3. Loader Helper
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


# =========================
# 4. Evaluation Helpers
# =========================
def _report(task_name, y_true, y_pred, id_to_label):
    """classification_report + confusion matrix 저장."""
    labels_order = [id_to_label[i] for i in range(len(id_to_label))]

    print(f"\n{'─'*55}")
    print(f"  [{task_name.upper()}]  Accuracy: "
          f"{accuracy_score(y_true, y_pred)*100:.2f}%  |  "
          f"Macro-F1: {f1_score(y_true, y_pred, average='macro')*100:.2f}%")
    print(f"{'─'*55}")
    print(classification_report(
        y_true, y_pred,
        target_names=labels_order,
        digits=4,
        zero_division=0,
    ))

    # Confusion matrix 저장
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(labels_order)), max(5, len(labels_order))))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels_order,
        yticklabels=labels_order,
        ax=ax,
    )
    ax.set_title(f"{task_name.capitalize()} — Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    save_path = f"pet_omni_test_cm_{task_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Confusion matrix saved → {save_path}")

    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')


# =========================
# 5. Per-task Test Functions
# =========================
@torch.no_grad()
def test_behavior(model, label_to_id):
    id_to_label = {v: k for k, v in label_to_id.items()}
    ds     = ImageDataset(os.path.join(WORK_DIR, "test", "behavior"))
    loader = make_loader(ds)

    # 저장된 label_to_id 로 재매핑 (test 폴더 순서와 checkpoint 순서가 다를 수 있음)
    local_to_ckpt = {v: label_to_id[k] for k, v in ds.label_to_id.items() if k in label_to_id}

    model.to(DEVICE).eval()
    y_true, y_pred = [], []
    for imgs, labels in tqdm(loader, desc="Test Behavior", leave=False):
        imgs = imgs.to(DEVICE)
        preds = model(imgs).argmax(-1).cpu().tolist()
        y_pred.extend([local_to_ckpt.get(p, p) for p in preds])
        y_true.extend([local_to_ckpt.get(l.item(), l.item()) for l in labels])
    model.cpu()

    return _report("behavior", y_true, y_pred, id_to_label)


@torch.no_grad()
def test_emotion(model, label_to_id):
    id_to_label = {v: k for k, v in label_to_id.items()}
    ds     = ImageDataset(os.path.join(WORK_DIR, "test", "emotion"))
    loader = make_loader(ds)

    local_to_ckpt = {v: label_to_id[k] for k, v in ds.label_to_id.items() if k in label_to_id}

    model.to(DEVICE).eval()
    y_true, y_pred = [], []
    for imgs, labels in tqdm(loader, desc="Test Emotion", leave=False):
        imgs = imgs.to(DEVICE)
        preds = model(imgs).argmax(-1).cpu().tolist()
        y_pred.extend([local_to_ckpt.get(p, p) for p in preds])
        y_true.extend([local_to_ckpt.get(l.item(), l.item()) for l in labels])
    model.cpu()

    return _report("emotion", y_true, y_pred, id_to_label)


@torch.no_grad()
def test_sound(model, label_to_id, id_to_label):
    ds     = AudioDataset(os.path.join(WORK_DIR, "test", "sound"))
    loader = make_loader(ds, is_audio=True)

    local_to_ckpt = {v: label_to_id[k] for k, v in ds.label_to_id.items() if k in label_to_id}

    model.to(DEVICE).eval()
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc="Test Sound", leave=False):
        audios = batch["input_values"].to(DEVICE)
        labels = batch["labels"]
        outputs = model(input_values=audios)
        preds = outputs.logits.argmax(-1).cpu().tolist()
        y_pred.extend([local_to_ckpt.get(p, p) for p in preds])
        y_true.extend([local_to_ckpt.get(l.item(), l.item()) for l in labels])
    model.cpu()

    return _report("sound", y_true, y_pred, id_to_label)


@torch.no_grad()
def test_patella(model, label_to_id):
    id_to_label = {v: k for k, v in label_to_id.items()}
    ds     = PatellaDataset(os.path.join(WORK_DIR, "test", "patella"))
    loader = make_loader(ds)

    local_to_ckpt = {v: label_to_id[k] for k, v in ds.label_to_id.items() if k in label_to_id}

    model.to(DEVICE).eval()
    y_true, y_pred = [], []
    for imgs, keypoints, labels in tqdm(loader, desc="Test Patella", leave=False):
        imgs, keypoints = imgs.to(DEVICE), keypoints.to(DEVICE)
        preds = model(imgs, keypoints).argmax(-1).cpu().tolist()
        y_pred.extend([local_to_ckpt.get(p, p) for p in preds])
        y_true.extend([local_to_ckpt.get(l.item(), l.item()) for l in labels])
    model.cpu()

    return _report("patella", y_true, y_pred, id_to_label)


# =========================
# 6. Summary Plot
# =========================
def save_summary_plot(results: dict):
    """
    results = {
        'behavior': (acc, f1),
        'emotion':  (acc, f1),
        'sound':    (acc, f1),
        'patella':  (acc, f1),
    }
    """
    tasks  = list(results.keys())
    accs   = [results[t][0] * 100 for t in tasks]
    f1s    = [results[t][1] * 100 for t in tasks]

    x      = np.arange(len(tasks))
    width  = 0.35
    colors_acc = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    colors_f1  = ['#8FA8D8', '#EEB98A', '#91C9A2', '#E08C8C']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy (%)',
                   color=colors_acc, edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x + width/2, f1s,  width, label='Macro-F1 (%)',
                   color=colors_f1,  edgecolor='white', linewidth=0.8)

    # 값 레이블
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in tasks], fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Pet Normal Omni — Test Set Performance", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = "pet_omni_test_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  💾 Summary chart saved → {save_path}")


# =========================
# 7. Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Pet Normal Omni — Test Evaluation")
    parser.add_argument("--model",      default=MODEL_PATH, help="checkpoint 경로")
    parser.add_argument("--work_dir",   default=WORK_DIR,   help="work 디렉토리 경로")
    parser.add_argument("--task",       default="all",
                        choices=["all", "behavior", "emotion", "sound", "patella"],
                        help="평가할 태스크 (기본: all)")
    parser.add_argument("--device",     default=DEVICE,     help="cuda:0 / cuda:1 / cpu")
    args = parser.parse_args()

    global WORK_DIR, DEVICE
    WORK_DIR = args.work_dir
    DEVICE   = args.device

    # ── Checkpoint 로드 ──
    print(f"\n📦 Loading checkpoint: {args.model}")
    ckpt = torch.load(args.model, map_location="cpu")

    behavior_label_to_id = ckpt["behavior_label_to_id"]
    emotion_label_to_id  = ckpt["emotion_label_to_id"]
    sound_label_to_id    = ckpt["sound_label_to_id"]
    sound_id_to_label    = ckpt["sound_id_to_label"]
    patella_label_to_id  = ckpt["patella_label_to_id"]

    print(f"  ✅ Best epoch : {ckpt.get('best_epoch', 'N/A')}")
    print(f"  ✅ Best val acc: {ckpt.get('best_acc', 0)*100:.2f}%")
    print(f"  ✅ Classes — behavior:{len(behavior_label_to_id)} | "
          f"emotion:{len(emotion_label_to_id)} | "
          f"sound:{len(sound_label_to_id)} | "
          f"patella:{len(patella_label_to_id)}")

    # ── 모델 복원 ──
    behavior_model = BehaviorModel(len(behavior_label_to_id))
    emotion_model  = EmotionModel(len(emotion_label_to_id))
    patella_model  = PatellaModel(len(patella_label_to_id))
    audio_model    = AudioModel(len(sound_label_to_id))

    behavior_model.load_state_dict(ckpt["behavior_model"])
    emotion_model .load_state_dict(ckpt["emotion_model"])
    patella_model .load_state_dict(ckpt["patella_model"])
    audio_model   .load_state_dict(ckpt["audio_model"])

    print("\n🧪 Starting Test Evaluation...")
    print(f"   Task: {args.task}")

    results = {}

    run_all      = (args.task == "all")
    run_behavior = run_all or args.task == "behavior"
    run_emotion  = run_all or args.task == "emotion"
    run_sound    = run_all or args.task == "sound"
    run_patella  = run_all or args.task == "patella"

    # ── Behavior ──
    if run_behavior:
        print("\n" + "="*55)
        print("🐾  BEHAVIOR TEST")
        print("="*55)
        acc, f1 = test_behavior(behavior_model, behavior_label_to_id)
        results["behavior"] = (acc, f1)

    # ── Emotion ──
    if run_emotion:
        print("\n" + "="*55)
        print("😊  EMOTION TEST")
        print("="*55)
        acc, f1 = test_emotion(emotion_model, emotion_label_to_id)
        results["emotion"] = (acc, f1)

    # ── Sound ──
    if run_sound:
        print("\n" + "="*55)
        print("🔊  SOUND TEST")
        print("="*55)
        acc, f1 = test_sound(audio_model, sound_label_to_id, sound_id_to_label)
        results["sound"] = (acc, f1)

    # ── Patella ──
    if run_patella:
        print("\n" + "="*55)
        print("🦴  PATELLA TEST")
        print("="*55)
        acc, f1 = test_patella(patella_model, patella_label_to_id)
        results["patella"] = (acc, f1)

    # ── 최종 요약 ──
    print("\n" + "="*55)
    print("📊  FINAL SUMMARY")
    print("="*55)
    for task, (acc, f1) in results.items():
        print(f"  {task:<12}  Acc: {acc*100:6.2f}%  |  Macro-F1: {f1*100:6.2f}%")
    if len(results) > 1:
        avg_acc = np.mean([v[0] for v in results.values()])
        avg_f1  = np.mean([v[1] for v in results.values()])
        print(f"  {'─'*45}")
        print(f"  {'AVERAGE':<12}  Acc: {avg_acc*100:6.2f}%  |  Macro-F1: {avg_f1*100:6.2f}%")
        save_summary_plot(results)

    print("\n✅ Test evaluation complete.")


if __name__ == "__main__":
    main()
