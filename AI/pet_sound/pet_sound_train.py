import os
import random
import shutil
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, get_linear_schedule_with_warmup  # 🔥 변경
import librosa
import numpy as np
from collections import defaultdict, Counter
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F

# =========================
# 0. 개선된 설정 (wav2vec2 전용)
# =========================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_ROOT = "files/3_Animal_Sound"
WORK_DIR = "files/work/sound_dataset"
MIN_SAMPLES_PER_CLASS = 50
SR = 16000
MAX_LEN = SR * 5
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-5

NUM_WORKERS = 2
PERSISTENT_WORKERS = False
PIN_MEMORY = True

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"🎯 {DEVICE}, Workers: {NUM_WORKERS}")

# 🔥 wav2vec2 전용 FeatureExtractor & Model
MODEL_NAME = "facebook/wav2vec2-base"
FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

# 데이터 증강 함수 (동일)
def augment_audio(waveform, p=0.5):
    if random.random() > p:
        return waveform
    
    n_steps = random.uniform(-2, 2)
    waveform = librosa.effects.pitch_shift(waveform, sr=SR, n_steps=n_steps)
    
    rate = random.uniform(0.9, 1.1)
    stretched = librosa.effects.time_stretch(waveform, rate=rate)
    if len(stretched) > MAX_LEN:
        stretched = stretched[:MAX_LEN]
    else:
        stretched = np.pad(stretched, (0, MAX_LEN - len(stretched)))
    waveform = stretched
    
    noise = np.random.normal(0, 0.003, len(waveform))
    waveform = waveform * 0.99 + noise
    
    return waveform

# split_dataset_improved(), ImprovedAudioDataset (동일 - 생략)
def split_dataset_improved():
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    
    os.makedirs(os.path.join(WORK_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, "val"), exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, "test"), exist_ok=True)

    all_samples = []
    class_dirs = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    
    print(f"Found {len(class_dirs)} classes")
    for class_dir in tqdm(class_dirs, desc="Collecting"):
        class_path = os.path.join(DATA_ROOT, class_dir)
        files = [os.path.join(root, f) for root, _, fs in os.walk(class_path) 
                for f in fs if f.lower().endswith(('.wav', '.mp3', '.m4a'))]
        all_samples.extend([(class_dir, f) for f in files])
    
    label_count = Counter(label for label, _ in all_samples)
    print("Per class:", dict(sorted(label_count.items(), key=lambda x: x[1], reverse=True)[:5]))
    
    train_samples = []
    for label in class_dirs:
        label_files = [path for l, path in all_samples if l == label]
        n_needed = max(MIN_SAMPLES_PER_CLASS, len(label_files) * 2)
        sampled = random.choices(label_files, k=min(n_needed, len(label_files)*3))
        train_samples.extend([(label, f) for f in sampled])
    
    print(f"Oversampled train: {len(train_samples)} samples")
    
    label_samples = defaultdict(list)
    for label, path in train_samples:
        label_samples[label].append(path)
    
    train_files, val_files = [], []
    for label, paths in label_samples.items():
        random.shuffle(paths)
        n_train = int(len(paths) * 0.85)
        train_files.extend([(label, p) for p in paths[:n_train]])
        val_files.extend([(label, p) for p in paths[n_train:n_train+max(5, len(paths)//10)]])
    
    test_files = []
    for label in class_dirs:
        label_files = [path for l, path in all_samples if l == label]
        test_files.extend([(label, f) for f in label_files[:max(10, len(label_files)//5)]])
    
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        print(f"Copying {split_name}: {len(files)}")
        for label, src_path in tqdm(files, desc=split_name):
            dst_dir = os.path.join(WORK_DIR, split_name, label)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src_path, os.path.join(dst_dir, os.path.basename(src_path)))
    
    print("✅ Wav2Vec2 dataset ready!")

class ImprovedAudioDataset(Dataset):
    def __init__(self, split_dir, augment=False):
        self.samples = []
        self.label_to_id = {}
        self.id_to_label = {}
        self.augment = augment
        next_id = 0
        
        for label in sorted(os.listdir(split_dir)):
            label_dir = os.path.join(split_dir, label)
            if os.path.isdir(label_dir):
                self.label_to_id[label] = next_id
                self.id_to_label[next_id] = label
                next_id += 1
                
                for filename in os.listdir(label_dir):
                    if filename.lower().endswith(('.wav', '.mp3', '.m4a')):
                        self.samples.append((os.path.join(label_dir, filename), label))
        
        print(f"{os.path.basename(split_dir)}: {len(self.samples)} files, {len(self.label_to_id)} classes, augment={augment}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_name = self.samples[idx]
        label_id = self.label_to_id[label_name]
        
        try:
            waveform, _ = librosa.load(path, sr=SR, mono=True)
        except:
            waveform = np.zeros(MAX_LEN)
        
        if self.augment:
            waveform = augment_audio(waveform)
        
        if len(waveform) > MAX_LEN:
            waveform = waveform[:MAX_LEN]
        else:
            waveform = np.pad(waveform, (0, MAX_LEN - len(waveform)))
        
        inputs = FEATURE_EXTRACTOR(waveform, sampling_rate=SR, return_tensors="pt")
        input_values = inputs.input_values.squeeze(0)
        
        return {
            "input_values": input_values,
            "labels": torch.tensor(label_id, dtype=torch.long)
        }

def collate_fn(batch):
    input_values = torch.stack([item["input_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_values": input_values, "labels": labels}

def train():
    split_dataset_improved()
    
    train_ds = ImprovedAudioDataset(os.path.join(WORK_DIR, 'train'), augment=True)
    val_ds = ImprovedAudioDataset(os.path.join(WORK_DIR, 'val'), augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=NUM_WORKERS//2, collate_fn=collate_fn)
    
    print(f"📦 Train: {len(train_loader)} batches, Val: {len(val_loader)}")
    
    num_classes = len(train_ds.label_to_id)
    
    # 🔥 Wav2Vec2ForSequenceClassification 사용
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    # 클래스 가중치
    train_labels = [train_ds.label_to_id[label] for _, label in train_ds.samples]
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_labels)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(DEVICE))
    
    # Warmup 스케줄러
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = 100
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    scaler = torch.amp.GradScaler(device='cuda') if torch.cuda.is_available() else None
    
    best_acc = 0
    history = []
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Wav2Vec2 E{epoch+1}")
        
        for batch in pbar:
            inputs = batch["input_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(input_values=inputs, labels=labels)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                
                outputs = model(input_values=inputs, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                
                pred = outputs.logits.argmax(-1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        print(f"\n📊 E{epoch+1}: Train {train_loss/len(train_loader):.4f}, "
              f"Val {val_loss/len(val_loader):.4f}, Acc {val_acc:.4f}")
        
        history.append({'epoch': epoch+1, 'train_loss': train_loss/len(train_loader), 
                       'val_loss': val_loss/len(val_loader), 'val_acc': val_acc})
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'pet_sound_best.pth')
            print(f"💾 New best: {best_acc:.4f}")
    
    # 그래프
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot([h['train_loss'] for h in history], 'b-', label='Train')
    plt.plot([h['val_loss'] for h in history], 'r-', label='Val')
    plt.legend(); plt.title('Wav2Vec2 Loss'); plt.grid(True, alpha=0.3)
    
    plt.subplot(122)
    plt.plot([h['val_acc'] for h in history], 'g-', linewidth=3)
    plt.title('Wav2Vec2 Val Accuracy'); plt.ylim(0,1); plt.grid(True, alpha=0.3)
    # plt.savefig('wav2vec2_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 Wav2Vec2 Best Val Acc: {best_acc:.4f}")
    print("Saved: pet_sound_best.pth")

if __name__ == "__main__":
    train()