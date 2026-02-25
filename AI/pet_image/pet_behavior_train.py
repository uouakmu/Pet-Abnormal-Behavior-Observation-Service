import os
import random
import shutil
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings("ignore")

# =========================
# 0. 설정 (통합)
# =========================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)

DATA_ROOT = "files/1_Animal_Behavior"
WORK_DIR = "files/work/behavior_dataset"
MAX_SAMPLES = 100_000  # 총 10만
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100

# NUM_WORKERS 통합 관리
NUM_WORKERS = 4
PERSISTENT_WORKERS = True
PIN_MEMORY = True

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"🐾 Pet Behavior | {DEVICE} | {MAX_SAMPLES:,} samples | Epochs: {EPOCHS}")

# ResNet18 전처리
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomErasing(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class BehaviorDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.label_to_id = {}
        label_id = 0
        
        for label in sorted(os.listdir(root)):
            label_dir = os.path.join(root, label)
            if os.path.isdir(label_dir):
                self.label_to_id[label] = label_id
                label_id += 1
                
                for img_file in os.listdir(label_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(label_dir, img_file), label))
        
        print(f"📁 {os.path.basename(root)}: {len(self.samples):,} images, {len(self.label_to_id)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label_name = self.samples[idx]
        label_id = self.label_to_id[label_name]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_id

def split_dataset():
    """75만 → 10만 균등 샘플링"""
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    
    os.makedirs(os.path.join(WORK_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, "val"), exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, "test"), exist_ok=True)

    print("🔍 Collecting all images...")
    all_samples = []
    class_dirs = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    
    for class_dir in tqdm(class_dirs, desc="Classes"):
        class_path = os.path.join(DATA_ROOT, class_dir)
        imgs = []
        for root, _, files in os.walk(class_path):
            imgs.extend([os.path.join(root, f) for f in files 
                        if f.lower().endswith(('.jpg','.jpeg','.png'))])
        all_samples.extend([(class_dir, img) for img in imgs])
    
    print(f"✅ Total: {len(all_samples):,} images, {len(class_dirs)} classes")

    # 클래스별 균등 샘플링
    label_count = Counter(label for label, _ in all_samples)
    samples_per_class = MAX_SAMPLES // len(class_dirs)
    print(f"🎯 클래스당 샘플: {samples_per_class:,}개")

    balanced_samples = []
    for label in class_dirs:
        label_imgs = [img for l, img in all_samples if l == label]
        n = min(samples_per_class, len(label_imgs))
        balanced_samples.extend(random.sample(label_imgs, n))
    
    print(f"📊 균형 샘플링 완료: {len(balanced_samples):,}개")

    # 8:1:1 클래스별 stratified split
    label_samples = defaultdict(list)
    for label, img in all_samples:
        if img in balanced_samples:
            label_samples[label].append(img)
    
    train_imgs, val_imgs, test_imgs = [], [], []
    for label, imgs in label_samples.items():
        random.shuffle(imgs)
        n_train = int(len(imgs) * 0.8)
        n_val = int(len(imgs) * 0.1)
        
        train_imgs.extend([(label, p) for p in imgs[:n_train]])
        val_imgs.extend([(label, p) for p in imgs[n_train:n_train+n_val]])
        test_imgs.extend([(label, p) for p in imgs[n_train+n_val:]])
    
    # 파일 복사
    for split_name, files in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
        print(f"📂 Copying {split_name}: {len(files):,}")
        for label, src_path in tqdm(files, desc=split_name):
            dst_dir = os.path.join(WORK_DIR, split_name, label)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src_path, os.path.join(dst_dir, os.path.basename(src_path)))
    
    print("✅ Dataset split complete!")

def plot_history(history):
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]
    val_accs = [h['val_acc'] for h in history]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2)
    axes[0].set_title('Behavior Classification Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, val_accs, 'g-^', linewidth=3, markersize=8)
    best_epoch = np.argmax(val_accs) + 1
    best_acc = max(val_accs)
    axes[1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_acc:.3f}')
    axes[1].set_title('Validation Accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pet_behavior_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Best Val Acc: {best_acc:.4f} (Epoch {best_epoch})")

def train_behavior():
    split_dataset()
    
    # 데이터셋
    train_ds = BehaviorDataset(os.path.join(WORK_DIR, 'train'), transform_train)
    val_ds = BehaviorDataset(os.path.join(WORK_DIR, 'val'), transform_val)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS//2, persistent_workers=PERSISTENT_WORKERS, pin_memory=PIN_MEMORY)
    
    num_classes = len(train_ds.label_to_id)
    print(f"\n🚀 ResNet18 | {num_classes} classes | {len(train_loader)} train batches")
    
    # ResNet18
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0
    history = []
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Train
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Train E{epoch+1:2d}")
        
        for batch in train_pbar:
            imgs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        scheduler.step()
        
        # Val
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                pred = outputs.argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history.append({
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': val_acc
        })
        
        epoch_time = time.time() - start_time
        print(f"\n📊 E{epoch+1:3d} | Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
              f"Acc: {val_acc:.4f} | {epoch_time:.0f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_to_id': train_ds.label_to_id,
                'best_epoch': epoch+1,
                'best_acc': best_acc,
                'history': history
            }, 'pet_behavior_best.pth')
            print(f"💾 BEST: {best_acc:.4f}")
    
    plot_history(history)
    print(f"\n🎉 Pet Behavior 학습 완료! pet_behavior_best_ex.pth 저장")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    train_behavior()
