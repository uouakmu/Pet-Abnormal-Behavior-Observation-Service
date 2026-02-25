import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor  # 🔥 wav2vec2 전용
import librosa
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# =========================
# 설정 (wav2vec2 전용)
# =========================
SR = 16000
MAX_LEN = SR * 5
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MODEL_NAME = "facebook/wav2vec2-base"

# wav2vec2 전용
FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

class Wav2Vec2TestDataset(Dataset):
    def __init__(self, test_dir):
        self.samples = []
        self.label_to_id = {}
        self.id_to_label = {}
        next_id = 0
        
        for label in sorted(os.listdir(test_dir)):
            label_dir = os.path.join(test_dir, label)
            if os.path.isdir(label_dir):
                self.label_to_id[label] = next_id
                self.id_to_label[next_id] = label
                next_id += 1
                
                for filename in os.listdir(label_dir):
                    if filename.lower().endswith(('.wav', '.mp3', '.m4a')):
                        self.samples.append((os.path.join(label_dir, filename), label))
        
        print(f"🔍 Wav2Vec2 Test: {len(self.samples)} files, {len(self.label_to_id)} classes")
        print("Classes:", list(self.label_to_id.keys()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_name = self.samples[idx]
        label_id = self.label_to_id[label_name]
        
        try:
            waveform, _ = librosa.load(path, sr=SR, mono=True)
        except:
            waveform = np.zeros(MAX_LEN)
        
        if len(waveform) > MAX_LEN:
            waveform = waveform[:MAX_LEN]
        else:
            waveform = np.pad(waveform, (0, MAX_LEN - len(waveform)))
        
        inputs = FEATURE_EXTRACTOR(waveform, sampling_rate=SR, return_tensors="pt")
        input_values = inputs.input_values.squeeze(0)
        
        return {
            "input_values": input_values,
            "labels": torch.tensor(label_id, dtype=torch.long),
            "label_name": label_name
        }

def collate_fn(batch):
    input_values = torch.stack([item["input_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    label_names = [item["label_name"] for item in batch]
    return {"input_values": input_values, "labels": labels, "label_names": label_names}

def test_wav2vec2_model():
    """wav2vec2 모델 테스트"""
    test_dir = "files/work/sound_dataset/test"  # wav2vec2 WORK_DIR
    
    # 1. 모델 로드
    num_classes = 16  # 자동 감지 또는 고정
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    model_path = 'pet_sound_best.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✅ {model_path} loaded!")
    else:
        print(f"⚠️  {model_path} 없음. 랜덤 모델 테스트")
    
    model.to(DEVICE)
    model.eval()
    
    # 2. 테스트 데이터셋
    test_ds = Wav2Vec2TestDataset(test_dir)
    
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )
    
    # 3. 테스트 실행
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n🔍 Wav2Vec2 Testing...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            inputs = batch["input_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            outputs = model(input_values=inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = outputs.logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 4. 결과
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean()
    print(f"\n🎯 Wav2Vec2 Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # 클래스 이름
    id_to_label = test_ds.id_to_label
    pred_labels = [id_to_label[p] for p in all_preds]
    true_labels = [id_to_label[t] for t in all_labels]
    
    # 5. 상세 리포트
    print("\n📋 Wav2Vec2 Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=list(id_to_label.values())))
      
    # 7. Top 오류 분석
    errors = [(i, true_labels[i], pred_labels[i], all_probs[i][all_preds[i]]) 
              for i in range(len(all_labels)) if all_preds[i] != all_labels[i]]
    
    if errors:
        print(f"\n❌ Wav2Vec2 Top-5 Errors (True → Pred, Conf):")
        errors.sort(key=lambda x: x[3], reverse=True)
        for i, (idx, true, pred, conf) in enumerate(errors[:5]):
            print(f"  {i+1}. {true} → {pred} (conf: {conf:.3f})")
    

def predict_wav2vec2_single(file_path):
    """단일 파일 예측 (wav2vec2)"""
    test_dir = "files/work/sound_dataset/test"
    test_ds = Wav2Vec2TestDataset(test_dir)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=16, ignore_mismatched_sizes=True
    )
    
    model_path = 'pet_sound_best.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    model.to(DEVICE)
    model.eval()
    
    waveform, _ = librosa.load(file_path, sr=SR, mono=True)
    if len(waveform) > MAX_LEN:
        waveform = waveform[:MAX_LEN]
    else:
        waveform = np.pad(waveform, (0, MAX_LEN - len(waveform)))
    
    inputs = FEATURE_EXTRACTOR(waveform, sampling_rate=SR, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = outputs.logits.argmax(dim=-1).item()
    
    id_to_label = test_ds.id_to_label
    top3 = torch.topk(probs, 3).indices[0].cpu().numpy()
    print(f"\n🎵 파일: {os.path.basename(file_path)}")
    print("🔥 Wav2Vec2 Top-3:")
    for i, tid in enumerate(top3):
        prob = probs[0, tid].item()
        print(f"  {i+1}. {id_to_label[tid]}: {prob:.3f}")
    print(f"📈 최종 예측: {id_to_label[pred_id]}")

if __name__ == "__main__":
    test_wav2vec2_model()
    
    # 단일 예측 예시
    # predict_wav2vec2_single("path/to/test.wav")
