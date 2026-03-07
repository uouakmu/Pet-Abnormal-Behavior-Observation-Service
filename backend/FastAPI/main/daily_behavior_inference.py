import io
import os
import gc
import json
import torch
import librosa
import numpy as np
import tempfile
import cv2
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# ───────────────────────────────── CONFIG ─────────────────────────────────────
AUDIO_MODEL_NAME = "facebook/wav2vec2-base"
SR = 16000
MAX_AUDIO_LEN = SR * 5
IMG_SIZE = 384
FEAT_DIM = 59

TRANSFORM_TEST = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─────────────────────────────── MODELS ───────────────────────────────────────
def _efficientnet_backbone():
    b = efficientnet_v2_s(weights=None)
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

class PatellaModel(nn.Module):
    def __init__(self, num_classes, feat_dim=FEAT_DIM):
        super().__init__()
        self.backbone, img_feat = _efficientnet_backbone()
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        fused_feat = img_feat + 128   # 1408
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(fused_feat, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, img, feat):
        img_f = self.backbone(img)
        kp_f  = self.feat_branch(feat)
        return self.head(torch.cat([img_f, kp_f], dim=1))

# ─────────────────────────────── ENGINE ───────────────────────────────────────
class DailyBehaviorEngine:
    def __init__(self):
        self.is_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.cat_behavior_classes = []
        self.cat_emotion_classes = []
        self.cat_sound_classes = []

        self.dog_behavior_classes = []
        self.dog_emotion_classes = []
        self.dog_sound_classes = []
        self.dog_patella_classes = []
        
        self.feature_extractor = None

    def _load_cat_models(self, ckpt_path):
        if not os.path.exists(ckpt_path): return False
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.cat_behavior_classes = ckpt.get("cat_behavior_classes", [])
        self.cat_emotion_classes = ckpt.get("cat_emotion_classes", [])
        self.cat_sound_classes = ckpt.get("cat_sound_classes", [])
        
        if not self.cat_behavior_classes: return False # Invalid checkpoint

        self.cat_behavior_model = ImageModel(len(self.cat_behavior_classes)).to(self.device)
        self.cat_emotion_model = ImageModel(len(self.cat_emotion_classes)).to(self.device)
        self.cat_audio_model = AudioModel(len(self.cat_sound_classes)).to(self.device)
        
        self.cat_behavior_model.load_state_dict(ckpt["behavior_model"])
        self.cat_emotion_model.load_state_dict(ckpt["emotion_model"])
        self.cat_audio_model.load_state_dict(ckpt["audio_model"])
        
        self.cat_behavior_model.eval()
        self.cat_emotion_model.eval()
        self.cat_audio_model.eval()
        return True

    def _load_dog_models(self, ckpt_path):
        if not os.path.exists(ckpt_path): return False
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.dog_behavior_classes = ckpt.get("dog_behavior_classes", [])
        self.dog_emotion_classes = ckpt.get("dog_emotion_classes", [])
        self.dog_sound_classes = ckpt.get("dog_sound_classes", [])
        self.dog_patella_classes = ckpt.get("dog_patella_classes", [])
        
        if not self.dog_behavior_classes: return False

        self.dog_behavior_model = ImageModel(len(self.dog_behavior_classes)).to(self.device)
        self.dog_emotion_model = ImageModel(len(self.dog_emotion_classes)).to(self.device)
        self.dog_audio_model = AudioModel(len(self.dog_sound_classes)).to(self.device)
        self.dog_patella_model = PatellaModel(len(self.dog_patella_classes), feat_dim=FEAT_DIM).to(self.device)

        self.dog_behavior_model.load_state_dict(ckpt["behavior_model"])
        self.dog_emotion_model.load_state_dict(ckpt["emotion_model"])
        self.dog_audio_model.load_state_dict(ckpt["audio_model"])
        self.dog_patella_model.load_state_dict(ckpt["patella_model"])

        self.dog_behavior_model.eval()
        self.dog_emotion_model.eval()
        self.dog_audio_model.eval()
        self.dog_patella_model.eval()
        return True

    def load_models(self):
        print("Loading Daily Behavior Omni Models...")
        base_dir = "/app/AI_pth" if os.path.exists("/app/AI_pth") else "AI Model/AI_pth"
        dog_path = os.path.join(base_dir, "dog_normal_omni_best.pth")
        cat_path = os.path.join(base_dir, "cat_normal_omni_best.pth")
        
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
        except Exception as e:
            print(f"Error loading feature extractor: {e}")
            
        cat_loaded = self._load_cat_models(cat_path)
        dog_loaded = self._load_dog_models(dog_path)
        
        if cat_loaded or dog_loaded:
            self.is_loaded = True
            print(f"Models loaded successfully! (Cat: {cat_loaded}, Dog: {dog_loaded})")
        else:
            print(f"Failed to load Omni models. Check {base_dir} paths.")

    def extract_frames(self, video_path: str):
        # Extract 1 frame per second to analyze overall behavior
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        frame_interval = int(round(fps))  # 1 frame per second
        
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                tensor = TRANSFORM_TEST(img)
                frames.append(tensor)
            count += 1
            if len(frames) >= 20:  # Max 20 frames (limit inference time)
                break
        cap.release()
        
        if frames:
            return torch.stack(frames).to(self.device)
        return None

    def extract_audio_tensor(self, video_path: str):
        if self.feature_extractor is None: return None
        try:
            w, _ = librosa.load(video_path, sr=SR, mono=True)
            w = (w[:MAX_AUDIO_LEN] if len(w) > MAX_AUDIO_LEN else np.pad(w, (0, MAX_AUDIO_LEN - len(w))).astype(np.float32))
            inp = self.feature_extractor(w, sampling_rate=SR, return_tensors="pt")
            return inp.input_values.to(self.device)
        except Exception as e:
            print(f"Audio extraction error: {e}")
            return None

    def _infer_image_model(self, model, frames, classes):
        if frames is None or frames.shape[0] == 0:
            return "Unknown", 0.0
        with torch.no_grad():
            with torch.amp.autocast("cuda" if "cuda" in self.device else "cpu"):
                outputs = model(frames)
                probs = torch.softmax(outputs, dim=1)
                mean_probs = probs.mean(dim=0)
                max_prob, max_idx = torch.max(mean_probs, dim=0)
                return classes[max_idx.item()], round(max_prob.item(), 3)

    def _infer_audio_model(self, model, audio_tensor, classes):
        if audio_tensor is None or audio_tensor.shape[0] == 0:
            return "Unknown", 0.0
        with torch.no_grad():
            with torch.amp.autocast("cuda" if "cuda" in self.device else "cpu"):
                outputs = model(audio_tensor).logits
                probs = torch.softmax(outputs, dim=1)
                max_prob, max_idx = torch.max(probs[0], dim=0)
                return classes[max_idx.item()], round(max_prob.item(), 3)
                
    def _infer_patella_model(self, model, frames, classes):
        if frames is None or frames.shape[0] == 0:
            return "Unknown", 0.0
        # Passing Zero vector dummy features as keypoints are usually extracted dynamically in edge devices 
        # or separate pose estimators.
        feat = torch.zeros((frames.shape[0], FEAT_DIM), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            with torch.amp.autocast("cuda" if "cuda" in self.device else "cpu"):
                outputs = model(frames, feat)
                probs = torch.softmax(outputs, dim=1)
                mean_probs = probs.mean(dim=0)
                max_prob, max_idx = torch.max(mean_probs, dim=0)
                return classes[max_idx.item()], round(max_prob.item(), 3)

    def analyze_clip(self, video_bytes: bytes, pet_type: str) -> dict:
        if not self.is_loaded:
            return {"status": "error", "message": "Behavior models are not loaded yet."}

        is_dog = (pet_type.lower() == "dog")
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(video_bytes)
                temp_video_path = tmp_file.name

            frames_tensor = self.extract_frames(temp_video_path)
            audio_tensor = self.extract_audio_tensor(temp_video_path)
            os.remove(temp_video_path)

            if is_dog:
                beh, b_conf = self._infer_image_model(self.dog_behavior_model, frames_tensor, self.dog_behavior_classes)
                emo, e_conf = self._infer_image_model(self.dog_emotion_model, frames_tensor, self.dog_emotion_classes)
                snd, s_conf = self._infer_audio_model(self.dog_audio_model, audio_tensor, self.dog_sound_classes)
                pat, p_conf = self._infer_patella_model(self.dog_patella_model, frames_tensor, self.dog_patella_classes)
                
                return {
                    "status": "success",
                    "pet_type_analyzed": "dog",
                    "behavior_analysis": {"detected_behavior": beh, "confidence": b_conf, "emotion": emo, "emotion_confidence": e_conf},
                    "audio_analysis": {"detected_sound": snd, "confidence": s_conf},
                    "patella_analysis": {"status": pat, "confidence": p_conf},
                    "summary": f"{beh} with {snd}"
                }
            else:
                beh, b_conf = self._infer_image_model(self.cat_behavior_model, frames_tensor, self.cat_behavior_classes)
                emo, e_conf = self._infer_image_model(self.cat_emotion_model, frames_tensor, self.cat_emotion_classes)
                snd, s_conf = self._infer_audio_model(self.cat_audio_model, audio_tensor, self.cat_sound_classes)
                
                return {
                    "status": "success",
                    "pet_type_analyzed": "cat",
                    "behavior_analysis": {"detected_behavior": beh, "confidence": b_conf, "emotion": emo, "emotion_confidence": e_conf},
                    "audio_analysis": {"detected_sound": snd, "confidence": s_conf},
                    "summary": f"{beh} with {snd}"
                }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": f"Behavior inference failed: {str(e)}"}

daily_behavior_engine = DailyBehaviorEngine()
