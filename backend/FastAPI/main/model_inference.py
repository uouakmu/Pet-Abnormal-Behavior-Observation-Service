import io
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s
from torchvision import transforms
from PIL import Image
import numpy as np

# ─────────────────────────── MODELS ───────────────────────────
def _efficientnet_backbone():
    b = efficientnet_v2_s(weights=None)
    feat = b.classifier[1].in_features
    b.classifier = nn.Identity()
    return b, feat

class _FusionBase(nn.Module):
    def __init__(self, num_classes, json_feat_dim, head_hidden):
        super().__init__()
        self.backbone, img_feat = _efficientnet_backbone()
        self.feat_branch = nn.Sequential(
            nn.Linear(json_feat_dim, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(32, 32), nn.GELU(),
        )
        fused  = img_feat + 32
        layers = [nn.Dropout(0.5), nn.Linear(fused, head_hidden[0]),
                  nn.BatchNorm1d(head_hidden[0]), nn.GELU()]
        for i in range(1, len(head_hidden)):
            layers += [nn.Dropout(0.4), nn.Linear(head_hidden[i-1], head_hidden[i]),
                       nn.BatchNorm1d(head_hidden[i]), nn.GELU()]
        layers += [nn.Dropout(0.4), nn.Linear(head_hidden[-1], num_classes)]
        self.head = nn.Sequential(*layers)

    def forward(self, img, feat):
        return self.head(torch.cat([self.backbone(img), self.feat_branch(feat)], dim=1))

class SkinModelCat(_FusionBase):
    def __init__(self, num_classes, json_feat_dim=5):
        super().__init__(num_classes, json_feat_dim, head_hidden=[256])

class EyesModelCat(_FusionBase):
    def __init__(self, num_classes, json_feat_dim=5):
        super().__init__(num_classes, json_feat_dim, head_hidden=[512])

class SkinModelDog(_FusionBase):
    def __init__(self, num_classes, json_feat_dim=5):
        super().__init__(num_classes, json_feat_dim, head_hidden=[256])

class EyesModelDog(_FusionBase):
    def __init__(self, num_classes, json_feat_dim=5):
        super().__init__(num_classes, json_feat_dim, head_hidden=[512, 256])

# ─────────────────────────── INFERENCE LOGIC ───────────────────────────
class AIEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Image Transform (same as test script)
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        self.models = {
            'cat': {'skin': None, 'eyes': None},
            'dog': {'skin': None, 'eyes': None}
        }
        self.classes = {
            'cat': {'skin': [], 'eyes': []},
            'dog': {'skin': [], 'eyes': []}
        }
        self.json_feat_dim = 5
        self.is_loaded = False

    def load_models(self):
        print(f"Loading AI Models to device: {self.device}...")
        try:
            # ==== CAT MODEL LOAD ====
            cat_ckpt = torch.load('/app/AI_pth/cat_abnormal_omni_best.pth', map_location=self.device)
            self.classes['cat']['skin'] = cat_ckpt['cat_skin_classes']
            self.classes['cat']['eyes'] = cat_ckpt['cat_eyes_classes']
            self.json_feat_dim = cat_ckpt.get('json_feat_dim', 5)
            
            cat_skin = SkinModelCat(len(self.classes['cat']['skin']), self.json_feat_dim).to(self.device).eval()
            cat_eyes = EyesModelCat(len(self.classes['cat']['eyes']), self.json_feat_dim).to(self.device).eval()
            
            cat_skin.load_state_dict(cat_ckpt['skin_model'])
            cat_eyes.load_state_dict(cat_ckpt['eyes_model'])
            
            self.models['cat']['skin'] = cat_skin
            self.models['cat']['eyes'] = cat_eyes
            print("loaded Cat Models.")
            
            # ==== DOG MODEL LOAD ====
            dog_ckpt = torch.load('/app/AI_pth/dog_abnormal_omni_best.pth', map_location=self.device)
            self.classes['dog']['skin'] = dog_ckpt['dog_skin_classes']
            self.classes['dog']['eyes'] = dog_ckpt['dog_eyes_classes']
            
            dog_skin = SkinModelDog(len(self.classes['dog']['skin']), self.json_feat_dim).to(self.device).eval()
            dog_eyes = EyesModelDog(len(self.classes['dog']['eyes']), self.json_feat_dim).to(self.device).eval()
            
            dog_skin.load_state_dict(dog_ckpt['skin_model'])
            dog_eyes.load_state_dict(dog_ckpt['eyes_model'])
            
            self.models['dog']['skin'] = dog_skin
            self.models['dog']['eyes'] = dog_eyes
            print("loaded Dog Models.")
            
            self.is_loaded = True
            print("AI Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_loaded = False

    def analyze(self, image_bytes: bytes, pet_type: str, disease_type: str):
        if not self.is_loaded:
            return {"status": "error", "message": "Models are not loaded yet."}

        # Normalize types
        pet_t = 'cat' if '고양이' in pet_type or 'cat' in pet_type.lower() else 'dog'
        dis_t = 'skin' if 'skin' in disease_type.lower() else 'eyes'

        model = self.models[pet_t][dis_t]
        class_list = self.classes[pet_t][dis_t]

        if model is None:
             return {"status": "error", "message": "Unsupported combination"}

        try:
            # Process Image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Dummy JSON feature constraint
            feat_tensor = torch.zeros((1, self.json_feat_dim), dtype=torch.float32).to(self.device)

            # Inference
            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        output = model(img_tensor, feat_tensor)
                else:
                    output = model(img_tensor, feat_tensor)
                    
                probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                    
            pred_idx = np.argmax(probs)
            pred_class = class_list[pred_idx]
            pred_prob = float(probs[pred_idx]) * 100
            
            return {
                "status": "success",
                "diagnosis": pred_class,
                "probability": pred_prob,
                "pet_type_detected": pet_t,
                "disease_category": dis_t
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Inference failed: {str(e)}"}

# Global engine instance
ai_engine = AIEngine()
