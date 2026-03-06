import torch

cat_ckpt = torch.load('AI_pth/cat_abnormal_omni_best.pth', map_location='cpu')
print("CAT Skin classes:", cat_ckpt.get('cat_skin_classes', []))
print("CAT Eyes classes:", cat_ckpt.get('cat_eyes_classes', []))

dog_ckpt = torch.load('AI_pth/dog_abnormal_omni_best.pth', map_location='cpu')
print("DOG Skin classes:", dog_ckpt.get('dog_skin_classes', []))
print("DOG Eyes classes:", dog_ckpt.get('dog_eyes_classes', []))
