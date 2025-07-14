import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import FPN
from torchvision.models.segmentation import deeplabv3_resnet50
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import requests
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Google Drive download utility
def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# Model loading
def load_models():
    models = {}
    try:
        models['fpn'] = load_fpn_model()
        models['deeplab'] = load_deeplabv3_model()
        models['attenunet'] = load_attenunet_model()
        return models
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None

def load_fpn_model():
    model_path = 'classifier_state_dict.pth'
    file_id = '1gkUXdb4S2Dbm0mTotxFeDpaoO9emmd0X'   
    if not os.path.exists(model_path):
        print("Downloading FPN model...")
        download_file_from_google_drive(file_id, model_path)

    model = FPN(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k.replace('model.', '').replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("FPN model loaded.")
    return model

def load_deeplabv3_model():
    model_path = 'deeplabv3_model_weights.pth'
    file_id = '1ZfWGp384hjY3T1HgxajNDRTdIvfVjVym'  # 🔁 Replace with your file ID
    if not os.path.exists(model_path):
        print("Downloading DeepLabV3+ model...")
        download_file_from_google_drive(file_id, model_path)

    model = deeplabv3_resnet50(pretrained=False, num_classes=1)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)

    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k.replace('model.', '').replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("DeepLabV3+ model loaded.")
    return model

def load_attenunet_model():
    model_path = 'attunet_state_dict.pth'
    file_id = '15CQVZymXH4qFQ6rnxJEQ0k7wkrNYeflg'  
    if not os.path.exists(model_path):
        print("Downloading Attention UNet model...")
        download_file_from_google_drive(file_id, model_path)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16],
        attention_type='scse'  # Use only if trained with attention
    )

    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('model.', '')
        if 'attention' in name:
            name = name.replace('attention_block.', '')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Attention UNet model loaded.")
    return model

# Preprocessing
def preprocess_image(image, model_type='attenunet'):
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    image_tensor = augmented["image"].unsqueeze(0).to(device)
    return image_tensor

# Prediction
def predict_mask(model, image_tensor, model_type='attenunet'):
    with torch.no_grad():
        output = model(image_tensor)
        if model_type == 'deeplab':
            output = output['out']

        prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        threshold = calculate_adaptive_threshold(prob_mask)
        binary_mask = (prob_mask > threshold).astype(np.uint8)
        binary_mask = postprocess_mask(binary_mask)
        return binary_mask * 255, prob_mask

def calculate_adaptive_threshold(prob_mask):
    flat_probs = prob_mask.flatten()
    mean_prob = np.mean(flat_probs)

    if mean_prob < 0.1:
        threshold = np.percentile(flat_probs, 95)
    elif mean_prob > 0.9:
        threshold = 0.7
    else:
        threshold = 0.5

    print(f"Using adaptive threshold: {threshold:.2f}")
    return threshold

# Post-processing
def postprocess_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned

# Overlay mask
def overlay_mask(image, mask, alpha=0.3):
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [0, 0, 255]  # Red in BGR

    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    return overlay

# Visualization
def visualize_results(image, prob_mask, binary_mask):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(prob_mask, cmap='jet')
    plt.colorbar()
    plt.title("Probability Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Binary Mask")
    plt.axis('off')

    plt.tight_layout()
    return plt

# Model information
def get_model_info(model_name):
    info = {
        'fpn': {
            'name': 'Feature Pyramid Network',
            'encoder': 'ResNet50',
            'decoder': 'FPN with 256D features',
            'input_size': '256x256',
            'normalization': 'ImageNet stats',
            'training': {
                'loss': 'BCEWithLogitsLoss',
                'optimizer': 'Adam (1e-4)',
                'augmentation': 'Flip, rotate, contrast'
            }
        },
        'deeplab': {
            'name': 'DeepLabV3+',
            'backbone': 'ResNet50',
            'features': 'ASPP with atrous conv',
            'input_size': '256x256',
            'normalization': 'ImageNet stats',
            'training': {
                'loss': 'BCEWithLogitsLoss',
                'optimizer': 'Adam (1e-4)',
                'augmentation': 'Flip, contrast'
            }
        },
        'attenunet': {
            'name': 'Attention UNet',
            'encoder': 'ResNet34 with SCSE attention',
            'decoder': 'U-Net with skip connections',
            'input_size': '256x256',
            'normalization': 'ImageNet stats',
            'attention': 'Spatial and Channel SE blocks',
            'training': {
                'loss': 'BCEWithLogitsLoss',
                'optimizer': 'Adam (1e-4)',
                'augmentation': 'Basic transforms',
                'metrics': 'F1: 0.85, IoU: 0.75'
            }
        }
    }

    return info.get(model_name.lower(), "Unknown model")
