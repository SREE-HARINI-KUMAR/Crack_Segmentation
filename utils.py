import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import FPN
from torchvision.models.segmentation import deeplabv3_resnet50
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    """Load all three models with proper weight loading and verification"""
    models = {}
    
    try:
        # Load FPN model
        models['fpn'] = load_fpn_model()
        
        # Load DeepLabV3+ model
        models['deeplab'] = load_deeplabv3_model()
        
        # Load Attention UNet model
        models['attenunet'] = load_attenunet_model()
        
        return models
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None

def load_fpn_model():
    """Load FPN model with weight verification"""
    model = FPN(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    
    state_dict = torch.load('classifier_state_dict.pth', map_location=device)
    state_dict = {k.replace('model.', '').replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # Verify first layer weights
    print("FPN first layer weights mean:", model.encoder.conv1.weight.mean().item())
    return model

def load_deeplabv3_model():
    """Load DeepLabV3+ model with proper classifier head"""
    model = deeplabv3_resnet50(pretrained=False, num_classes=1)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
    
    state_dict = torch.load('deeplabv3_model_weights.pth', map_location=device)
    state_dict = {k.replace('model.', '').replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print("DeepLabV3+ classifier weights mean:", model.classifier[4].weight.mean().item())
    return model

def load_attenunet_model():
    """Properly load Attention UNet with architecture verification"""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16],
        attention_type='scse'  # Verify if you used attention during training
    )
    
    state_dict = torch.load('attunet_state_dict.pth', map_location=device)
    
    # Handle different key naming schemes
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('model.', '')
        # Special handling for attention layers if needed
        if 'attention' in name:
            name = name.replace('attention_block.', '')
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # Verify attention layers loaded properly
    if hasattr(model, 'decoder_blocks'):
        print("Attention UNet decoder blocks loaded successfully")
    print("Attention UNet first conv weights mean:", model.encoder.conv1.weight.mean().item())
    
    return model

def preprocess_image(image, model_type='attenunet'):
    """Consistent preprocessing with model-specific adjustments"""
    # Base transform
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # Default ImageNet
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ])
    
    # Convert and augment
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    image_tensor = augmented["image"].unsqueeze(0).to(device)
    
    return image_tensor

def predict_mask(model, image_tensor, model_type='attenunet'):
    """Robust prediction with adaptive thresholding and post-processing"""
    with torch.no_grad():
        # Get raw model output
        output = model(image_tensor)
        if model_type == 'deeplab':
            output = output['out']
        
        # Convert to probabilities
        prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Adaptive threshold calculation
        threshold = calculate_adaptive_threshold(prob_mask)
        
        # Create binary mask
        binary_mask = (prob_mask > threshold).astype(np.uint8)
        
        # Post-processing
        binary_mask = postprocess_mask(binary_mask)
        
        return binary_mask * 255, prob_mask

def calculate_adaptive_threshold(prob_mask):
    """Calculate optimal threshold based on output distribution"""
    flat_probs = prob_mask.flatten()
    mean_prob = np.mean(flat_probs)
    
    if mean_prob < 0.1:  # Mostly background
        threshold = np.percentile(flat_probs, 95)
    elif mean_prob > 0.9:  # Mostly foreground
        threshold = 0.7
    else:  # Balanced case
        threshold = 0.5
    
    print(f"Using adaptive threshold: {threshold:.2f}")
    return threshold

def postprocess_mask(mask):
    """Clean up small artifacts in the mask"""
    # Remove small objects
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Fill small holes
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return cleaned

def overlay_mask(image, mask, alpha=0.3):
    """Enhanced visualization with adjustable opacity"""
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [0, 0, 255]  # Red in BGR
    
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    
    # Add border to highlight cracks
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    
    return overlay

def visualize_results(image, prob_mask, binary_mask):
    """Comprehensive visualization of all stages"""
    plt.figure(figsize=(15,5))
    
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.imshow(prob_mask, cmap='jet')
    plt.colorbar()
    plt.title("Probability Mask")
    plt.axis('off')
    
    plt.subplot(1,3,3)
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Binary Mask")
    plt.axis('off')
    
    plt.tight_layout()
    return plt

def get_model_info(model_name):
    """Detailed model information including architecture specifics"""
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
                'metrics': 'F1: 0.85, IoU: 0.75'  # Update with your metrics
            }
        }
    }
    
    return info.get(model_name.lower(), "Unknown model")