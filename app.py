import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import FPN, Unet
from torchvision.models.segmentation import deeplabv3_resnet50
import matplotlib.pyplot as plt
import os
import segmentation_models_pytorch as smp

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths (update these with your actual paths)
FPN_MODEL_PATH = "classifier_state_dict.pth"
DEEPLABV3_MODEL_PATH = "deeplabv3_model_weights.pth"
ATTENUNET_MODEL_PATH = "attunet_state_dict.pth"

# Model definitions
def load_fpn_model():
    model = FPN(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    state_dict = torch.load(FPN_MODEL_PATH, map_location=device)
    
    # Remove 'model.' prefix from state_dict keys if present
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def load_deeplabv3_model():
    # Create model with correct architecture
    model = deeplabv3_resnet50(pretrained=False, num_classes=1)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    
    # Load state dict
    state_dict = torch.load(DEEPLABV3_MODEL_PATH, map_location=device)
    
    # Remove 'model.' prefix from state_dict keys if present
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Load with strict=False to handle architecture differences
    model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device)
    model.eval()
    return model

def load_attenunet_model():
    # Create Attention UNet model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    
    # Load state dict
    state_dict = torch.load(ATTENUNET_MODEL_PATH, map_location=device)
    
    # Remove 'model.' prefix from state_dict keys if present
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

# Transformation
def get_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# Prediction function
def predict(image, model, transform):
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Apply transformations
    transformed = transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        if isinstance(model, FPN):
            output = model(image_tensor)
        elif isinstance(model, Unet):  # Attention UNet
            output = model(image_tensor)
        else:  # DeepLabV3
            output = model(image_tensor)['out']
        
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
    
    return pred_mask

# Visualization function
def visualize(image, mask):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')
    
    return fig

# Model information
def get_model_info(model_name):
    if model_name == "Attention UNet":
        return """
        ## Attention UNet Model Information
        
        **Architecture**: 
        - Encoder: ResNet34 with attention gates
        - Decoder: UNet-style with attention mechanisms
        - Output: Single channel segmentation mask
        
        **Training Details**:
        - Loss Function: BCEWithLogitsLoss
        - Optimizer: Adam (lr=1e-4)
        - Input Size: 256x256
        - Augmentations: Resize, normalization
        
        """
    elif model_name == "DeepLabV3+":
        return """
        ## DeepLabV3+ Model Information
        
        **Architecture**: 
        - Backbone: ResNet50
        - ASPP (Atrous Spatial Pyramid Pooling)
        - Output: Single channel segmentation mask
        
        **Training Details**:
        - Loss Function: BCEWithLogitsLoss
        - Optimizer: Adam (lr=1e-4)
        - Input Size: 256x256
        - Augmentations: Horizontal flip, brightness/contrast adjustment
        """
    else:  
        return """
        **Architecture**: 
        - Encoder: ResNet50
        - Decoder: Feature Pyramid Network
        - Output: Single channel segmentation mask
        
        **Training Details**:
        - Loss Function: BCEWithLogitsLoss
        - Optimizer: Adam (lr=1e-4)
        - Input Size: 256x256
        - Augmentations: Horizontal/Vertical flip, rotation, brightness/contrast adjustment
        """.format(0.85, 0.75, 0.95)  

# Streamlit app
def main():
    st.set_page_config(page_title="Crack Segmentation App", layout="wide")
    
    st.title("Crack Segmentation App")
    st.write("This app performs crack segmentation using FPN, DeepLabV3+ or Attention UNet models")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose what you want to do",
                               ["Model Information", "Segmentation"])
    
    if app_mode == "Model Information":
        st.header("Model Information")
        model_choice = st.radio("Select Model", ["Attention UNet", "DeepLabV3+", "FPN"])
        
        st.markdown(get_model_info(model_choice))
        
    elif app_mode == "Segmentation":
        st.header("Crack Segmentation")
        
        # Model selection
        model_choice = st.selectbox("Select Model", ["Attention UNet", "DeepLabV3+", "FPN"])
        
        # Load model
        with st.spinner(f"Loading {model_choice} model..."):
            if model_choice == "Attention UNet":
                model = load_fpn_model()
            elif model_choice == "DeepLabV3+":
                model = load_deeplabv3_model()
            else:  # Attention UNet
                model = load_attenunet_model()
        
        # Image upload
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read and display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Predict
            transform = get_transform()
            with st.spinner("Performing segmentation..."):
                pred_mask = predict(image, model, transform)
                
                # Display results
                fig = visualize(np.array(image), pred_mask)
                st.pyplot(fig)
                
                # Option to download the mask
                mask_img = Image.fromarray(pred_mask * 255)
                st.download_button(
                    label="Download Mask",
                    data=mask_img.tobytes(),
                    file_name="mask.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()


