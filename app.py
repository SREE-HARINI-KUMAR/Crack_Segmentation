import streamlit as st
import torch
import torch.nn as nn
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
import requests

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
FPN_MODEL_PATH = "classifier_state_dict.pth"
DEEPLABV3_MODEL_PATH = "deeplabv3_model_weights.pth"
ATTENUNET_MODEL_PATH = "attunet_state_dict.pth"

# File download utility
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

# Load models
def load_fpn_model():
    file_id = "1gkUXdb4S2Dbm0mTotxFeDpaoO9emmd0X"
    if not os.path.exists(FPN_MODEL_PATH):
        with st.spinner("Downloading FPN model..."):
            download_file_from_google_drive(file_id, FPN_MODEL_PATH)

    model = FPN(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    state_dict = torch.load(FPN_MODEL_PATH, map_location=device)
    new_state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def load_deeplabv3_model():
    file_id = "1ZfWGp384hjY3T1HgxajNDRTdIvfVjVym"
    if not os.path.exists(DEEPLABV3_MODEL_PATH):
        with st.spinner("Downloading DeepLabV3+ model..."):
            download_file_from_google_drive(file_id, DEEPLABV3_MODEL_PATH)

    model = deeplabv3_resnet50(pretrained=False, num_classes=1)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    state_dict = torch.load(DEEPLABV3_MODEL_PATH, map_location=device)
    new_state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def load_attenunet_model():
    file_id = "15CQVZymXH4qFQ6rnxJEQ0k7wkrNYeflg"
    if not os.path.exists(ATTENUNET_MODEL_PATH):
        with st.spinner("Downloading Attention UNet model..."):
            download_file_from_google_drive(file_id, ATTENUNET_MODEL_PATH)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    state_dict = torch.load(ATTENUNET_MODEL_PATH, map_location=device)
    new_state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()}
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

# Prediction
def predict(image, model, transform):
    image_np = np.array(image)
    transformed = transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        if isinstance(model, FPN) or isinstance(model, Unet):
            output = model(image_tensor)
        else:
            output = model(image_tensor)['out']

        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

    return pred_mask

# Visualization
def visualize(image, mask):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')

    return fig

# Model info
def get_model_info(model_name):
    if model_name == "Attention UNet":
        return """
        ### Attention UNet Model

        - **Encoder**: ResNet34 with attention gates  
        - **Decoder**: U-Net style with skip connections  
        - **Input Size**: 256x256  
        - **Normalization**: ImageNet stats  

        **Training Details**:  
        - Loss Function: BCEWithLogitsLoss  
        - Optimizer: Adam (lr=1e-4)  
        - Augmentations: Resize, normalization  
        """
    elif model_name == "DeepLabV3+":
        return """
        ### DeepLabV3+ Model

        - **Backbone**: ResNet50  
        - **Features**: ASPP (Atrous Spatial Pyramid Pooling)  
        - **Input Size**: 256x256  
        - **Normalization**: ImageNet stats  

        **Training Details**:  
        - Loss Function: BCEWithLogitsLoss  
        - Optimizer: Adam (lr=1e-4)  
        - Augmentations: Horizontal flip, brightness/contrast  
        """
    else:
        return """
        ### FPN Model

        - **Encoder**: ResNet50  
        - **Decoder**: Feature Pyramid Network  
        - **Input Size**: 256x256  
        - **Normalization**: ImageNet stats  

        **Training Details**:  
        - Loss Function: BCEWithLogitsLoss  
        - Optimizer: Adam (lr=1e-4)  
        - Augmentations: Flip, rotate, contrast  
        """

# Streamlit App
def main():
    st.set_page_config(page_title="Crack Segmentation App", layout="wide")
    st.title("Crack Segmentation App")
    st.write("This app performs crack segmentation using FPN, DeepLabV3+ or Attention UNet models")

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose what you want to do", ["Model Information", "Segmentation"])

    if app_mode == "Model Information":
        st.header("Model Information")
        model_choice = st.radio("Select Model", ["FPN", "DeepLabV3+", "Attention UNet"])
        st.markdown(get_model_info(model_choice))

    elif app_mode == "Segmentation":
        st.header("Crack Segmentation")
        model_choice = st.selectbox("Select Model", ["FPN", "DeepLabV3+", "Attention UNet"])

        with st.spinner(f"Loading {model_choice} model..."):
            if model_choice == "FPN":
                model = load_fpn_model()
            elif model_choice == "DeepLabV3+":
                model = load_deeplabv3_model()
            else:
                model = load_attenunet_model()

        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            transform = get_transform()
            with st.spinner("Performing segmentation..."):
                pred_mask = predict(image, model, transform)

                fig = visualize(np.array(image), pred_mask)
                st.pyplot(fig)

                # Download option
                mask_img = Image.fromarray(pred_mask * 255)
                st.download_button(
                    label="Download Mask",
                    data=mask_img.tobytes(),
                    file_name="predicted_mask.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
