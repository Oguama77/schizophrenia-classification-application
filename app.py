import streamlit as st
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import tempfile
import zipfile
import gzip
import shutil
from collections import Counter

# Load the trained model
@st.cache_resource()
def load_model(model_path):
    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for single-channel input
    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to decompress .nii.gz files
def decompress_nii_gz(nii_gz_path):
    nii_path = nii_gz_path.replace(".gz", "")
    with gzip.open(nii_gz_path, "rb") as f_in:
        with open(nii_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return nii_path

# Function to extract NIfTI file from ZIP
def extract_nii_from_zip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    
    nii_files = [f for f in os.listdir(extract_dir) if f.endswith((".nii", ".nii.gz"))]
    if not nii_files:
        return None
    
    return os.path.join(extract_dir, nii_files[0])

# Function to validate NIfTI file
def validate_nifti(file_path):
    try:
        nib.load(file_path)
        return True
    except nib.filebasedimages.ImageFileError:
        return False

# Preprocess NIfTI file
def preprocess_nii(file_path):
    nii = nib.load(file_path)
    data = nii.get_fdata()
    
    # Normalize the data
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    slices = []
    for i in range(data.shape[2]):  # Iterate through all slices along the z-axis
        slice_data = (data[:, :, i] * 255).astype(np.uint8)
        slice_data = transforms.ToPILImage()(slice_data)
        slice_tensor = transform(slice_data).unsqueeze(0)
        slices.append(slice_tensor)
    
    return slices

# Make predictions for all slices and return the modal value
def predict(model, input_tensors):
    predictions = []
    with torch.no_grad():
        for tensor in input_tensors:
            output = model(tensor)
            prob = torch.softmax(output, dim=1)[0][1].item()
            prediction = "Schizophrenic" if prob >= 0.5 else "Control"
            predictions.append(prediction)
    
    # Get the most frequent prediction
    modal_prediction = Counter(predictions).most_common(1)[0][0]
    return modal_prediction

# Streamlit app
def main():
    # Streamlit app setup
    # define logo
    logo = "logo.png"
    st.logo(logo, size="large")
    st.set_page_config(
        page_title="Schiz-Classification-App",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    
    # set default text area size for Streamlit with CSS
    st.markdown(
        """
        <style>
        /* Adjust the height of the text area and the image display */
        div[data-testid="stTextArea"] textarea {
            height: 50px;
            min-height: 50px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Define image
    mri = "mri.jpg"
      
    st.markdown(
    "<h1 style='text-align: center;'>Schizophrenia Detection from 3D MRI Scans</h1>", 
    unsafe_allow_html=True)

# Display the centered image
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image(mri, width=300)
    
    st.write("")
    st.write("")
    st.write("Upload a 3D MRI scan in NIfTI format (.nii, .nii.gz, or .zip) to check for schizophrenia.")
    
    uploaded_file = st.file_uploader("", 
                                    type=["nii", "nii.gz", "zip"])
    model_path = "resnet50_schizophrenia4.pth"
    model = load_model(model_path)

    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            if file_path.endswith(".zip"):
                st.write("Extracting ZIP file...")
                file_path = extract_nii_from_zip(file_path, temp_dir)
                if not file_path:
                    st.error("No valid NIfTI file found in the ZIP archive.")
                    st.stop()

            if file_path.endswith(".nii.gz"):
                st.write("Decompressing .nii.gz file...")
                file_path = decompress_nii_gz(file_path)

            if not validate_nifti(file_path):
                st.error("The uploaded file is not a valid NIfTI file. Please check your input.")
                st.stop()

            st.write("Preprocessing the scan...")
            input_tensors = preprocess_nii(file_path)

            st.write("Making predictions on all slices...")
            modal_prediction = predict(model, input_tensors)

            st.write(f"**Diagnosis:** {modal_prediction}")

if __name__ == "__main__":
    main()
