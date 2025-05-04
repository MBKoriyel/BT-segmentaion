import streamlit as st
import numpy as np
import nibabel as nib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import gdown  # For downloading files from Google Drive
import zipfile  # To handle folder uploads
import tempfile  # To handle temporary files
from tensorflow.keras.utils import to_categorical

# Title of the app
st.title("Brain Tumor Segmentation using 3D U-Net - (Lightweight Architecture on Normal CPUs)")
# ...
