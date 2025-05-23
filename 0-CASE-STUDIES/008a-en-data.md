# Getting Started with the Medical Image Classification Project

This guide explains how to obtain the necessary dataset and set up your environment for the Medical Image Classification for Disease Detection case study.

## Required Python Packages

First, install the required Python libraries:

```python
pip install numpy pandas matplotlib seaborn tensorflow opencv-python pillow scikit-learn flask
```

For GPU acceleration (recommended for this large dataset):

```python
pip install tensorflow-gpu
```

## Dataset Information

This project uses the NIH Chest X-ray Dataset, which contains over 100,000 chest X-ray images with disease labels.

### Option 1: Download from NIH

The NIH Chest X-ray Dataset can be downloaded directly from the National Institutes of Health:

```python
import os
import urllib.request
import tarfile

# Create directories
os.makedirs('NIH_Chest_X_rays', exist_ok=True)
os.makedirs('NIH_Chest_X_rays/images', exist_ok=True)

# Note: This dataset is large (>40GB) and may take significant time to download
print("Starting download of NIH Chest X-ray Dataset (this may take several hours)...")

# Download images (large file - approximately 42GB)
images_url = "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345"
print(f"Please download the dataset manually from: {images_url}")
print("Download all image files and extract them to the NIH_Chest_X_rays/images directory")

# Download data entry CSV (much smaller)
csv_url = "https://nihcc.app.box.com/v/ChestXray-NIHCC/file/220660789869"
print(f"Please download the data entry CSV from: {csv_url}")
print("Save this file as NIH_Chest_X_rays/Data_Entry_2017.csv")
```

### Option 2: Download from Kaggle

The dataset is also available on Kaggle, which may provide faster downloads:

1. Visit [NIH Chest X-rays on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
2. Download the dataset (requires Kaggle account)
3. Extract files to the `NIH_Chest_X_rays` directory
4. Ensure images are in `NIH_Chest_X_rays/images` and the CSV file is named `Data_Entry_2017.csv`

### Option 3: Use a Small Subset for Testing

If you want to test the code without downloading the full dataset:

```python
import os
import urllib.request
import zipfile

# Create directories
os.makedirs('NIH_Chest_X_rays/images', exist_ok=True)

# Download a small subset (100 images) for testing
# Note: This is not the official subset, but created for demonstration purposes
sample_url = "https://github.com/ieee8023/covid-chestxray-dataset/archive/refs/heads/master.zip"

try:
    print("Downloading sample chest X-ray images...")
    urllib.request.urlretrieve(sample_url, "sample_xray.zip")
    
    # Extract sample images
    with zipfile.ZipFile("sample_xray.zip", "r") as zip_ref:
        zip_ref.extractall("./")
    
    # Create a simplified metadata file
    import pandas as pd
    import glob
    
    # Find all images
    image_files = glob.glob("covid-chestxray-dataset-master/images/*.png") + \
                  glob.glob("covid-chestxray-dataset-master/images/*.jpg")
    
    # Create sample metadata
    data = []
    for img_path in image_files:
        filename = os.path.basename(img_path)
        # Copy image to our target directory
        from shutil import copy
        copy(img_path, f"NIH_Chest_X_rays/images/{filename}")
        # Add metadata entry
        findings = "No Finding" if "normal" in img_path.lower() else "Pneumonia"
        data.append({
            "Image Index": filename,
            "Finding Labels": findings,
            "Patient Age": 45,  # Default value
            "Patient Gender": "M"  # Default value
        })
    
    # Create and save the CSV
    df = pd.DataFrame(data)
    df.to_csv("NIH_Chest_X_rays/Data_Entry_2017.csv", index=False)
    
    print(f"Sample dataset created with {len(data)} images")
    
except Exception as e:
    print(f"Error downloading sample dataset: {e}")
    print("Please try one of the other download options.")
```

## Project Structure Setup

Create the necessary directory structure for the project:

```python
import os

# Create project directories
directories = [
    'NIH_Chest_X_rays',
    'NIH_Chest_X_rays/images',
    'models',
    'visualizations'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")
```

## About the NIH Chest X-ray Dataset

The NIH Chest X-ray Dataset contains:

- 112,120 X-ray images from 30,805 unique patients
- 14 disease labels (multi-label, as one image may have multiple conditions)
- Labels were created using NLP mining of radiology reports
- Conditions include: Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural thickening, Cardiomegaly, Nodule, Mass, and Hernia

### Data Structure

- `images/`: Directory containing all X-ray images in PNG format
- `Data_Entry_2017.csv`: Metadata file containing:
  - Image filenames
  - Disease labels (multiple labels separated by '|')
  - Patient age
  - Patient gender
  - Other metadata

## Verifying the Setup

Run this code to check if your dataset is properly downloaded and organized:

```python
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

try:
    # Check if data directory exists
    data_dir = "NIH_Chest_X_rays"
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory {data_dir} not found")
    
    # Check if metadata file exists
    metadata_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    if not os.path.exists(metadata_path):
        print(f"ERROR: Metadata file {metadata_path} not found")
    else:
        # Load and check metadata
        df = pd.read_csv(metadata_path)
        print(f"Metadata loaded successfully with {len(df)} records")
        print("\nMetadata first 5 rows:")
        print(df.head())
        
        # Check if image directory exists
        image_dir = os.path.join(data_dir, "images")
        if not os.path.exists(image_dir):
            print(f"ERROR: Image directory {image_dir} not found")
        else:
            # Check a few images
            if 'Image Index' in df.columns:
                sample_images = df['Image Index'].iloc[:5].values
                
                # Try to load and display sample images
                plt.figure(figsize=(15, 10))
                for i, img_name in enumerate(sample_images):
                    img_path = os.path.join(image_dir, img_name)
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            plt.subplot(1, 5, i+1)
                            plt.imshow(img, cmap='gray')
                            plt.title(f"Image {i+1}")
                            plt.axis('off')
                        else:
                            print(f"WARNING: Could not load image {img_path}")
                    else:
                        print(f"WARNING: Image file {img_path} not found")
                
                plt.tight_layout()
                plt.show()
                print("Setup verification complete!")
            else:
                print("ERROR: 'Image Index' column not found in metadata")
    
except Exception as e:
    print(f"Setup verification failed: {e}")
```

## Expected Output Files

When running the complete code, the following files will be generated:

### Models
- `best_model.h5`: The trained deep learning model
- `chest_xray_model.tflite`: TensorFlow Lite model for mobile deployment
- `chest_xray_model_quantized.tflite`: Size-optimized TensorFlow Lite model

### Visualizations
- Distribution of conditions in the dataset
- Sample X-ray images
- Augmented training images
- Training and validation curves
- ROC curves for each disease
- Precision-recall curves
- Performance metrics by disease
- Grad-CAM visualizations showing model attention regions

## Using Alternative Datasets

If you can't access the full NIH dataset, consider these alternatives:

1. **COVID-19 Radiography Database**: 
   - https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
   - Smaller dataset (~3GB) with COVID-19, viral pneumonia, and normal X-rays

2. **ChestX-ray8**: 
   - A smaller subset of the full NIH dataset
   - https://www.kaggle.com/datasets/nih-chest-xrays/sample

3. **RSNA Pneumonia Detection Challenge**:
   - https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
   - ~30GB dataset focused on pneumonia detection

## Important Notes for Medical AI Projects

1. **Computational Requirements**:
   - This project uses deep learning with transfer learning on medical images
   - Recommended: 16GB+ RAM, GPU with 8GB+ VRAM
   - Without GPU, training will be extremely slow

2. **Medical Expertise**:
   - The model outputs should be interpreted as assistance tools, not final diagnoses
   - Always consult medical professionals for actual diagnosis
   - This is an educational project, not a clinical deployment

3. **Ethical Considerations**:
   - Patient privacy is paramount - ensure all data is properly anonymized
   - Model performance varies across demographics - test for biases
   - Clear documentation of model limitations is essential

With this setup, you'll have everything needed to build an end-to-end medical image classification system for chest X-rays, including data handling, model development, evaluation, and deployment considerations.