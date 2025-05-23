# Getting Started with the Bird Species Classification Project

This guide explains how to obtain the necessary dataset and set up your environment for the Bird Species Classification case study.

## Required Python Packages

First, install the required Python libraries:

```python
pip install numpy pandas matplotlib seaborn tensorflow opencv-python pillow scikit-learn flask
```

For TensorFlow model optimization:

```python
pip install tensorflow-model-optimization
```

## Dataset Information

This project uses the Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset, which contains 11,788 images across 200 bird species.

### Option 1: Download the CUB-200-2011 Dataset Directly

```python
import os
import urllib.request
import tarfile
from pathlib import Path

# Create directories
os.makedirs('birds_dataset', exist_ok=True)

# Download the dataset
print("Downloading CUB-200-2011 dataset (about 1.1GB)...")
url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"

try:
    # Download the tarball
    urllib.request.urlretrieve(url, "CUB_200_2011.tgz")
    print("Download complete. Extracting files...")
    
    # Extract the tarball
    with tarfile.open("CUB_200_2011.tgz", "r:gz") as tar:
        tar.extractall()
    
    # Rename and organize directories for the project structure
    os.makedirs('birds_dataset/images', exist_ok=True)
    os.makedirs('birds_dataset/annotations', exist_ok=True)
    
    # Move image directories
    os.rename("CUB_200_2011/images", "birds_dataset/images")
    
    # Copy annotations
    os.rename("CUB_200_2011/parts", "birds_dataset/annotations/parts")
    os.rename("CUB_200_2011/bounding_boxes.txt", "birds_dataset/annotations/bounding_boxes.txt")
    
    # Copy class labels
    os.rename("CUB_200_2011/classes.txt", "birds_dataset/labels.txt")
    
    # Clean up
    os.remove("CUB_200_2011.tgz")
    print("Dataset extraction and organization complete!")
    
except Exception as e:
    print(f"Error downloading or extracting dataset: {e}")
    print("Please try manual download instead.")
```

### Option 2: Manual Download

1. Visit the Caltech-UCSD Birds dataset page: https://www.vision.caltech.edu/datasets/cub_200_2011/
2. Download the CUB_200_2011.tgz file (approximately 1.1GB)
3. Extract the archive using a program like 7-Zip, WinRAR, or the command line
4. Organize the files into the following structure:
   ```
   birds_dataset/
   ├── images/
   │   ├── 001.Black_footed_Albatross/
   │   ├── 002.Laysan_Albatross/
   │   └── ... (remaining species folders)
   ├── annotations/
   │   ├── parts/
   │   └── bounding_boxes.txt
   └── labels.txt
   ```

## Project Structure Setup

Create the necessary directory structure for the project:

```python
import os

# Create project directories
directories = [
    'birds_dataset',
    'models',
    'visualizations',
    'data/processed'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")
```

## About the CUB-200-2011 Dataset

The Caltech-UCSD Birds-200-2011 dataset contains:

- 11,788 images of birds
- 200 bird species categories
- Annotations for bounding boxes and part locations
- Each species has between 40-60 images
- Images were collected from Flickr and include photos of birds in their natural habitat

This dataset is widely used in fine-grained visual categorization research and is perfect for bird species classification tasks.

## Verifying the Setup

Run this code to check if your dataset is properly downloaded and organized:

```python
import os
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np

# Check the dataset structure
def verify_dataset(base_path="birds_dataset"):
    try:
        # Check if main directories exist
        images_path = os.path.join(base_path, "images")
        labels_file = os.path.join(base_path, "labels.txt")
        
        if not os.path.exists(images_path):
            print(f"ERROR: Images directory not found at {images_path}")
            return False
        
        if not os.path.exists(labels_file):
            print(f"ERROR: Labels file not found at {labels_file}")
            return False
        
        # Count species directories
        species_dirs = [d for d in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, d))]
        print(f"Found {len(species_dirs)} species directories")
        
        # Count total images
        total_images = 0
        for species_dir in species_dirs:
            species_path = os.path.join(images_path, species_dir)
            images = [f for f in os.listdir(species_path) if f.endswith('.jpg')]
            total_images += len(images)
        
        print(f"Found {total_images} total images")
        
        # Read class labels
        with open(labels_file, 'r') as f:
            class_names = [line.strip().split('.')[1].replace('_', ' ') for line in f.readlines()]
        
        print(f"Found {len(class_names)} class names")
        
        # Display random images from 3 random classes
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        sample_dirs = random.sample(species_dirs, 3)
        
        for i, species_dir in enumerate(sample_dirs):
            species_path = os.path.join(images_path, species_dir)
            images = [f for f in os.listdir(species_path) if f.endswith('.jpg')]
            
            if images:
                for j in range(2):
                    img_file = random.choice(images)
                    img_path = os.path.join(species_path, img_file)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    axes[i, j].imshow(img)
                    axes[i, j].set_title(species_dir.split('.')[1].replace('_', ' '))
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("Dataset verification complete!")
        return True
    
    except Exception as e:
        print(f"Error during verification: {e}")
        return False

# Run verification
verify_dataset()
```

## Expected Output Files

When running the complete code, the following files will be generated:

### Models
- `bird_classification_model.h5`: The trained machine learning model
- `bird_classification_model.tflite`: TensorFlow Lite version of the model
- `bird_classification_model_quantized.tflite`: Size-optimized quantized model

### Visualizations
- Various plots showing dataset distribution
- Sample bird images with augmentation
- Training and validation accuracy/loss curves
- Confusion matrix for model evaluation
- Class activation maps highlighting important image regions
- Prediction examples

## Mobile Applications

The case study includes pseudocode for developing mobile applications:
- Android implementation using Kotlin and TensorFlow Lite
- iOS implementation using Swift and TensorFlow Lite

## Additional Resources

If you need alternative datasets for bird species classification:

1. **NABirds**: A comprehensive dataset with 555 species of North American birds (48,000 images)
   - Available at: https://dl.allaboutbirds.org/nabirds

2. **iNaturalist Birds**: Part of the broader iNaturalist dataset with bird observations
   - Available at: https://github.com/visipedia/inat_comp

3. **Birds-525**: A dataset with 525 bird species (87,000 images)
   - Available at various computer vision repositories

## Using the Bird Classification API

The case study includes a Flask API for serving bird classifications. After training your model, you can start the API:

```python
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the model
interpreter = tf.lite.Interpreter(model_path='bird_classification_model_quantized.tflite')
interpreter.allocate_tensors()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # API implementation is included in the case study code
    # ...

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

## Ethical Considerations

The case study emphasizes important ethical considerations when working with wildlife AI:
- Privacy and location data handling for rare and endangered species
- Conservation messaging and educational content
- Responsible wildlife observation practices
- Data collection ethics

By following this guide and the code in the case study, you'll build a complete bird species classification system with considerations for deployment on mobile devices and ethical wildlife applications.