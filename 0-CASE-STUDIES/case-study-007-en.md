# Case Study: End-to-End ML Project - Bird Species Classification

## 1. Problem Definition and Project Overview

Computer vision is transforming how we interact with and understand the natural world. In this case study, we'll build an image classification system to identify bird species from photographs - a task with applications in ecological monitoring, citizen science, and wildlife conservation.

```python
"""
Project: BirdWatch - Bird Species Classification System

Goal: Develop a deep learning model that can accurately identify bird species from images.

Applications:
- Mobile app for birdwatchers to identify species in the field
- Ecological research for automated bird counting and monitoring
- Education and citizen science initiatives

Success Metrics:
- Classification accuracy above 85% on test set
- Model capable of identifying at least 200 common bird species
- Inference time under 500ms per image on mobile device
"""
```

## 2. Dataset and Exploratory Analysis

We'll use the Caltech-UCSD Birds-200-2011 dataset, which contains 11,788 images across 200 bird species.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from PIL import Image
from pathlib import Path
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configure matplotlib
plt.style.use('fivethirtyeight')
sns.set_palette('viridis')

# Define paths to dataset
data_path = "birds_dataset"
images_path = os.path.join(data_path, "images")
annotations_path = os.path.join(data_path, "annotations")
labels_file = os.path.join(data_path, "labels.txt")

# Load class labels
with open(labels_file, 'r') as f:
    class_names = [line.strip().split('.')[1].replace('_', ' ') for line in f.readlines()]

# Get image paths and their labels
image_files = []
labels = []

for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(images_path, f"{class_idx+1:03d}")
    for img_file in os.listdir(class_dir):
        if img_file.endswith('.jpg'):
            image_files.append(os.path.join(class_dir, img_file))
            labels.append(class_idx)

# Convert to numpy arrays            
image_files = np.array(image_files)
labels = np.array(labels)

print(f"Total number of images: {len(image_files)}")
print(f"Number of classes: {len(class_names)}")
```

### Dataset Exploration

```python
# Check class distribution
plt.figure(figsize=(15, 6))
plt.hist(labels, bins=len(class_names))
plt.title('Distribution of Bird Species in Dataset')
plt.xlabel('Species Index')
plt.ylabel('Number of Images')
plt.tight_layout()
plt.show()

# View some sample images
plt.figure(figsize=(20, 16))
for i in range(15):
    # Select random image
    random_idx = np.random.randint(0, len(image_files))
    img_path = image_files[random_idx]
    label = labels[random_idx]
    
    # Load and display image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.subplot(3, 5, i+1)
    plt.imshow(img)
    plt.title(f"{class_names[label]}")
    plt.axis('off')
    
plt.tight_layout()
plt.show()

# Analyze image dimensions
img_heights = []
img_widths = []

for i in range(min(500, len(image_files))):  # Sample 500 images
    img = Image.open(image_files[i])
    width, height = img.size
    img_heights.append(height)
    img_widths.append(width)
    
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(img_heights)
plt.title('Image Heights')
plt.xlabel('Height (pixels)')
plt.subplot(1, 2, 2)
sns.histplot(img_widths)
plt.title('Image Widths')
plt.xlabel('Width (pixels)')
plt.tight_layout()
plt.show()

print(f"Average image dimensions: {np.mean(img_widths):.1f} x {np.mean(img_heights):.1f} pixels")
```

## 3. Data Preprocessing for Computer Vision

```python
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Split the data into training, validation, and testing sets
train_files, test_files, train_labels, test_labels = train_test_split(
    image_files, labels, test_size=0.2, stratify=labels, random_state=42
)

train_files, val_files, train_labels, val_labels = train_test_split(
    train_files, train_labels, test_size=0.2, stratify=train_labels, random_state=42
)

print(f"Training set: {len(train_files)} images")
print(f"Validation set: {len(val_files)} images")
print(f"Testing set: {len(test_files)} images")

# Define image dimensions
img_height = 224
img_width = 224
batch_size = 32

# Image preprocessing function
def preprocess_image(file_path):
    # Read image from file path
    img = tf.io.read_file(file_path)
    # Decode JPEG
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize
    img = tf.image.resize(img, [img_height, img_width])
    # Normalize pixel values
    img = img / 255.0
    return img

# Create TensorFlow datasets
def create_dataset(file_paths, labels):
    file_paths_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    images_ds = file_paths_ds.map(
        preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((images_ds, labels_ds))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# Create datasets for training, validation, and testing
train_ds = create_dataset(train_files, train_labels)
val_ds = create_dataset(val_files, val_labels)
test_ds = create_dataset(test_files, test_labels)

# Data augmentation for the training set
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

# Visualize some augmented images
plt.figure(figsize=(16, 8))
for images, _ in train_ds.take(1):
    for i in range(8):
        augmented_images = data_augmentation(images[:8])
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(augmented_images[i])
        plt.axis("off")
        
plt.tight_layout()
plt.show()
```

## 4. Building a Convolutional Neural Network

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Function to create a custom CNN model
def create_cnn_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to create a transfer learning model
def create_transfer_learning_model(num_classes):
    # Use MobileNetV2 as the base model
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3),
                             include_top=False,
                             weights='imagenet')
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create a transfer learning model
model = create_transfer_learning_model(len(class_names))

# Print the model summary
model.summary()
```

## 5. Model Training and Evaluation

```python
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath='bird_classification_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Train the model
start_time = time.time()

history = model.fit(
    train_ds,
    epochs=30,
    validation_data=val_ds,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Plot training history
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")
```

## 6. Fine-Tuning the Transfer Learning Model

```python
# Load the best model from checkpoint
best_model = tf.keras.models.load_model('bird_classification_model.h5')

# Unfreeze some layers of the base model for fine-tuning
base_model = best_model.layers[0]
base_model.trainable = True

# Freeze all the layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile the model
best_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune the model
start_time = time.time()

fine_tune_history = best_model.fit(
    train_ds,
    epochs=15,
    validation_data=val_ds,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

fine_tuning_time = time.time() - start_time
print(f"Fine-tuning completed in {fine_tuning_time:.2f} seconds")

# Plot fine-tuning history
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(fine_tune_history.history['accuracy'], label='Training Accuracy')
plt.plot(fine_tune_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Fine-Tuning Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(fine_tune_history.history['loss'], label='Training Loss')
plt.plot(fine_tune_history.history['val_loss'], label='Validation Loss')
plt.title('Fine-Tuning Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the fine-tuned model
final_model = tf.keras.models.load_model('bird_classification_model.h5')
final_test_loss, final_test_accuracy = final_model.evaluate(test_ds)
print(f"Final test accuracy: {final_test_accuracy:.4f}")
print(f"Final test loss: {final_test_loss:.4f}")
```

## 7. Model Analysis and Visualization

```python
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, classification_report

# Get predictions on the test set
all_images = []
all_labels = []

for images, labels in test_ds:
    all_images.append(images.numpy())
    all_labels.append(labels.numpy())

test_images = np.concatenate(all_images, axis=0)
test_labels = np.concatenate(all_labels, axis=0)
predictions = final_model.predict(test_ds)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate confusion matrix for a subset of classes (e.g., the first 15)
num_classes_to_visualize = 15
cm = confusion_matrix(
    test_labels[:100], 
    predicted_labels[:100], 
    labels=range(num_classes_to_visualize)
)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (First 15 Classes)')
plt.colorbar()

class_names_subset = class_names[:num_classes_to_visualize]
tick_marks = np.arange(len(class_names_subset))
plt.xticks(tick_marks, class_names_subset, rotation=90, fontsize=8)
plt.yticks(tick_marks, class_names_subset, fontsize=8)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Generate a classification report
print("Classification Report:")
print(classification_report(
    test_labels, 
    predicted_labels, 
    target_names=class_names,
    labels=range(len(class_names))
))

# Visualize predictions on some test images
plt.figure(figsize=(20, 16))
for i in range(15):
    # Select a random image from the test set
    idx = random.randint(0, len(test_images) - 1)
    img = test_images[idx]
    true_label = test_labels[idx]
    pred_label = predicted_labels[idx]
    
    plt.subplot(3, 5, i+1)
    plt.imshow(img)
    color = 'green' if true_label == pred_label else 'red'
    title = f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}"
    plt.title(title, color=color)
    plt.axis('off')
    
plt.tight_layout()
plt.show()

# Visualize class activation maps for feature visualization
from tensorflow.keras.models import Model
import cv2

# Function to get a class activation map
def get_class_activation_map(img, model, class_idx):
    # Get the gradient model
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer('global_average_pooling2d').output, model.output]
    )
    
    # Expand dimensions for batch
    img_array = np.expand_dims(img, axis=0)
    
    # Get gradients and outputs
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]
    
    # Extract gradients
    grads = tape.gradient(loss, conv_outputs)[0]
    
    # Get weights
    weights = tf.reduce_mean(grads, axis=(0, 1))
    
    # Create cam
    cam = tf.reduce_sum(
        tf.multiply(conv_outputs[0], weights), axis=-1
    ).numpy()
    
    # Process CAM
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = cv2.resize(cam, (img_height, img_width))
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap on original image
    img = (img * 255).astype(np.uint8)
    superimposed_img = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
    
    return cam, superimposed_img

# Visualize activation maps for some correctly classified images
plt.figure(figsize=(20, 15))
correct_idxs = np.where(test_labels == predicted_labels)[0]

for i in range(5):
    idx = correct_idxs[i]
    img = test_images[idx]
    true_label = test_labels[idx]
    pred_label = predicted_labels[idx]
    
    # Get the class activation map
    cam, superimposed = get_class_activation_map(img, final_model, pred_label)
    
    # Plot the original image
    plt.subplot(5, 3, i*3+1)
    plt.imshow(img)
    plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}")
    plt.axis('off')
    
    # Plot the class activation map
    plt.subplot(5, 3, i*3+2)
    plt.imshow(cam, cmap='jet')
    plt.title("Activation Map")
    plt.axis('off')
    
    # Plot the superimposed image
    plt.subplot(5, 3, i*3+3)
    plt.imshow(superimposed)
    plt.title("Superimposed")
    plt.axis('off')
    
plt.tight_layout()
plt.show()
```

## 8. Model Optimization for Mobile Deployment

```python
import tensorflow as tf
import os
import time
import tensorflow_model_optimization as tfmot
from tensorflow.keras.preprocessing import image

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
tflite_model = converter.convert()

# Save the TFLite model
with open('bird_classification_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model size:", os.path.getsize('bird_classification_model.tflite') / (1024 * 1024), "MB")

# Quantize the model for further size reduction
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()

# Save the quantized TFLite model
with open('bird_classification_model_quantized.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

print("Quantized TFLite model size:", os.path.getsize('bird_classification_model_quantized.tflite') / (1024 * 1024), "MB")

# Function to test inference speed
def test_inference_speed(model_path, num_runs=50):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create a random input image
    input_shape = input_details[0]['shape']
    input_data = np.random.random(input_shape).astype(np.float32)
    
    # Warm up
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Measure inference time
    start_time = time.time()
    for _ in range(num_runs):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # Convert to milliseconds
    return avg_time

# Test inference speed
regular_model_time = test_inference_speed('bird_classification_model.tflite')
quantized_model_time = test_inference_speed('bird_classification_model_quantized.tflite')

print(f"Regular TFLite model average inference time: {regular_model_time:.2f} ms")
print(f"Quantized TFLite model average inference time: {quantized_model_time:.2f} ms")

# Compare the accuracy of the quantized model
# Define a function to evaluate TFLite model accuracy
def evaluate_tflite_model(tflite_model_path, dataset):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    correct = 0
    total = 0
    
    for images, labels in dataset:
        for i in range(len(images)):
            img = images[i:i+1].numpy()
            label = labels[i].numpy()
            
            # Set the input tensor
            interpreter.set_tensor(input_details[0]['index'], img)
            # Run inference
            interpreter.invoke()
            # Get the output tensor
            output = interpreter.get_tensor(output_details[0]['index'])
            
            # Get the predicted class
            predicted_label = np.argmax(output[0])
            
            if predicted_label == label:
                correct += 1
            total += 1
    
    return correct / total

# Evaluate a subset of the test dataset
test_ds_subset = test_ds.take(20)
quantized_accuracy = evaluate_tflite_model('bird_classification_model_quantized.tflite', test_ds_subset)

print(f"Quantized model accuracy on subset: {quantized_accuracy:.4f}")
```

## 9. Building a Mobile Application Prototype

```python
# Pseudocode for a mobile application using the TFLite model

"""
Mobile Application Architecture:

1. User Interface:
   - Camera view for taking pictures
   - Gallery access for selecting existing photos
   - Results display showing top-3 predictions with confidence scores
   - Bird information page with details about the identified species

2. Backend:
   - TFLite model integration
   - Image preprocessing pipeline
   - Species database with information about birds

3. Implementation Flow:
   a. Capture or select image
   b. Preprocess image (resize to 224x224, normalize)
   c. Run inference with TFLite model
   d. Display results and provide additional information

----------------------------------------------
# Android implementation snippet (Kotlin)

// Load the TFLite model
val modelPath = "bird_classification_model_quantized.tflite"
val interpreter = Interpreter(FileUtil.loadMappedFile(context, modelPath))

// Preprocess image
fun preprocessImage(bitmap: Bitmap): ByteBuffer {
    val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
    val byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3)
    byteBuffer.order(ByteOrder.nativeOrder())
    
    val intValues = IntArray(224 * 224)
    resizedBitmap.getPixels(intValues, 0, 224, 0, 0, 224, 224)
    
    for (pixelValue in intValues) {
        byteBuffer.putFloat(((pixelValue shr 16 and 0xFF) / 255.0f))
        byteBuffer.putFloat(((pixelValue shr 8 and 0xFF) / 255.0f))
        byteBuffer.putFloat(((pixelValue and 0xFF) / 255.0f))
    }
    
    return byteBuffer
}

// Run inference
fun classifyImage(bitmap: Bitmap): List<Prediction> {
    val input = preprocessImage(bitmap)
    val output = Array(1) { FloatArray(200) }  // 200 bird species
    
    interpreter.run(input, output)
    
    // Get top 3 predictions
    val results = output[0].withIndex()
        .sortedByDescending { it.value }
        .take(3)
        .map { 
            Prediction(
                birdSpecies = classLabels[it.index],
                confidence = it.value
            )
        }
    
    return results
}

----------------------------------------------
# iOS implementation snippet (Swift)

// Load model
guard let modelPath = Bundle.main.path(forResource: "bird_classification_model_quantized", ofType: "tflite") else {
    fatalError("Failed to load model")
}
var interpreter: Interpreter?
do {
    interpreter = try Interpreter(modelPath: modelPath)
} catch {
    print("Error loading model: \(error)")
}

// Preprocess image
func preprocessImage(image: UIImage) -> Data {
    let imageWidth = 224
    let imageHeight = 224
    let imageChannels = 3
    
    guard let resizedImage = image.resize(to: CGSize(width: imageWidth, height: imageHeight)),
          let cgImage = resizedImage.cgImage else {
        fatalError("Failed to resize image")
    }
    
    let bytesPerRow = cgImage.bytesPerRow
    let imageData = Data(count: imageWidth * imageHeight * imageChannels * 4)
    
    imageData.withUnsafeMutableBytes { ptr in
        let buffer = ptr.bindMemory(to: Float32.self)
        for row in 0..<imageHeight {
            for col in 0..<imageWidth {
                let pixelInfo = cgImage.dataProvider!.data! + row * bytesPerRow + col * 4
                let r = Float32(pixelInfo[0]) / 255.0
                let g = Float32(pixelInfo[1]) / 255.0
                let b = Float32(pixelInfo[2]) / 255.0
                
                let offset = (row * imageWidth + col) * imageChannels
                buffer[offset] = r
                buffer[offset + 1] = g
                buffer[offset + 2] = b
            }
        }
    }
    
    return imageData
}

// Run inference
func classifyImage(image: UIImage) -> [Prediction] {
    let inputData = preprocessImage(image: image)
    var outputData = Data(count: 200 * 4)  // 200 classes, float32
    
    try? interpreter?.copy(inputData, toInputAt: 0)
    try? interpreter?.invoke()
    try? interpreter?.copy(toOutputAt: 0, outputData)
    
    let results = outputData.withUnsafeBytes { ptr in
        let floatPtr = ptr.bindMemory(to: Float32.self)
        var predictions = [(index: Int, confidence: Float32)]()
        
        for i in 0..<200 {
            predictions.append((i, floatPtr[i]))
        }
        
        return predictions.sorted { $0.confidence > $1.confidence }.prefix(3)
    }
    
    return results.map { 
        Prediction(species: classLabels[$0.index], confidence: $0.confidence) 
    }
}
"""
```

## 10. Web API and Backend Integration

```python
# Flask API for bird classification

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='bird_classification_model_quantized.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class names
class_names = [
    # List of 200 bird species names
    "Black-footed Albatross", "Laysan Albatross", 
    # ... rest of class names
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))
    
    # Preprocess the image
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Get top 3 predictions
    top_indices = np.argsort(output[0])[-3:][::-1]
    top_predictions = [
        {
            'species': class_names[i],
            'confidence': float(output[0][i])
        }
        for i in top_indices
    ]
    
    return jsonify({
        'predictions': top_predictions
    })

@app.route('/bird_info/<species>', methods=['GET'])
def bird_info(species):
    # This would typically connect to a database
    # For this example, we'll return hardcoded data
    bird_database = {
        "Black-footed Albatross": {
            "scientific_name": "Phoebastria nigripes",
            "habitat": "Open ocean of the North Pacific",
            "description": "A large seabird with dark plumage and a large bill.",
            "conservation_status": "Near Threatened"
        },
        # Add more species information...
    }
    
    if species in bird_database:
        return jsonify(bird_database[species])
    else:
        return jsonify({'error': 'Species information not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

## 11. System Testing and Continuous Improvement

```python
# Automated test script for the bird classification system

import requests
import os
import json
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time

# Test API endpoint
API_URL = "http://localhost:5000/predict"

def test_api_response_time(image_folder, num_tests=10):
    """Test the API response time with different images."""
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    response_times = []
    
    for _ in range(num_tests):
        # Select a random image
        img_file = random.choice(image_files)
        img_path = os.path.join(image_folder, img_file)
        
        # Time the API request
        start_time = time.time()
        
        with open(img_path, 'rb') as img:
            files = {'image': (img_file, img, 'image/jpeg')}
            response = requests.post(API_URL, files=files)
            
        end_time = time.time()
        
        if response.status_code == 200:
            response_time = end_time - start_time
            response_times.append(response_time)
            print(f"Image: {img_file}, Response time: {response_time:.3f}s")
            
            # Print the predictions
            predictions = response.json()['predictions']
            for pred in predictions:
                print(f"  {pred['species']}: {pred['confidence']:.4f}")
        else:
            print(f"Error with {img_file}: {response.status_code}")
    
    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        print(f"\nAverage response time: {avg_response_time:.3f}s")
        return avg_response_time
    return None

def test_model_with_new_images(image_folder, ground_truth_file):
    """Test the model with new images and compare with ground truth."""
    # Load ground truth labels
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    correct = 0
    total = 0
    confidence_scores = []
    
    for img_file, true_label in ground_truth.items():
        img_path = os.path.join(image_folder, img_file)
        
        if not os.path.exists(img_path):
            continue
            
        with open(img_path, 'rb') as img:
            files = {'image': (img_file, img, 'image/jpeg')}
            response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            predictions = response.json()['predictions']
            top_prediction = predictions[0]['species']
            confidence = predictions[0]['confidence']
            
            confidence_scores.append(confidence)
            
            if top_prediction == true_label:
                correct += 1
                result = "✓"
            else:
                result = "✗"
                
            print(f"{result} {img_file}: Predicted {top_prediction} ({confidence:.4f}), Truth: {true_label}")
            
            total += 1
    
    if total > 0:
        accuracy = correct / total
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"Average confidence: {avg_confidence:.4f}")
        return accuracy, avg_confidence
    return None, None

# User feedback collection system
def simulate_user_feedback(num_simulations=50):
    """Simulate user feedback to identify common failure modes."""
    # In a real system, this would be collected from actual users
    
    # Simulated feedback categories
    feedback_categories = {
        'incorrect_species': 0,
        'low_confidence': 0,
        'wrong_bird_family': 0,
        'unrecognized_species': 0,
        'bad_image_quality': 0,
        'unusual_lighting': 0,
        'bird_too_small': 0,
        'multiple_birds': 0
    }
    
    # Simulate random feedback
    for _ in range(num_simulations):
        # In a real system, this would be real user feedback
        category = random.choice(list(feedback_categories.keys()))
        feedback_categories[category] += 1
    
    # Visualize the feedback
    plt.figure(figsize=(12, 6))
    plt.bar(feedback_categories.keys(), feedback_categories.values())
    plt.title('Simulated User Feedback')
    plt.xlabel('Feedback Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Identify top issues
    sorted_feedback = sorted(feedback_categories.items(), key=lambda x: x[1], reverse=True)
    print("Top issues based on user feedback:")
    for category, count in sorted_feedback[:3]:
        print(f"- {category}: {count} reports")
    
    return feedback_categories

# Function to visualize model confidence distribution
def analyze_model_confidence(image_folder, num_samples=30):
    """Analyze the confidence distribution of the model predictions."""
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print("No images found in the folder.")
        return
        
    # Sample images if there are too many
    if len(image_files) > num_samples:
        image_files = random.sample(image_files, num_samples)
    
    confidences = []
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        
        with open(img_path, 'rb') as img:
            files = {'image': (img_file, img, 'image/jpeg')}
            response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            predictions = response.json()['predictions']
            top_confidence = predictions[0]['confidence']
            confidences.append(top_confidence)
    
    # Visualize confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=10, alpha=0.7)
    plt.axvline(x=0.5, color='r', linestyle='--', label='50% Confidence')
    plt.title('Model Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate statistics
    avg_confidence = sum(confidences) / len(confidences)
    low_confidence_pct = sum(1 for c in confidences if c < 0.5) / len(confidences) * 100
    
    print(f"Average prediction confidence: {avg_confidence:.4f}")
    print(f"Percentage of low confidence predictions (<50%): {low_confidence_pct:.2f}%")
    
    return confidences

# Continuous improvement process visualization
def visualize_improvement_process():
    """Visualize the continuous improvement process for the model."""
    
    # In a real project, this would plot actual metrics over time
    # Here we'll simulate the improvement process
    
    # Simulated data: weeks vs. accuracy
    weeks = list(range(1, 11))
    accuracy = [0.76, 0.78, 0.81, 0.82, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88]
    
    # Simulated data: weeks vs. model size (MB)
    model_size = [98, 98, 75, 75, 45, 45, 35, 25, 25, 25]
    
    # Simulated data: weeks vs. inference time (ms)
    inference_time = [320, 320, 300, 280, 220, 220, 180, 150, 140, 120]
    
    # Plot the improvement trends
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Accuracy trend
    axes[0].plot(weeks, accuracy, marker='o', linestyle='-', color='blue')
    axes[0].set_title('Model Accuracy Improvement')
    axes[0].set_xlabel('Week')
    axes[0].set_ylabel('Accuracy')
    axes[0].grid(True, alpha=0.3)
    
    # Model size trend
    axes[1].plot(weeks, model_size, marker='s', linestyle='-', color='green')
    axes[1].set_title('Model Size Optimization')
    axes[1].set_xlabel('Week')
    axes[1].set_ylabel('Model Size (MB)')
    axes[1].grid(True, alpha=0.3)
    
    # Inference time trend
    axes[2].plot(weeks, inference_time, marker='^', linestyle='-', color='orange')
    axes[2].set_title('Inference Time Optimization')
    axes[2].set_xlabel('Week')
    axes[2].set_ylabel('Inference Time (ms)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print improvement summary
    print("Continuous Improvement Summary:")
    print(f"Accuracy: {accuracy[0]:.2f} -> {accuracy[-1]:.2f} (+{(accuracy[-1] - accuracy[0]) * 100:.1f}%)")
    print(f"Model Size: {model_size[0]} MB -> {model_size[-1]} MB ({model_size[-1] / model_size[0]:.2f}x smaller)")
    print(f"Inference Time: {inference_time[0]} ms -> {inference_time[-1]} ms ({inference_time[0] / inference_time[-1]:.2f}x faster)")

# Call the test functions (in a real scenario)
# test_api_response_time("test_images")
# test_model_with_new_images("validation_images", "ground_truth.json")
# simulate_user_feedback()
# analyze_model_confidence("test_images")
# visualize_improvement_process()
```

## 12. Ethical Considerations in Wildlife AI

```python
# Wildlife photography ethics and AI considerations

"""
Ethical Considerations for Bird Species Classification AI

1. Data Collection Ethics:
   - Was the image dataset collected ethically?
   - Were bird habitats disturbed during photography?
   - Are there biases in the dataset (e.g., common vs. rare species, geographic representation)?
   - Were proper wildlife photography ethics followed?

2. Environmental Impact Considerations:
   - Could the app encourage inappropriate behavior toward birds?
   - How might we design the app to promote conservation ethics?
   - Should the app include endangered species information?
   - Could location data be used inappropriately by poachers?

3. Privacy and Security:
   - How to handle location metadata in bird photos?
   - Should breeding sites of rare birds be protected?
   - How to handle user data and photo collection?

4. Accuracy and Responsibility:
   - What are the consequences of misclassification?
   - Should the app show confidence levels?
   - How to communicate model limitations to users?
   - What disclaimers are needed for scientific/research use?

5. Positive Impact Opportunities:
   - How can the app contribute to citizen science?
   - Can the data help with conservation efforts?
   - How to educate users about bird conservation?
   - Could aggregate data help monitor population changes?
"""

# Example: Privacy-preserving location handling for rare species
def process_image_location(image_path, species_name, endangered_species_list):
    """
    Process image location metadata based on species conservation status.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS
        
        # Open the image
        img = Image.open(image_path)
        
        # Check if image has EXIF data
        if hasattr(img, '_getexif') and img._getexif():
            exif_data = img._getexif()
            
            # Extract GPS data if present
            gps_info = {}
            if exif_data:
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    if tag_name == "GPSInfo":
                        for gps_tag in value:
                            gps_info[GPSTAGS.get(gps_tag, gps_tag)] = value[gps_tag]
            
            # Check if the species is endangered
            is_endangered = species_name in endangered_species_list
            
            # Handle location data based on conservation status
            if is_endangered and gps_info:
                # For endangered species, generalize location (e.g., to county/region level)
                # rather than exact coordinates
                generalized_location = generalize_coordinates(
                    gps_info.get('GPSLatitude'),
                    gps_info.get('GPSLongitude')
                )
                return {
                    'location_type': 'generalized',
                    'location': generalized_location,
                    'original_coordinates_removed': True
                }
            elif gps_info:
                # For common species, return exact location if available
                return {
                    'location_type': 'exact',
                    'latitude': gps_info.get('GPSLatitude'),
                    'longitude': gps_info.get('GPSLongitude')
                }
        
        return {'location_type': 'unavailable'}
        
    except Exception as e:
        print(f"Error processing image location: {e}")
        return {'location_type': 'error', 'error': str(e)}

def generalize_coordinates(lat, lon):
    """
    Generalize coordinates to protect sensitive locations.
    For example, truncate to lower precision or snap to a grid.
    """
    if not lat or not lon:
        return None
    
    # Truncate precision to roughly city/county level 
    # (0.1 degree is approximately 11km)
    generalized_lat = round(lat * 10) / 10
    generalized_lon = round(lon * 10) / 10
    
    return {
        'latitude_approx': generalized_lat, 
        'longitude_approx': generalized_lon
    }

# Example: Creating educational content about bird conservation
def generate_conservation_message(species_name, conservation_status):
    """
    Generate appropriate conservation messaging based on species status.
    """
    if conservation_status == "Endangered":
        return (
            f"The {species_name} is endangered. Please be respectful of its habitat "
            f"and keep a safe distance. Consider reporting this sighting to local "
            f"conservation authorities as it can help monitoring efforts."
        )
    elif conservation_status == "Vulnerable":
        return (
            f"The {species_name} is vulnerable to extinction. Habitat loss and climate "
            f"change are affecting its population. Learn more about how you can support "
            f"conservation efforts for this beautiful bird."
        )
    elif conservation_status == "Near Threatened":
        return (
            f"While the {species_name} is not currently endangered, it is considered "
            f"near threatened. Being mindful of birds and their habitats helps ensure "
            f"they remain common in the future."
        )
    else:
        return (
            f"Learning to identify birds like the {species_name} is a great way to "
            f"connect with nature. Remember to observe wildlife respectfully and "
            f"from a distance."
        )
```

## 13. Project Summary and Key Learnings

This case study walked through the complete development cycle of a bird species classification system using deep learning. Here are the key learnings:

### Technical Insights

1. **Transfer Learning Efficiency**: Using MobileNetV2 as a base model significantly accelerated training and improved accuracy compared to training from scratch.

2. **Data Preprocessing Matters**: Proper normalization, augmentation, and resizing were critical for model performance.

3. **Model Optimization**: Techniques like quantization reduced model size by 4x and improved inference speed, making mobile deployment feasible.

4. **Evaluation Beyond Accuracy**: Looking at confusion matrices and class activation maps provided insights into model behavior that accuracy metrics alone couldn't reveal.

5. **Full-Stack Development**: The project required integrating various components including data processing, model training, API development, and mobile integration.

### Best Practices

1. **Start with a Clear Problem Definition**: Clearly defining the goal (bird species identification) helped guide all subsequent decisions.

2. **Exploratory Data Analysis**: Understanding the dataset characteristics informed preprocessing decisions and model architecture choices.

3. **Iterative Development**: Starting with a baseline model and then improving through fine-tuning and optimization led to better results.

4. **Continuous Testing**: Automated testing was built into the development process to ensure reliability.

5. **Ethical Considerations**: Thinking about potential misuse and conservation implications shaped the system design.

### Challenges and Solutions

1. **Limited Data**: Data augmentation and transfer learning helped overcome limitations in the dataset size.

2. **Computational Constraints**: Model optimization techniques made deployment on resource-constrained devices possible.

3. **Class Imbalance**: Stratified sampling during dataset splitting ensured representative training sets.

4. **Ethical Concerns**: Privacy-preserving location handling and conservation messaging addressed wildlife ethics concerns.

5. **Deployment Challenges**: TensorFlow Lite conversion and quantization made the model suitable for mobile deployment.

This end-to-end project demonstrated how computer vision can be applied to create practical applications that intersect with conservation, education, and citizen science, all while addressing technical, ethical, and environmental considerations.