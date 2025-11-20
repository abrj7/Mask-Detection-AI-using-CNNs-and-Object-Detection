# Mask Detection AI using CNNs and Object Detection

A small project that demonstrates mask detection using convolutional neural networks (CNNs) and object detection techniques. This repository contains training code, model definitions, and inference utilities to detect whether people in images or video frames are wearing face masks. The project is intended as an educational example and a starting point for building real-world mask/compliance detection systems.

## Project Overview

The goal of this project is to detect face masks in images and video using a hybrid approach:

- A CNN classifier trained to determine mask vs. no-mask for cropped face images.
- An object detection pipeline (e.g., Haar cascades, SSD/YOLO/Faster-RCNN) to locate faces in full images or video frames, then run the CNN classifier on each detected face.

This two-stage approach helps keep the classifier simple while allowing flexible choices for face localization.

## Features

- Training pipeline for a mask/no-mask CNN classifier.
- Integration with an object detector to localize faces in images or video.
- Scripts for training, evaluating, and running inference (live webcam or file-based).
- Utilities for dataset preprocessing, augmentation, and result visualization.

## Repository Structure

A typical layout (your actual repo may vary):

- data/
  - raw/                # Original images
  - processed/          # Cropped faces, train/val/test splits
- notebooks/            # Jupyter notebooks for exploration or training
- models/               # Saved model weights/checkpoints
- src/                  # Source code: model, data loader, utils
  - model.py
  - train.py
  - evaluate.py
  - detect.py
  - utils.py
- requirements.txt
- README.md

Adjust this section to match your repository's actual structure and filenames.

## Requirements

- Python 3.8+
- numpy
- pandas (optional, for dataset handling)
- opencv-python
- matplotlib (optional, for plots)
- scikit-learn
- tensorflow (or torch) — depending on which backend you use
- imutils (optional)
- tqdm (optional)

## Dataset

This project expects a face mask dataset organized into subfolders (for example):

- data/processed/train/with_mask/*.jpg
- data/processed/train/without_mask/*.jpg
- data/processed/val/with_mask/*.jpg
- data/processed/val/without_mask/*.jpg

Common public datasets you can use or adapt:

- "Real-World Masked Face Dataset"
- "Masked Face Recognition Dataset (MFRC)"
- Or create your own dataset by collecting, labeling, and cropping faces.

Include a preprocessing script (e.g., src/preprocess.py) to:
- Detect faces in raw images (using OpenCV Haar cascades or an object detector).
- Crop and resize faces (e.g., 64x64 or 128x128).
- Split into train/validation/test sets.

## Training

Run the training script to train the CNN classifier. Example usage:

```bash
python src/train.py \
  --data-dir data/processed/train \
  --val-dir data/processed/val \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --save-dir models/
```

Key training notes:
- Choose an appropriate image size (e.g., 128x128).
- Use data augmentation (flip, rotation, brightness) to improve robustness.
- Monitor validation accuracy and loss to determine early stopping.

## Evaluation

Evaluate a saved model on a test set:

```bash
python src/evaluate.py \
  --model models/mask_detector.h5 \
  --test-dir data/processed/test \
  --batch-size 32
```

Evaluation outputs typically include:
- Accuracy, precision, recall, F1-score
- Confusion matrix
- Example predictions and visualizations

## Inference (Real-time / Images / Video)

A detection pipeline typically performs these steps:
1. Detect faces in the frame using an object detector (Haar cascade, SSD, YOLO, etc.).
2. For each detected face, crop & preprocess and pass the crop to the CNN classifier.
3. Draw bounding boxes and labels (e.g., "Mask" or "No Mask") on the frame.

Example commands:

Run on an image:

```bash
python src/detect.py --model models/mask_detector.h5 --input-path examples/image.jpg --output-path results/out.jpg
```

Run on a video or webcam:

```bash
# Video
python src/detect.py --model models/mask_detector.h5 --input-path examples/video.mp4 --output-path results/video_out.mp4

# Webcam (use device index 0)
python src/detect.py --model models/mask_detector.h5 --webcam 0
```

Adjust flags and filenames according to the actual CLI implemented in your repo.

## Model Architecture

The classifier is a CNN that can be shallow for speed or deeper for accuracy. Example architecture options:
- A small custom CNN: Conv -> ReLU -> Pool -> Dropout -> Dense
- Transfer learning with a pretrained backbone: MobileNetV2, EfficientNet, ResNet (fine-tuned)

For real-time use, prefer lightweight backbones (MobileNet, EfficientNet-lite) to achieve higher FPS.

## Results & Visualizations

Include sample outputs:
- Confusion matrix and classification report
- Sample images with bounding boxes and labels
- Accuracy/loss training curves (plot in notebooks/ or results/)

If you have quantitative numbers (accuracy, precision, recall), put them here. For example:

- Validation accuracy: 96.5%
- Test accuracy: 95.2%
- Precision/Recall (No Mask): 94% / 93%

## Contact
Maintainer: abrj7 (GitHub)
