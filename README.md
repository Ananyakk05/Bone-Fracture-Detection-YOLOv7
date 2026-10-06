# Bone Fracture Detection Using YOLOv7

## Overview

This project implements a deep learning-based bone fracture detection system using the YOLOv7 object detection architecture. The system analyzes X-ray images and identifies fracture regions by generating bounding boxes and confidence scores.

The project also includes an ONNX-based inference pipeline, a desktop GUI, and a Streamlit web application for easy fracture detection.

## Features

- Bone fracture detection from X-ray images
- YOLOv7-based object detection
- Bounding box localization of fracture regions
- Confidence score generation
- ONNX-based inference
- PySide6 desktop GUI
- Streamlit web application
- Prediction result visualization

## Dataset

The project uses the GRAZPEDWRI-DX pediatric wrist trauma X-ray dataset.

The dataset contains annotated pediatric wrist radiographs with different abnormality classes, including fracture annotations.

The dataset itself is not included in this repository because of its size and dataset distribution restrictions.

## Technologies Used

- Python
- YOLOv7
- PyTorch
- ONNX
- ONNX Runtime
- OpenCV
- PySide6
- Streamlit
- NumPy
- Pandas

## Project Structure

```text
YOLOv7-Bone-Fracture-Detection/
│
├── app/
│   └── webapp.py
│
├── gui/
│   └── gui.py
│
├── icons/
├── images/
├── predictions/
│
├── data_loader.py
├── inference_onnx.py
├── split.py
├── yolov7_cfg.yaml
├── yolov7_data.yaml
├── requirements.txt
├── README.md
└── LICENSE