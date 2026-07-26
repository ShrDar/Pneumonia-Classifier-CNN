# Pneumonia Detection using Convolutional Neural Networks

A full-stack deep learning application for detecting **Pneumonia** from chest X-ray images using Convolutional Neural Networks (CNNs). The application allows users to upload an X-ray image, select from multiple trained models, receive a prediction with confidence and probability scores, and visualize the model's decision using **Grad-CAM++**.

---

# Overview

Pneumonia is a serious respiratory disease that can be diagnosed through chest X-ray imaging. This project demonstrates the application of deep learning techniques for binary image classification by classifying chest X-rays into:

* **Normal**
* **Pneumonia**

The application provides an interactive interface where users can:

* Upload chest X-ray images
* Select among multiple trained models
* View prediction results
* View model confidence and probability
* Generate Grad-CAM++ visualizations
* Download the generated Grad-CAM image

The project consists of a **FastAPI backend** responsible for inference and image processing and a **Next.js frontend** that provides the user interface.

---

# Features

* Chest X-ray image upload
* Binary classification (Normal / Pneumonia)
* Multiple model selection
* Two custom CNN baseline models
* Transfer Learning using ResNet18
* Frozen and Fine-Tuned transfer learning models
* Confidence score calculation
* Probability score calculation
* Grad-CAM++ visualization
* Downloadable Grad-CAM image
* REST API built with FastAPI
* Responsive frontend built with Next.js
* Type-safe frontend using TypeScript

---

# Tech Stack

## Frontend

* Next.js
* React
* TypeScript
* Tailwind CSS

## Backend

* FastAPI
* PyTorch
* TorchVision
* Pillow
* OpenCV
* NumPy
* Matplotlib
* Grad-CAM++

## Machine Learning

* Custom Convolutional Neural Networks
* ResNet18 Transfer Learning
* Binary Classification
* BCEWithLogitsLoss
* Adam Optimizer

---

# Deep Learning Models

The application provides four trained CNN models

## Baseline Model 1

* Custom CNN architecture
* Trained from scratch

## Baseline Model 2

* Improved custom CNN architecture
* Trained from scratch

## Transfer Learning (Frozen)

* ResNet18 pretrained on ImageNet
* Feature extractor frozen
* Custom classification head trained

## Transfer Learning (Fine-Tuned)

* ResNet18 pretrained on ImageNet
* Entire network fine-tuned
* Custom classification head

---

# Installation

## Clone the repository

```bash
git clone https://github.com/ShrDar/Pneumonia-Classifier-CNN.git

cd <repository>
```

---

## Backend Setup

Create venv

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Frontend Setup

```bash
cd frontend

npm i
```

---

# Running the Application

## Start the backend

```bash
uvicorn main:app --reload
```

Backend will be available at:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

## Start the frontend

```bash
npm run dev
```

Frontend:

```text
http://localhost:3000
```

---

