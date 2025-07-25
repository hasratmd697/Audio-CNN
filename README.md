# Audio CNN

<img width="1920" height="1001" alt="Screenshot 2025-07-14 094831" src="https://github.com/user-attachments/assets/d8ac076b-6cc7-4cc8-88b0-48fb5b1ed0c0" />


## Overview

In this project, I trained and deployed an audio classification Convolutional Neural Network (CNN) from scratch using PyTorch. The model is capable of classifying environmental sounds, such as dog barks and bird chirps, from audio files. I implemented advanced techniques including Residual Networks (ResNet), data augmentation through audio mixing, and Mel Spectrogram transformations to create a robust training pipeline. To make the model accessible and interactive, I built a full-stack web dashboard using Next.js, React, and Tailwind CSS (following the T3 Stack architecture). The dashboard allows users to upload audio files and visualize how the model processes and interprets them through its internal layers. All tools and services used in this project are open-source and freely available.

## Features:

- 🧠 Deep Audio CNN for sound classification
- 🧱 ResNet-style architecture with residual blocks
- 🎼 Mel Spectrogram audio-to-image conversion
- 🎛️ Data augmentation with Mixup & Time/Frequency Masking
- ⚡ Serverless GPU inference with Modal
- 📊 Interactive Next.js & React dashboard
- 👁️ Visualization of internal CNN feature maps
- 📈 Real-time audio classification with confidence scores
- 🌊 Waveform and Spectrogram visualization
- 🚀 FastAPI inference endpoint
- ⚙️ Optimized training with AdamW & OneCycleLR scheduler
- 📈 TensorBoard integration for training analysis
- 🛡️ Batch Normalization for stable & fast training
- 🎨 Modern UI with Tailwind CSS & Shadcn UI
- ✅ Pydantic data validation for robust API requests

## Setup

Follow these steps to install and set up the project.

### Clone the Repository

```bash
https://github.com/hasratmd697/Audio-CNN.git
```

### Install Python

Download and install Python if not already installed. Use the link below for guidance on installation:
[Python Download](https://www.python.org/downloads/)

Create a virtual environment with **Python 3.12**.

### Backend

Navigate to folder:

```bash
cd audio-cnn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Modal setup:

```bash
modal setup
```

Run on Modal:

```bash
modal run main.py
```

Deploy backend:

```bash
modal deploy main.py
```

### Frontend

Install dependencies:

```bash
cd audio-cnn-visualisation
npm i
```

Run:

```bash
npm run dev
```
