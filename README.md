# Audio CNN

<img width="1280" height="720" alt="thumbnail" src="https://github.com/user-attachments/assets/4a46d85a-f2fb-4ea3-973e-b41881f8e274" />


## Overview

Hi 🤙 In this video, you'll learn to train and deploy an audio classification CNN from scratch with PyTorch. I'll cover all the required concepts, so no prior experience is needed. The model will classify sounds like a dog barking or birds chirping from an audio file. You'll work with advanced techniques like Residual Networks (ResNet), data mixing, and Mel Spectrograms to build a robust training pipeline. Afterwards, we'll build a dashboard using Next.js and React to upload audio and visualize the model's internal layers to see what it "sees". The project uses Python, PyTorch, Next.js, React, and Tailwind, based on the T3 Stack. You can build along with me from start to finish. All services used are 100% free for you to use.

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
git clone https://github.com/Andreaswt/audio-cnn.git
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
