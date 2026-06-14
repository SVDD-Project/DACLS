# Singing Voice Deepfake Detection (DACLS Project)

## Authors
- **Luca Perlini**
- **Gioele Manuguerra**

## Overview
This repository contains a Deep Learning framework developed in Python and PyTorch for synthetic audio and deepfake singing voice detection. The project focuses on utilizing Generative Adversarial Networks (GANs) to accurately classify authentic and AI-generated audio samples, specifically targeting the SingFake dataset.

The complete signal processing and machine learning pipeline integrates:
- **Source Separation:** To isolate vocal tracks from background instrumentation.
- **Voice Activity Detection (VAD):** To filter out silent or non-vocal segments.
- **Feature Extraction:** To map raw audio signals into robust representations for the GAN architecture.

## Architecture and Active Development
The core network architectures and training loops are actively being developed and refined. The primary files currently under testing are:
- `dcgan_modelV2.py`: Contains the updated PyTorch implementation of the DCGAN generator and discriminator models.
- `TrainV2.py`: Handles the core training logic, loss computation, and metric logging.

## Prerequisites
Ensure you have the required dependencies installed before running the scripts. Key libraries include:
- Python 3.x
- PyTorch
- Optuna (for hyperparameter optimization)

## Usage

### 1. Single Training Run
To execute a standard, single training iteration of the DCGAN model, use the `run.py` script. Specify the architecture and the path to the training dataset:

```bash
python run.py --gan dcgan --dataset ~/dataset/Log3_real/BalancedTraining -log
