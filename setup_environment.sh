#!/bin/bash
# Conda environment setup for RGB-DINO-Gaze Phase 1
set -e  # Exit on error

# Create conda environment
conda create -n rgb-dino-gaze python=3.10 -y
conda activate rgb-dino-gaze

# Install PyTorch (CUDA 11.8)
pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install mediapipe==0.10.9
pip install transformers==4.36.2 datasets==2.15.0
pip install opencv-python==4.8.1.78 numpy==1.26.2 pandas==2.1.4 scikit-learn==1.3.2
pip install tqdm==4.66.1
pip install scipy

# Verify installations
echo "Verifying key packages:"
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"
python -c "import mediapipe as mp; print(f'MediaPipe version: {mp.__version__}')"
python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('facebook/dinov2-base'); print('DINOv2 loaded successfully')"

echo "Environment setup complete! Activate with: conda activate rgb-dino-gaze"
