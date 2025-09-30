#!/usr/bin/env python3
"""
Setup script to get your training environment ready.
Run this before training to install dependencies and verify GPU setup.
"""

import subprocess
import sys
import torch

def install_requirements():
    """Install Python packages from requirements.txt"""
    print("Installing Python requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✓ Requirements installed")

def check_gpu():
    """Verify CUDA/GPU setup"""
    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠ CUDA not available - will train on CPU (very slow)")

    print(f"Device that will be used: {'cuda' if torch.cuda.is_available() else 'cpu'}")

def verify_transformers():
    """Test that transformers library works"""
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("✓ Transformers library working")
        print("✓ GPT-2 tokenizer downloaded")
    except Exception as e:
        print(f"✗ Error with transformers: {e}")

if __name__ == "__main__":
    print("Setting up Go code model training environment...")
    print("="*50)

    install_requirements()
    print()

    check_gpu()
    print()

    verify_transformers()
    print()

    print("Setup complete! Run 'python train_go_model.py' to start training.")