#!/usr/bin/env python3
"""
Checkpoint Demo Script for Google Colab and Kaggle

This script demonstrates how to use the checkpoint save/load functionality
for training models on environments with session timeouts.

Usage Examples:
--------------

1. Start fresh training with periodic checkpoints (recommended for Colab/Kaggle):
   python linevul_main.py \
     --output_dir=./saved_models \
     --model_name_or_path=microsoft/codebert-base \
     --tokenizer_name=microsoft/codebert-base \
     --train_data_file=../data/train.csv \
     --eval_data_file=../data/val.csv \
     --test_data_file=../data/test.csv \
     --epochs=10 \
     --train_batch_size=16 \
     --eval_batch_size=32 \
     --do_train \
     --checkpoint_steps=500  # Save every 500 steps for timeout protection

2. List available checkpoints:
   python linevul_main.py --output_dir=./saved_models --list_checkpoints

3. Resume from the latest checkpoint:
   python linevul_main.py \
     --output_dir=./saved_models \
     --model_name_or_path=microsoft/codebert-base \
     --tokenizer_name=microsoft/codebert-base \
     --train_data_file=../data/train.csv \
     --eval_data_file=../data/val.csv \
     --test_data_file=../data/test.csv \
     --epochs=10 \
     --train_batch_size=16 \
     --eval_batch_size=32 \
     --do_train \
     --checkpoint_steps=500 \
     --resume_from_checkpoint=./saved_models/checkpoint-latest/checkpoint.pt

4. Resume from a specific epoch checkpoint:
   python linevul_main.py \
     --resume_from_checkpoint=./saved_models/checkpoint-epoch-3/checkpoint.pt \
     ... (other args same as above)

Google Colab Tips:
-----------------
1. Mount Google Drive to save checkpoints persistently:
   from google.colab import drive
   drive.mount('/content/drive')

   # Then use: --output_dir=/content/drive/MyDrive/linevul_checkpoints

2. Use checkpoint_steps to save frequently:
   --checkpoint_steps=200  # Saves every 200 steps

3. The script automatically saves:
   - checkpoint-epoch-N/checkpoint.pt  (after each epoch)
   - checkpoint-latest/checkpoint.pt   (always points to most recent)
   - checkpoint-best-f1/model.bin      (best model by F1 score)

Kaggle Tips:
-----------
1. Save to /kaggle/working/ which persists across kernel restarts
   --output_dir=/kaggle/working/checkpoints

2. Download checkpoints before session ends:
   from IPython.display import FileLink
   display(FileLink('/kaggle/working/checkpoints/checkpoint-latest/checkpoint.pt'))

What gets saved in each checkpoint:
----------------------------------
- Model weights (state_dict)
- Optimizer state (momentum, adaptive learning rates)
- Learning rate scheduler state
- Current epoch and global step
- Best F1 score achieved
- Training loss values
- Random states (for reproducibility)
- Mid-epoch step position (if checkpoint_steps is used)
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from linevul_main import list_checkpoints, print_available_checkpoints


def demo_list_checkpoints(output_dir="./saved_models"):
    """Demo function to list available checkpoints."""
    print("\n" + "="*50)
    print("CHECKPOINT LISTING DEMO")
    print("="*50)

    checkpoints = list_checkpoints(output_dir)

    if not checkpoints:
        print(f"\nNo checkpoints found in: {output_dir}")
        print("\nTo create checkpoints, run training first:")
        print("  python linevul_main.py --do_train --output_dir=./saved_models ...")
        return

    print_available_checkpoints(output_dir)

    # Show example resume command
    latest = next((c for c in checkpoints if c['name'] == 'checkpoint-latest'), None)
    if latest:
        print("\nQuick resume command:")
        print(f"  --resume_from_checkpoint={latest['path']}")


def demo_checkpoint_info(checkpoint_path):
    """Load and display info from a specific checkpoint."""
    import torch

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print("\n" + "-"*40)
    print("CHECKPOINT CONTENTS")
    print("-"*40)
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Global Step: {checkpoint.get('global_step', 'N/A')}")
    print(f"Best F1: {checkpoint.get('best_f1', 'N/A')}")
    print(f"Training Loss: {checkpoint.get('tr_loss', 'N/A')}")
    print(f"Step in Epoch: {checkpoint.get('step_in_epoch', 'End of epoch')}")
    print(f"Total Steps in Epoch: {checkpoint.get('total_steps_in_epoch', 'N/A')}")

    # Show model size
    model_state = checkpoint.get('model_state_dict', {})
    num_params = sum(p.numel() for p in model_state.values())
    print(f"Model Parameters: {num_params:,}")

    print("-"*40)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Checkpoint Demo Utility")
    parser.add_argument("--output_dir", default="./saved_models",
                        help="Directory containing checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint file to inspect")
    args = parser.parse_args()

    if args.checkpoint:
        demo_checkpoint_info(args.checkpoint)
    else:
        demo_list_checkpoints(args.output_dir)
