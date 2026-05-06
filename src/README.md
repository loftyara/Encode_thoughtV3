# Source Directory (src/)

Contains the core model implementation, dataset handling, training pipeline, and autoregressive analysis scripts for the Encode_thoughtV3 project.

## Core Libraries
- `model.py` – Main `EncodeThoughtModel` architecture. Implements a Set Transformer encoder with ISAB blocks, a Learnable Queries bottleneck for slot extraction (16 slots x 192 dim), a SlotCrossProjector for geometric alignment to T5 space, and a frozen T5-small decoder wrapper with native cross-attention conditioning.
- `dataset.py` – Custom `StoryEmbeddingDataset` class for  for loading flat chunked .pt embeddings (embeddings_{model}_{split}chunk\*.pt). Supports PRELOAD mode, automatic dict-format normalization, and contiguous tensor preparation for fast GPU transfer.

## Training Script (`train.py`)
Single unified training pipeline for the prajjwal1/bert-mini encoder configuration.
Key features:
- Loads precomputed embeddings and prepares pinned RAM tensors
- Wraps T5-small cross-attention matrices (q, k, v, o) with PEFT LoRA (r=8, alpha=16)
- Trains with combined CrossEntropy + Slot-Alignment Loss (cosine)
- Applies Curriculum Embedding Noise (alpha up to 0.12) for AR loop stabilization
- Uses AMP, gradient accumulation (2 steps), gradient clipping (1.0), and early stopping (patience=8)
- Saves a single unified checkpoint to ../checkpoints/best.pt strictly on validation loss improvement
Files: `train.py`

## Analysis Script (`analyze.py`)
Autoregressive diagnostics and text recovery pipeline.
Key features:
- Loads the best checkpoint and re-wraps T5 with matching LoRA configuration
- Runs closed-loop autoregressive generation using native T5 cross-attention (slots injected as encoder_outputs)
- Uses sampling parameters: temperature=0.8, top_p=0.9, repetition_penalty=1.2
- Computes Theme Overlap metric and exact match rate
- Prints side-by-side original vs recovered text with diagnostic summaries
Files: `analyze.py`

## Typical Workflow
1. Ensure embeddings exist in `../data/processed/`
2. Run training: `python train.py`
3. Monitor console output for Train/Val loss and early stopping. Checkpoint saves automatically to ../checkpoints/best.pt
4. Run analysis: `python analyze.py`
5. Review AR stability, Theme Overlap metrics, and recovered samples in the console output


## Notes
- All scripts import model.py and dataset.py via relative paths. No external YAML configs are used; hyperparameters are fixed in the script headers.
- Requires PyTorch with CUDA support (CPU fallback is disabled by design), transformers, peft, and internet access for initial Hugging Face model downloads.
- The pipeline is strictly CUDA-only and optimized for consumer GPUs.
- Analysis runs autoregressive generation natively through T5; no embedding-to-token nearest-neighbor search is used.
- Checkpoint structure includes model weights, optimizer state, val loss, and architecture config for exact reproducibility.
