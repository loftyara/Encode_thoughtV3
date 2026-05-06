# Encode Thought V3
*“Thoughts die the moment they are embodied by words.” A. Schopenhauer*

A neural architecture for extracting the invariant semantic core of text into a compact matrix of learnable slots and generating stable autoregressive text via native cross-attention. Unlike V2, which collapsed in closed-loop generation, V3 successfully closes the AR loop by injecting slots as encoder_outputs into a frozen T5-small decoder, adapting cross-attention matrices via LoRA, and training with curriculum embedding noise and slot-alignment loss. The pipeline is model-agnostic, operates on frozen base encoders, and achieves robust symbol-by-symbol generation without distribution collapse.
Full Paper: docs/Encode_thought.pdf

Version 1 (Theoretical): https://github.com/loftyara/Encode_thought/ or https://loftyara.github.io/encode_thought.html

Version 2 (Compression & TF Reconstruction): https://github.com/loftyara/Encode_thoughtV2/ or https://loftyara.github.io/encode_thoughtv2.html

## KEY HIGHLIGHTS
- Closed-Loop AR Stability: Generates coherent text for 80-128 tokens without lexical collapse or repetition loops.
- Native Cross-Attention Conditioning: Slots act as continuous encoder_outputs, accessed by the decoder at every generation step.
- Parameter Efficiency: Approximately 150K trainable parameters (slot projector + Cross-Attention LoRA). Less than 0.3 percent of the base model weight.
- Geometric Alignment and Drift Compensation: Slot-Alignment Loss explicitly ties slot geometry to the target semantic core. Curriculum embedding noise (alpha up to 0.12) trains the model to compensate for its own prediction drift.
- Reproducible Baseline: Theme Overlap ~0.72 on both train and val sets. Early stopping and unified checkpointing (best.pt) ensure fair generalization.
- Consumer GPU Friendly: Runs stably on 3.5-4.0 GB VRAM with AMP and gradient accumulation.

## ARCHITECTURE OVERVIEW
Text -> Token Embeddings (bert-mini) -> Set Transformer Encoder -> Hidden States

                                                    ↓  
Learnable Queries -> Cross-Attention -> Slot Matrix (16 x 192)

                                                    ↓  
SlotCrossProjector (192 -> 512) -> T5 Decoder (frozen) + Cross-Attention LoRA

                                                    ↓  
Continuous Autoregressive Generation

## INSTALLATION AND SETUP
Prerequisites:
    Python 3.12 or higher
    NVIDIA GPU with CUDA support (16 GB VRAM recommended)
    Git

## Setup steps:
Clone the repository:
```
git clone https://github.com/loftyara/Encode_thoughtV3.git
```
Navigate to the project directory:
```
cd Encode_thoughtV3
```
Create virtual environment:
```
python -m venv venv
```
Activate virtual environment:
Windows:
```
venv\Scripts\activate
```
Linux/macOS:
```
source venv/bin/activate
```
Install PyTorch:
```
pip install torch --index-url https://download.pytorch.org/whl/cu130
```
Install dependencies:
```
pip install -r requirements.txt
```

##QUICK START
Prepare Dataset:
```
cd scripts
python 01_download_dataset.py
```
Generate Embeddings (run for each base model):
```
python 02_gen_embeddings_bertmini.py
python 02_gen_embeddings_distilbert.py
python 02_gen_embeddings_minilm.py
python 02_gen_embeddings_tinybert.py
python 02_gen_embeddings_jina.py
```
Outputs are saved as chunked .pt files in ../data/processed/
Train the Slot Model:
```
cd ../src
python train.py
```
Checkpoints are saved to ../checkpoints/
Analyze and Reconstruct:
```
python analyze.py
```
Output: Side-by-side comparison of original vs recovered text, Theme Overlap metrics, and AR stability diagnostics.

## PROJECT STRUCTURE
Encode_thoughtV3/  
  checkpoints/ - Saved model weights (.pt)  
  data/  
&nbsp;&nbsp;&nbsp;    raw/ - TinyStories train.txt and val.txt  
&nbsp;&nbsp;&nbsp;    processed/ - Chunked embeddings (.pt)  
  docs/ - Documentation and paper (PDF)  
  scripts/ - Dataset download and embedding generation  
  src/ - Training scripts, analysis, model/dataset libraries  
&nbsp;&nbsp;&nbsp;    model.py  
&nbsp;&nbsp;&nbsp;    dataset.py  
&nbsp;&nbsp;&nbsp;    train.py  
&nbsp;&nbsp;&nbsp;    analyze.py  
  README.md  

## CURRENT EXPERIMENTAL RESULTS (TinyStories + bert-mini)
Metric: Theme Overlap (Val) ~0.72

Metric: Theme Overlap (Train) ~0.70

AR Stability: 80-128 tokens without syntactic collapse or lexical attractors

Trainable Parameters: ~150K (Projector + LoRA adapters)

Diagnosis: The architecture successfully closes the autoregressive loop and maintains syntax, style, and causal relationships. Residual mutations of names/objects and slight echoes at sentence boundaries are 
identified as the fundamental capacity limit of the fixed 16x192 bottleneck with frozen T5-small priors. This is not an optimization failure, but a clear architectural boundary.

## ROADMAP (NEXT PHASE: SEQUENTIAL SEMANTIC BLOCKS)
The fixed slot matrix acts as a global bottleneck for longer narratives. The next iteration will transition from a single static matrix to a sequence of interconnected semantic blocks.
Planned changes:
- Chunker: Split text into semantic windows (30-50 tokens) with overlap.
- Inter-Slot Attention: Lightweight causal transformer over the sequence of slot matrices to resolve coreferences and pass context between fragments.
- Transition Mechanism: Soft interpolation or explicit context slot copying to prevent semantic breaks at boundaries.
- Goal: Remove the length constraint, eliminate entity drift at fragment boundaries, and scale to full narratives without changing the core local stack.
All experiments will follow a strict one-dimensional protocol against the current V3 baseline.

## CONTACT
Author: Dmitri Lyubimkov  
Email: loftlong@gmail.com  
GitHub: @loftyara  

