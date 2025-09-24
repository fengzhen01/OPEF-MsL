# OPEF-MsL
OPEF-MsL: Interpretable protein–small molecule binding site prediction via optimal PLM embedding fusion and multi-scale learning
## Introduction
OPEF-MsL is a novel framework that integrates Optimal Protein Language Model (PLM) Embedding Fusion with Multi-scale Learning for protein–small molecule binding site prediction.
## 1. System Requirements
The source code was developed in **Python 3.9** using **PyTorch 2.5.1**.  
The required Python dependencies are listed below:
- **Python**: 3.9+
- **PyTorch**: 2.5.1
- **Torchvision**: 0.20.1
- **Torchaudio**: 2.5.1
- **CUDA**: 11.8 (recommended)
- **Transformers**: 4.46.3
- **SentencePiece**: 0.2.0
- **fair-esm**: 2.0.0
- **scikit-learn**: 1.5.2
- **pandas**: 2.2.3
- **matplotlib**: 3.9.4
- **pytorch-lightning**: 1.9.5

## Installation
### Usage
1. Prepare input protein sequences in FASTA format.
2. Run the prediction script:
   ```bash
   python run_opef_msl.py --input input.fasta --output results/
