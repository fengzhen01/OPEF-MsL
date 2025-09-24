# OPEF-MsL
OPEF-MsL: Interpretable protein–small molecule binding site prediction via optimal PLM embedding fusion and multi-scale learning
## Introduction
OPEF-MsL is a novel framework that integrates Optimal Protein Language Model (PLM) Embedding Fusion with Multi-scale Learning for protein–small molecule binding site prediction.
## 1. System Requirements

The source code was developed in **Python 3.10** using **PyTorch 2.6.0** with CUDA 11.8 support.  
The required Python dependencies are listed below:

- **Python**: 3.10
- **PyTorch**: 2.6.0+cu118
- **Torchvision**: *(match PyTorch version if needed)*
- **Torchaudio**: 2.6.0+cu118
- **CUDA**: 11.8 (recommended)
- **Transformers**: *(match your project, e.g., 4.46.x)*
- **SentencePiece**: 0.1.99
- **fair-esm**: 2.0.0
- **scikit-learn**: 1.6.1
- **pandas**: 2.2.3
- **matplotlib**: 3.10.5
- **shap**: 0.48.0
- **datasets**: 2.21.0
- **numpy / scipy**: 2.2.4 / 1.15.2

## Installation
### Usage
1. Prepare input protein sequences in FASTA format.
2. Run the prediction script:
   ```bash
   python run_opef_msl.py --input input.fasta --output results/
