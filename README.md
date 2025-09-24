# OPEF-MsL：Interpretable protein–small molecule binding site prediction via optimal PLM embedding fusion and multi-scale learning
**OPEF-MsL** (Optimal PLM Embedding Fusion with Multi-scale Learning) is a framework for protein–small molecule binding site prediction that integrates optimal protein language model (PLM) embeddings with a multi-scale convolutional neural network (CNN) architecture enhanced by attention mechanisms.
OPEF-MsL leverages two large-scale pre-trained PLMs—**Ankh** and **ProstT5**—to generate informative residue-level embeddings. These embeddings are processed using PyTorch and the Hugging Face `transformers` library to extract rich protein representations.  
To ensure optimal model selection, we systematically compared several widely used PLMs, including **ESM-2** ([link](https://huggingface.co/facebook/esm2_t12_35M_UR50D)), **ProtT5** ([link](https://huggingface.co/Rostlab/prot_t5_xl_uniref50)), **ESM-1b** ([link](https://huggingface.co/facebook/esm1b_t33_650M_UR50S)), and **ProtBERT** ([link](https://huggingface.co/Rostlab/prot_bert)). Our comparative evaluation demonstrated that the combination of **Ankh** ([link](https://huggingface.co/ElnaggarLab/ankh-large/tree/main)) and **ProstT5** ([link](https://huggingface.co/Rostlab/prot_t5_xl_uniref50)) consistently achieved superior performance over single-model and cross-category combinations, and was therefore selected as the optimal embedding fusion strategy for downstream tasks.
By integrating optimal PLM embeddings with a multi-scale graph learning architecture, OPEF-MsL provides a robust and interpretable solution for predicting protein–small molecule interaction sites.

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
