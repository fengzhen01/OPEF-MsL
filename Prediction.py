import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer
import glob

# Import custom modules from your project
from model import Module
from data_loader import coll_paddding


class PredictionDataset(Dataset):
    """Dataset for prediction using ProstT5 and Ankh embeddings"""

    def __init__(self, X, embedding_folder):
        self.X = X
        self.embedding_folder = embedding_folder

    def __getitem__(self, index):
        filename = self.X[index]

        try:
            # Load ProstT5 embeddings
            df_prostt5 = pd.read_csv(f'{self.embedding_folder}/{filename}_T5.data', header=None)
            prot_prostt5 = df_prostt5.values.astype(float)
            prot_prostt5 = torch.tensor(prot_prostt5)

            # Load Ankh embeddings
            df_ankh = pd.read_csv(f'{self.embedding_folder}/{filename}_Ankh.data', header=None)
            prot_ankh = df_ankh.values.astype(float)
            prot_ankh = torch.tensor(prot_ankh)

            # Combine two embeddings with length alignment
            min_len = min(prot_prostt5.size(0), prot_ankh.size(0))
            prot_combined = torch.cat((prot_prostt5[:min_len], prot_ankh[:min_len]), dim=1)

            return prot_combined

        except Exception as e:
            print(f"Error loading embeddings for {filename}: {e}")
            return torch.tensor([])

    def __len__(self):
        return len(self.X)


def read_query_file(queryfile):
    """Read query sequence file"""
    with open(queryfile, 'r') as f:
        data = f.readlines()
    return [line.strip() for line in data]


def generate_prostt5_embeddings(seq, seq_name, output_folder, device, local_model_path):
    """Generate ProstT5 embeddings using local model"""
    print("Loading ProstT5 model from local...")
    tokenizer = T5Tokenizer.from_pretrained(local_model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(local_model_path)
    model = model.to(device)
    model.eval()

    # Sequence preprocessing
    processed_seq = ' '.join(list(seq))
    processed_seq = re.sub(r"[UZOB]", "X", processed_seq)

    # Tokenize with proper padding/truncation
    ids = tokenizer.batch_encode_plus(
        [processed_seq],
        add_special_tokens=True,
        padding="max_length",
        max_length=1024,
        truncation=True
    )
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # Extract features
    with torch.no_grad():
        embeddings = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

    # Save features (remove special tokens)
    seq_len = (attention_mask[0] == 1).sum().item()
    seq_emb = embeddings[0][1:seq_len].cpu().numpy()

    np.savetxt(
        f'{output_folder}/{seq_name}_T5.data',
        seq_emb,
        delimiter=',',
        fmt='%.6f'
    )

    print('ProstT5 embedding generation completed')
    torch.cuda.empty_cache()
    return seq_emb.shape[1]  # Return embedding dimension


def generate_ankh_embeddings(seq, seq_name, output_folder, device, local_model_path):
    """Generate Ankh embeddings using local model"""
    print("Loading Ankh model from local...")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = T5EncoderModel.from_pretrained(local_model_path)
    model = model.to(device)
    model.eval()

    # Sequence preprocessing
    seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")

    # Tokenize
    inputs = tokenizer(seq, return_tensors="pt", max_length=1024, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Extract features
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state[0]

    # Remove special tokens
    seq_len = (attention_mask[0] == 1).sum().item()
    embedding = embedding[1:seq_len - 1].cpu().numpy()

    np.savetxt(
        f'{output_folder}/{seq_name}_Ankh.data',
        embedding,
        delimiter=',',
        fmt='%.6f'
    )

    print('Ankh embedding generation completed')
    torch.cuda.empty_cache()
    return embedding.shape[1]  # Return embedding dimension


def predict_single_task(model_path, query_name, embedding_folder, device, input_dim):
    """Predict using single-task model"""
    # Load model
    model = Module(input_dim, False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Create dataset and dataloader
    dataset = PredictionDataset([query_name], embedding_folder)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=coll_paddding)

    print(f"=== Predicting with {os.path.basename(model_path)} ===")

    with torch.no_grad():
        for prot_xs, lengths in dataloader:
            if prot_xs.nelement() == 0:  # Skip if embeddings failed to load
                print("Error: No embeddings found")
                return None

            prot_xs = prot_xs.to(device)
            lengths = lengths.to(device)

            # Get predictions
            outputs = model(prot_xs, lengths)
            outputs = outputs[0]  # Remove batch dimension

            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            positive_probs = probabilities[:, 1].cpu().numpy()

            return positive_probs


def main():
    """Main prediction function for single-task SMB prediction"""
    # Device configuration
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"Using device: {device}")

    # Configuration - Update these paths according to your setup
    query_file = 'query/query.txt'     # Path to your query file
    output_folder = 'query'            # Output folder for embeddings and results
    model_folder = 'pre_model/SMB'     # Single task model folder, load .pkl file

    # Local model paths (from your extract scripts)
    prostt5_model_path = "D:/fengzhen/3OPEF-MsL-main/ProstT5/"
    ankh_model_path = "D:/fengzhen/3OPEF-MsL-main/Ankh/Ankh_Large/"

    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    # Read query sequence
    sequences = read_query_file(query_file)
    if not sequences:
        print("No sequences found in query file")
        return

    protein_sequence = sequences[0]
    query_name = 'SMB_query'

    print(f"Processing sequence: {query_name}")
    print(f"Sequence length: {len(protein_sequence)}")

    # Generate embeddings
    print("\nGenerating embeddings...")
    prostt5_dim = generate_prostt5_embeddings(protein_sequence, query_name, output_folder, device, prostt5_model_path)
    ankh_dim = generate_ankh_embeddings(protein_sequence, query_name, output_folder, device, ankh_model_path)

    input_dim = prostt5_dim + ankh_dim
    print(f"Combined input dimension: {input_dim} (ProstT5: {prostt5_dim} + Ankh: {ankh_dim})")

    # Prepare results dictionary
    results = {
        'position': list(range(1, len(protein_sequence) + 1)),
        'residue': list(protein_sequence),
        'SMB_probability': []
    }

    # Find all model files
    model_files = list(glob.iglob(f'{model_folder}/*.pkl'))
    if not model_files:
        print(f"No model files found in {model_folder}")
        # If no model files, create a dummy result for testing
        results['SMB_probability'] = [0.0] * len(protein_sequence)
    else:
        print(f"Found {len(model_files)} model files")

        # Run predictions for each model and average results
        all_predictions = []
        successful_predictions = 0

        for model_path in model_files:
            try:
                predictions = predict_single_task(model_path, query_name, output_folder, device, input_dim)
                if predictions is not None and len(predictions) == len(protein_sequence):
                    all_predictions.append(predictions)
                    successful_predictions += 1
                    print(f"Prediction successful with {os.path.basename(model_path)}")
                else:
                    print(f"Prediction length mismatch with {os.path.basename(model_path)}")
            except Exception as e:
                print(f"Error predicting with {model_path}: {e}")

        # Average predictions across models
        if all_predictions:
            averaged_predictions = np.mean(all_predictions, axis=0)
            results['SMB_probability'] = averaged_predictions.tolist()
        else:
            print("No successful predictions")
            results['SMB_probability'] = [0.0] * len(protein_sequence)

    # Create DataFrame and save results
    df = pd.DataFrame(results)
    output_file = f'{output_folder}/SMB_prediction_results.xlsx'
    df.to_excel(output_file, index=False)
    print(f"\nPrediction results saved to: {output_file}")

    # Print summary
    print(f"\n=== Prediction Summary ===")
    print(f"Sequence length: {len(protein_sequence)}")
    print(f"Average SMB probability: {np.mean(results['SMB_probability']):.4f}")
    print(f"Max SMB probability: {np.max(results['SMB_probability']):.4f}")
    print(f"Min SMB probability: {np.min(results['SMB_probability']):.4f}")

    # Count residues with probability > 0.5
    high_prob_count = np.sum(np.array(results['SMB_probability']) > 0.5)
    print(f"Residues with probability > 0.5: {high_prob_count}")

    # Show top predictions
    if len(results['SMB_probability']) > 0:
        probabilities = np.array(results['SMB_probability'])
        top_indices = np.argsort(probabilities)[-5:][::-1]
        print(f"\nTop 5 predictions:")
        for i, idx in enumerate(top_indices, 1):
            print(f"{i}. Position {idx + 1} ({results['residue'][idx]}): {probabilities[idx]:.4f}")


if __name__ == "__main__":
    main()
