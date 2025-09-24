from time import sleep
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
import gc
import os
import pandas as pd


def read_data_file_trip(filename):
    with open(filename) as f:
        data = f.readlines()

    results = []
    block = len(data) // 2
    for index in range(block):
        item1 = data[index * 2 + 0].split()
        name = item1[0].strip()
        seq = item1[1].strip()
        item2 = data[index * 2 + 1].split()
        results.append([name, seq, len(seq), item2[1].strip()])
    return results


def read_data_file_trip_from_fasta(filename):
    with open(filename) as f:
        data = f.readlines()

    return [
        [data[i].replace('>', '').strip(), data[i + 1].strip()]
        for i in range(0, len(data), 2)
    ]


def extract_prostt5_embeddings(file, dest_folder):
    sequences = read_data_file_trip(file)

    for i, (name, seq, length, label) in enumerate(sequences, 1):
        print(f'Progress {i}/{len(sequences)} - {name}')

        if os.path.exists(os.path.join(dest_folder, name + '.data')):
            print(f'{name} already exists')
            continue

        with open(os.path.join(dest_folder, name + '.label'), 'w') as f:
            f.write(label)

        processed_seq = ' '.join(list(seq))
        processed_seq = re.sub(r"[UZOB]", "X", processed_seq)

        # Tokenize
        ids = tokenizer.batch_encode_plus(
            [processed_seq],
            add_special_tokens=True,
            padding="max_length",
            max_length=1024,  
            truncation=True
        )
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embeddings = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state

        # seq_len = (attention_mask[0] == 1).sum().item() - 1  
        seq_len = (attention_mask[0] == 1).sum().item() 
        seq_emb = embeddings[0][1:seq_len].cpu().numpy() 

        np.savetxt(
            os.path.join(dest_folder, name + '.data'),
            seq_emb,
            delimiter=',',
            fmt='%s'
        )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    local_model_path = "D:/fengzhen/OPEF-MsL-main/ProstT5"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spiece.model"     
    ]

    missing_files = [
        f for f in required_files
        if not os.path.exists(os.path.join(local_model_path, f))
    ]

    if missing_files:
        raise FileNotFoundError(f"Required model file(s) missing: {missing_files}")

    print(f"Loading ProstT5 model from local path: {local_model_path}")
    tokenizer = T5Tokenizer.from_pretrained(
        local_model_path,
        do_lower_case=False
    )
    model = T5EncoderModel.from_pretrained(local_model_path)
    model = model.to(device)
    model.eval()

    input_files = [
        'D:/fengzhen/OPEF-MsL-main/DataSet/UniProtSMB/SMB2_Train.txt',
        'D:/fengzhen/OPEF-MsL-main/DataSet/UniProtSMB/SMB2_Test.txt'
    ]

    output_dir = 'D:/xiangmu/3embedding/prostT5_embedding_SMB2/'
    os.makedirs(output_dir, exist_ok=True)

    for file in input_files:
        print(f"\nProcessing: {file}")
        extract_prostt5_embeddings(file, output_dir)

    print('\n---- Finished ----')
