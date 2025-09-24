from time import sleep
import torch
from transformers import BertTokenizer, BertModel
import re
import numpy as np
import gc
import os
import pandas as pd

def read_data_file_trip(filename):
    f = open(filename)
    data = f.readlines()
    f.close()

    results = []
    block = len(data) // 2
    for index in range(block):
        item1 = data[index * 2 + 0].split()
        name = item1[0].strip()
        seq = item1[1].strip()
        item2 = data[index * 2 + 1].split()
        item = []
        item.append(name)
        item.append(seq)
        item.append(len(seq))
        item.append(item2[1].strip())
        results.append(item)
    return results

def read_data_file_trip_from_fasta(filename):
    f = open(filename)
    data = f.readlines()
    f.close()

    results = []
    block = len(data) // 2
    for index in range(block):
        name = data[index * 2 + 0].replace('>', '').strip()
        seq = data[index * 2 + 1].strip()
        item = []
        item.append(name)
        item.append(seq)
        results.append(item)
    return results

def extratdata(file, destfolder, tokenizer, model, device):
    student_tuples = read_data_file_trip(file)
    i = 1
    recordlength = len(student_tuples)
    for name, seq, length, label in student_tuples:
        print(f'progress {i}/{recordlength}')
        i += 1

        with open(os.path.join(destfolder, name + '.label'), 'w') as f:
            f.write(','.join(l for l in label))

        newseq = ' '.join(s for s in seq)
        newseq = re.sub(r"[UZOB]", "X", newseq)

        ids = tokenizer.batch_encode_plus([newseq], add_special_tokens=True, padding=True, return_tensors="pt")
        input_ids = ids['input_ids'].to(device)
        attention_mask = ids['attention_mask'].to(device)

        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)

        embedding = embedding.last_hidden_state.cpu().numpy()
        features = []
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len - 1]

            # Save the embeddings to a file
            with open(os.path.join(destfolder, name + '.data'), 'w') as f:
                np.savetxt(f, seq_emd, delimiter=',', fmt='%s')

        torch.cuda.empty_cache()


# Main execution block
if __name__ == "__main__":
    # Load ProtBERT tokenizer and model (using a BERT-based model)
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model = model.eval()

    trainfiles1 = ['D:/fengzhen/OPEF-MsL-main/DataSet/SMB/SMB_Train.txt']
    testfiles1 = ['D:/fengzhen/OPEF-MsL-main/DataSet/SMB/SMB_Test.txt']

    for item in trainfiles1:
        print(item)
        extratdata(item, 'D:/fengzhen/1embedding/protbert_embedding_SMB/', tokenizer, model, device)

    for item in testfiles1:
        print(item)
        extratdata(item, 'D:/fengzhen/1embedding/protbert_embedding_SMB/', tokenizer, model, device)

    print('----finish-------')
