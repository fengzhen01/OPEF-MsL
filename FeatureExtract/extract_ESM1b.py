import torch
import esm
import numpy as np
import os


def read_data_file_trip(filename):
    with open(filename) as f:
        data = f.readlines()

    results = []
    block = len(data) // 2
    for index in range(block):
        item1 = data[index * 2 + 0].split()
        name = item1[0].strip()
        seq = item1[1].strip()
        results.append([name, seq])
    return results


def extratdata(file, destfolder):
    student_tuples = read_data_file_trip(file)
    for name, seq in student_tuples:
        print(f'Processing: {name}')
        data = [(name, seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # Move to GPU if available
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.to("cuda:0")

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33].cpu()

        for token_representation, tokens_len, batch_label in zip(token_representations, batch_lens, batch_labels):
            with open(os.path.join(destfolder, batch_label + '.data'), 'w') as f:
                # Remove BOS and EOS token representations
                np.savetxt(f, token_representation[1: tokens_len - 1], delimiter=',', fmt='%s')


if __name__ == "__main__":
    print('----Loading ESM-1b model-------')
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    if torch.cuda.is_available():
        model = model.to("cuda:0")

    print('----Preparing dataset------')

    trainfiles1 = ['D:/fengzhen/1NucGMTL-main/DataSet/SJC/SMB1_Train.txt']
    testfiles1 = ['D:/fengzhen/1NucGMTL-main/DataSet/SJC/SMB1_Test.txt']

    for item in trainfiles1:
        print(item)
        extratdata(item, 'D:/fengzhen/1embedding/ESM1b_embedding_SMB1/')

    for item in testfiles1:
        print(item)
        extratdata(item, 'D:/fengzhen/1embedding/ESM1b_embedding_SMB1/')

    print('----Finished-------')
