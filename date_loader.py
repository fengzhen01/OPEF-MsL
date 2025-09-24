# data_loader.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def read_data_file_trip(istraining):
    """Read SMB data files"""
    if istraining:
        file_path = 'D:/fengzhen/OPEF-MsL-main/DataSet/SMB/SMB_Train.txt'
    else:
        file_path = 'D:/fengzhen/OPEF-MsL-main/DataSet/SMB/SMB_Test.txt'

    with open(file_path, 'r') as f:
        data = f.readlines()

    results = []
    block = len(data) // 2
    for index in range(block):
        item = data[index * 2 + 0].split()
        name = item[0].strip()
        results.append(name)

    return results


def coll_paddding(batch_traindata):
    batch_traindata.sort(key=lambda data: len(data[0]), reverse=True)
    feature_plms = []
    train_y = []
    task_ids = []

    for data in batch_traindata:
        feature_plms.append(data[0])
        train_y.append(data[1])
        task_ids.append(data[2])
    data_length = [len(data) for data in feature_plms]

    feature_plms = torch.nn.utils.rnn.pad_sequence(feature_plms, batch_first=True, padding_value=0)
    train_y = torch.nn.utils.rnn.pad_sequence(train_y, batch_first=True, padding_value=0)
    task_ids = torch.nn.utils.rnn.pad_sequence(task_ids, batch_first=True, padding_value=0)
    return feature_plms, train_y, task_ids, torch.tensor(data_length)


class BioinformaticsDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __getitem__(self, index):
        filename = self.X[index]
        # esm_embedding1280 prot_embedding  esm_embedding2560 msa_embedding
        df0 = pd.read_csv('D:/fengzhen/1embedding/prostT5_embedding_SMB/' + filename + '.data', header=None)
        prot0 = df0.values.astype(float).tolist()
        prot0 = torch.tensor(prot0)

        df1 = pd.read_csv('D:/fengzhen/1embedding/Ankh_embedding_SMB/' + filename + '.data', header=None)
        prot1 = df1.values.astype(float).tolist()
        prot1 = torch.tensor(prot1)

        min_len = min(prot0.size(0), prot1.size(0))
        prot = torch.cat((prot0[:min_len], prot1[:min_len]), dim=1)

        df2 = pd.read_csv('D:/fengzhen/1embedding/label/' + filename + '.label', header=None)
        label = df2.values.astype(int).tolist()
        label = torch.tensor(label)
        # reduce 2D to 1D
        label = torch.squeeze(label)
        task_id_label = torch.ones(prot.shape[0], dtype=int) * 0

        return prot, label, task_id_label

    def __len__(self):
        return len(self.X)
