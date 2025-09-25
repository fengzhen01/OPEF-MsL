# main.py
import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
import torch.multiprocessing
import datetime
from focalLoss import FocalLoss
from model import Module
from data_loader import read_data_file_trip, BioinformaticsDataset, coll_paddding


def train(modelstoreapl, device):
    model = Module(1024 + 1536, True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 30

    per_cls_weights = torch.FloatTensor([0.15, 0.85]).to(device)
    fcloss = FocalLoss_v2(alpha=per_cls_weights, gamma=2)

    model.train()

    file = read_data_file_trip(True)

    train_set = BioinformaticsDataset(file)
    train_loader = DataLoader(dataset=train_set, batch_size=16, pin_memory=True,
                              persistent_workers=True, shuffle=True, num_workers=16,
                              collate_fn=coll_paddding)

    best_val_loss = 3000
    best_epo = 0
    patience = 5
    counter = 0

    for epo in range(epochs):
        epoch_loss_train = 0
        nb_train = 0
        for prot_xs, data_ys, taskids, lengths in train_loader:
            outputs = model(prot_xs.to(device), lengths.to(device))
            data_ys = data_ys.to(device)
            lengths = lengths.to('cpu')

            outputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=True)
            data_ys = torch.nn.utils.rnn.pack_padded_sequence(data_ys, lengths, batch_first=True)

            loss = fcloss(outputs.data, data_ys.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_train = epoch_loss_train + loss.item()
            nb_train += 1

        epoch_loss_avg = epoch_loss_train / nb_train
        print('epo ', epo, ' epoch_loss_avg,', epoch_loss_avg)

        if best_val_loss > epoch_loss_avg:
            model_fn = modelstoreapl
            torch.save(model.state_dict(), model_fn)
            best_val_loss = epoch_loss_avg
            best_epo = epo
            if epo % 10 == 0:
                print('epo ', epo, " Save model, best_val_loss: ", best_val_loss)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    print('best loss,', best_val_loss, 'best epo,', best_epo)


def test(modelstoreapl, device):
    model = Module(1024 + 1536, False)
    model = model.to(device)
    model.load_state_dict(torch.load(modelstoreapl))
    model.eval()

    file = read_data_file_trip(False)
    test_set = BioinformaticsDataset(file)
    test_load = DataLoader(dataset=test_set, batch_size=32,
                           num_workers=16, pin_memory=True, persistent_workers=True,
                           collate_fn=coll_paddding)

    print("==========================Test RESULT================================")

    predicted_probs = []
    labels_actual = []
    labels_predicted = []

    with torch.no_grad():
        for prot_xs, data_ys, taskids, lengths in test_load:
            outputs = model(prot_xs.to(device), lengths.to(device))
            outputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, lengths.to('cpu'), batch_first=True)
            data_ys = torch.nn.utils.rnn.pack_padded_sequence(data_ys, lengths, batch_first=True)

            pred_probs = torch.nn.functional.softmax(outputs.data, dim=1)
            pred_probs = pred_probs.to('cpu')

            predicted_probs.extend(pred_probs[:, 1])
            labels_actual.extend(data_ys.data)
            labels_predicted.extend(torch.argmax(pred_probs, dim=1))

    sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult('SMB', labels_actual,
                                                                             predicted_probs, labels_predicted)
    tmresult = {'SMB': [sensitivity, specificity, acc, precision, mcc, auc, aupr_1]}

    return tmresult


def printresult(ligand, actual_label, predict_prob, predict_label):
    print('\n---------', ligand, '-------------')
    auc = metrics.roc_auc_score(actual_label, predict_prob)
    precision_1, recall_1, threshold_1 = metrics.precision_recall_curve(actual_label, predict_prob)
    aupr_1 = metrics.auc(recall_1, precision_1)
    acc = metrics.accuracy_score(actual_label, predict_label)
    print('acc ', acc)
    print('balanced_accuracy ', metrics.balanced_accuracy_score(actual_label, predict_label))
    tn, fp, fn, tp = metrics.confusion_matrix(actual_label, predict_label).ravel()
    print('tn, fp, fn, tp ', tn, fp, fn, tp)
    mcc = metrics.matthews_corrcoef(actual_label, predict_label)
    print('MCC ', mcc)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1score = 2 * precision * recall / (precision + recall)
    youden = sensitivity + specificity - 1
    print('sensitivity ', sensitivity)
    print('specificity ', specificity)
    print('precision ', precision)
    print('recall ', recall)
    print('f1score ', f1score)
    print('youden ', youden)
    print('auc', auc)
    print('AUPR ', aupr_1)
    print('---------------END------------')
    return sensitivity, specificity, acc, precision, mcc, auc, aupr_1


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    cuda = torch.cuda.is_available()
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")

    tasks = ['SMB']

    circle = 5
    a = str(datetime.datetime.now())
    a = a.replace(':', '_')

    totalkv = {task: [] for task in tasks}
    storename = '_'.join(p for p in tasks)
    print(storename)

    for i in range(circle):
        storeapl = 'RSMB/Result_' + storename + '_' + str(i) + '_' + a + '.pkl'
        train(storeapl, device) 
        tmresult = test(storeapl, device)  
        for task in tasks:
            totalkv[task].append(tmresult[task])
        torch.cuda.empty_cache()

    with open('RSMB/Result_' + storename + '_' + a + '.txt', 'w') as f:
        for nuc in tasks:
            np.savetxt(f, totalkv[nuc], delimiter=',', footer='Above is  record ' + SMB, fmt='%s')
            m = np.mean(totalkv[nuc], axis=0)
            np.savetxt(f, [m], delimiter=',', footer='----------Above is AVG -------' + SMB, fmt='%s')
