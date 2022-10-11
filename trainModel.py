import os
import random
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch_geometric
from scipy.special import softmax
from sklearn.manifold import TSNE
from sklearn.metrics import (auc, precision_recall_curve, roc_auc_score,
                             roc_curve)
from torch_geometric.data import Data, DataLoader, Dataset
from tqdm import tqdm

from datasets.datasetMouseBrain import MouseBrainDataset
from gcnmodel import GCN


def test(loader, size):
    model.eval()
    output = np.zeros((len(loader), size))
    actual = np.zeros((len(loader), size))
    accuracy = 0
    for ind, data in enumerate(loader):  # Iterate in batches over the training/test dataset.
        data.x = torch.reshape(data.x, (data.x.shape[0], 1))
        data.x = data.x.type(torch.FloatTensor)
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        output[ind] = softmax(out.cpu().detach().numpy())
        actual[ind][data.y] = 1
        accuracy += int((out.argmax(dim=1) == data.y).sum())
    all_labels = list(dataset.cellToIndex.keys())
    actual = np.array(actual)
    precision = dict()
    recall = dict()
    averageAUROC = []
    averageAUPR = []
    for (idx, c_label) in enumerate(all_labels):
        
        fpr, tpr, thresholds = roc_curve(actual[:,idx].astype(int), output[:,idx])
        precision[idx], recall[idx], _ = precision_recall_curve(actual[:, idx],
                                                        output[:, idx])
        averageAUROC.append(auc(fpr, tpr))
        averageAUPR.append(round(auc(recall[idx], precision[idx]),4))

    return accuracy/len(loader.dataset), mean(averageAUROC), mean(averageAUPR)


def train():
    model.train()
    avgLoss = 0
    for data in tqdm(train_loader, total=81):  # Iterate in batches over the training dataset.
        data.x = torch.reshape(data.x, (data.x.shape[0], 1))
        data.x = data.x.type(torch.FloatTensor)
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)# Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        avgLoss += loss
    return avgLoss / 81

# def test(loader):
#     model.eval()

#     correct = 0
#     avgAUC = 0
#     for data in loader:  # Iterate in batches over the training/test dataset.
#         data.x = torch.reshape(data.x, (data.x.shape[0], 1))
#         data.x = data.x.type(torch.FloatTensor)
#         data = data.to(device)
#         out = model(data.x, data.edge_index, data.batch)  
#         pred = out.argmax(dim=1)  # Use the class with highest probability.
#         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
#         avgAUCbatch = 0
#         aucCounter = 0
#         one_hot = np.eye(7)[data.y.cpu()]
#         for i in range(0,7):
#             try:
#                 roc = roc_auc_score(one_hot[:,i], out.cpu().detach().numpy()[:,i])
#                 avgAUCbatch += roc
#                 aucCounter +=1
#             except Exception as e:
#                 continue
#         if aucCounter == 0:
#             continue
#         avgAUC += avgAUCbatch/aucCounter
#     return correct / len(loader.dataset), avgAUC / len(loader)  # Derive ratio of correct predictions.



if __name__ ==  '__main__':
    dataset = MouseBrainDataset("/gpfs/data/rsingh47/hzaki1/data")
    
    torch.manual_seed(12345)
    
    # x = list(range(0, len(dataset)))
    # random.shuffle(x)
    # train_dataset = dataset[x[0:2403]]
    # test_dataset = dataset[x[2403:]]
    # np.savetxt("shuffleIndex_MouseBrain_1.txt", x)
    
    shuffle_index = np.loadtxt('shuffle_indices/shuffleIndex_MouseBrain.txt')
    shuffle_index = shuffle_index.astype(np.int32)
    train_size, val_size = int(len(shuffle_index)* 0.8), int(len(shuffle_index)* 0.9)
    train_dataset = [dataset[i] for i in shuffle_index[0:train_size]]
    val_dataset = [dataset[i] for i in shuffle_index[train_size: val_size]]
    test_dataset =  [dataset[i] for i in shuffle_index[val_size:]]
    
    train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
    train_loader_testing = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GCN(hidden_channels=128, data=dataset, output_size=7).to(device)
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, 350):
        loss = train()
        train_acc, trainAUC, trainAUPR = test(train_loader_testing, 7)
        test_acc,testAUC, testAUPR = test(val_loader, 7)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train AUC: {trainAUC:.4f}, Train AUPR: {trainAUPR:.4f}, Test Acc: {test_acc:.4f}, Test Auc: {testAUC:.4f}, Test AUPR: {testAUPR:.4f},  Loss: {loss:.4f}')
    
    torch.save(model.state_dict(), 'model_weightsDec4LogwShuffleIndex_350Epochs_5Dropout_1.pth')
    
