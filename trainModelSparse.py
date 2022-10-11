import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch_geometric
from sklearn.manifold import TSNE
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch_geometric.data import Data, DataLoader, Dataset
from torch_spline_conv import spline_conv
from tqdm import tqdm

from datasets.datasetMouseBrainSparse import MouseBrainDataset
from gcnmodel_sparseAttention import GCN_Sparse


def train():
    model.train()
    avgLoss = 0
    for data in tqdm(train_loader, total=81):  # Iterate in batches over the training dataset.
        data.x = torch.reshape(data.x, (data.x.shape[0], 1))
        data.x = data.x.type(torch.FloatTensor)
        data = data.to(device)
        out = model(data.x, data.edge_index.squeeze(), data.batch)# Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        avgLoss += loss
    return avgLoss / 81

def test(loader):
    model.eval()

    correct = 0
    avgAUC = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data.x = torch.reshape(data.x, (data.x.shape[0], 1))
        data.x = data.x.type(torch.FloatTensor)
        data = data.to(device)
        out = model(data.x, data.edge_index.squeeze(), data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        avgAUCbatch = 0
        aucCounter = 0
        one_hot = np.eye(7)[data.y.cpu()]
        for i in range(0,7):
            try:
                roc = roc_auc_score(one_hot[:,i], out.cpu().detach().numpy()[:,i])
                avgAUCbatch += roc
                aucCounter +=1
            except Exception as e:
                continue
        if aucCounter == 0:
            continue
        avgAUC += avgAUCbatch/aucCounter
    return correct / len(loader.dataset), avgAUC / len(loader)  # Derive ratio of correct predictions.

if __name__ == '__main__':
    dataset = MouseBrainDataset("/gpfs/data/rsingh47/hzaki1/data-attention")

    torch.manual_seed(12345)
    dataset = dataset.shuffle()

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
    model = GCN_Sparse(hidden_channels=128, data=dataset, output_size=7).to(device)

    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 200):
        loss = train()
        train_acc, trainAUC = test(train_loader)
        test_acc,testAUC = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train AUC: {trainAUC:.4f}, Train AUPR: {trainAUPR:.4f}, Test Acc: {test_acc:.4f}, Test Auc: {testAUC:.4f}, Test AUPR: {testAUPR:.4f},  Loss: {loss:.4f}')

        torch.save(model.state_dict(), 'model_weightsMar3Sparse.pth')
