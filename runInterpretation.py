import pickle

import torch
import torch_geometric

from datasetMouseBrain import MouseBrainDataset
from ExplanationEvaluation.explainers.GNNExplainer import GNNExplainer
from gcnmodel import GCN


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ ==  '__main__':
    dataset = MouseBrainDataset("/gpfs/data/rsingh47/hzaki1/data")

    labels = torch.zeros(len(dataset), 1)#torch.zeros(len(dataset), 1)
    features = torch.zeros(len(dataset), dataset[0].x.shape[0], 1)
    graphs = torch.zeros(len(dataset), 2, dataset[0].edge_index.shape[1])
    for index, data in enumerate(dataset):
        labels[index] = data.y
        features[index] = torch.unsqueeze(data.x, 1).type(torch.FloatTensor)
        graphs[index] = data.edge_index
    
    model = GCN(hidden_channels=128, data=dataset, output_size=7)
    model.load_state_dict(torch.load('model_weightsjul22_max_pool.pth', map_location=torch.device('cpu')))

    task = 'graph'

    explainer = GNNExplainer(model, graphs, features, task)

    geneDict = {}
    for key in dataset.cellToIndex.keys():
        geneDict[key] = {}
    
    for i in range (0, len(dataset), 5):
        cellType = dataset.indexToCell[dataset[i].y[0].item()]
        print(cellType)
        graph, expl = explainer.explain(i)
        print(i)
        indiciesOfTopExplainers = torch.topk(expl, 40)[1]
        for ind in indiciesOfTopExplainers:
            geneInd = int(graph[0][ind].item())
            geneInd1 = int(graph[1][ind].item())
            gene = dataset.indexToGene[int(graph[0][geneInd].item())]
            gene1 = dataset.indexToGene[int(graph[0][geneInd1].item())]
            if gene in geneDict[cellType]:
                geneDict[cellType][gene] +=1
            else:
                geneDict[cellType][gene] = 1
            if gene1 in geneDict[cellType]:
                geneDict[cellType][gene1] +=1
            else:
                geneDict[cellType][gene1] = 1
    print(geneDict)
    save_obj(geneDict, "InterpretationDict_both")
