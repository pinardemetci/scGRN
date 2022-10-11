import pickle

import torch
import torch_geometric
from torch_geometric.nn import GNNExplainer

from datasetmuraro import MuraroDataset
from gcnmodel import GCN


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ ==  '__main__':
    dataset = MuraroDataset("/gpfs/data/rsingh47/hzaki1/data-muraro")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GCN(hidden_channels=128, data=dataset, output_size=9).to(device)
    model.load_state_dict(torch.load('model_weightsNov26_muraro_fixedOutputSize.pth'))

    explainer = GNNExplainer(model, epochs=200, return_type='log_prob', feat_mask_type='individual_feature')

    geneDict = {}
    for key in dataset.cellToIndex.keys():
        geneDict[key] = {}
    
    for i in range (0, len(dataset)):
        cellType = dataset.indexToCell[dataset[i].y[0].item()]

        feats = dataset[i].x.to(device)
        edgeIndex = dataset[i].edge_index.to(device)

        feats = torch.reshape(feats, (feats.shape[0], 1))

        feats = feats.type(torch.cuda.FloatTensor)

        node_feat_mask, edge_mask = explainer.explain_graph(feats, edgeIndex)

        indiciesOfTopExplainers = torch.topk(node_feat_mask, 20)[1]
        for ind in indiciesOfTopExplainers:
            gene = dataset.indexToGene[int(ind.item())]
            if gene in geneDict[cellType]:
                geneDict[cellType][gene] +=1
            else:
                geneDict[cellType][gene] = 1
    print(geneDict)
    save_obj(geneDict, "InterpretationDictNov27_muraro")
