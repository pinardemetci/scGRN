import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

"""
NOTE: This file was replaced by dataset_featurizer.py
but is kept to illustrate how to build a custom dataset in PyG.
"""
DATA_FOLDER="/gpfs/data/rsingh47/hzaki1/atacseqdataChromatin/"
RESOURCES_FOLDER="/gpfs/data/rsingh47/hzaki1/atacseqdataChromatin/resources"
METADATA_FNAME = os.path.join(RESOURCES_FOLDER, 'SNAREseq_CellMixture_types.txt')
MM_TFS_FNAME = os.path.join(RESOURCES_FOLDER, 'mm_mgi_tfs.txt')
SC_EXP_FNAME = os.path.join(RESOURCES_FOLDER, "GSE126074_CellLineMixture_SNAREseq_cDNA_counts.tsv")
COUNTS_FNAME = os.path.join(RESOURCES_FOLDER, "GSE126074_CellLineMixture_SNAREseq_chromatin_counts.tsv")

def save_obj(obj, name):
    with open('/gpfs/data/rsingh47/hzaki1/atacseqdataChromatin/obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('/gpfs/data/rsingh47/hzaki1/atacseqdataChromatin/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

class AtacSeqChromatinDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.exmatrix = pd.read_csv(SC_EXP_FNAME, sep='\t', header=0, index_col=0).T
        
        self.metadata = pd.read_csv(METADATA_FNAME, sep='\t') 
        self.counts = pd.read_csv(COUNTS_FNAME, sep='\t').T

        self.root = root
        self.adj = self.getAdj()

        geneList = self.exmatrix.columns.values.tolist()
        self.geneToIndex = {}
        for i, gene in enumerate(geneList):
            self.geneToIndex[gene] = i
        self.indexToGene = {y:x for x,y in self.geneToIndex.items()}

        self.cellToIndex = {}
        for i, cell in enumerate(self.metadata['CellType'].unique()):
            self.cellToIndex[cell] = i
        self.indexToCell = {y:x for x,y in self.cellToIndex.items()}
        
        self.exmatrix = self.exmatrix.apply(np.log1p)
        self.maxi = self.exmatrix.to_numpy().max()
        self.exmatrix = self.exmatrix.apply(self.normalize)
        
        self.counts = self.counts.where(self.counts == 0, 1) #binarize
        toDelete = []
        for index, row in self.counts.iterrows(): #remove values with more than 100 or less than 5
            tot = row.sum()
            if tot < 5 or tot > 105:
                toDelete.append(index)
        
        self.counts.drop(toDelete)
            

#         self.counts = self.counts.apply(np.log1p)
#         self.maxi = self.counts.to_numpy().max()
#         self.counts = self.counts.apply(self.normalize)

        super(AtacSeqChromatinDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return 'adjacencies.tsv'

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(len(list(self.metadata["CellType"])))]

    def download(self):
        pass

    def process(self):
        counter = 0
        regionToGene = load_obj('regionToGene_human10')
        lengthOfNodes = len(self.exmatrix.columns)

        for index, row in tqdm(self.counts.iterrows(), total=self.counts.shape[0]):
            # Get node features
            node_feats = torch.zeros((lengthOfNodes,2))
            for gene in list(regionToGene.keys()):
                val = 0
                for region in regionToGene[gene]:
                    val += self.counts[region][row.name]
                    node_feats[self.geneToIndex[gene]] = val
                node_feats[self.geneToIndex[gene]][0] = val
            node_feats[:, 1] = torch.from_numpy(np.array(self.exmatrix.loc[index].values))
            # Get adjacency info
            edge_index = self.adj
            # Get labels info
            #print(self.metadata.loc[self.metadata['Barcode'] == row[0]]['CellType'].values)
            label = self.cellToIndex[self.metadata.loc[self.metadata['Barcode'] == row.name]['CellType'].values[0]]
            labelTensor = self._get_labels(label)
            # Create data object
            data = Data(x=node_feats,
                        edge_index=edge_index,
                        y=labelTensor,
                        )
            torch.save(data,
                       os.path.join(self.processed_dir,
                                    f'data_{counter}.pt'))
            counter+=1
            
    def normalize(self, input):
        return input/self.maxi

    def getAdj(self):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        adjacency = pd.read_csv(os.path.join(self.root,'raw/adjacencies.tsv'), sep='\t')
        geneList = self.exmatrix.columns.values.tolist()
        geneToIndex = {}
        for i, gene in enumerate(geneList):
            geneToIndex[gene] = i
        self.filteredDF = adjacency[(adjacency['importance'] > (adjacency['importance'].mean() + adjacency['importance'].std()))]
        counts = self.filteredDF.count().values[0]
        adjacencyMatrix = np.zeros((2, counts))
        for index, row in tqdm(self.filteredDF.iterrows(), total=counts):
            adjacencyMatrix[0][index] = geneToIndex[row['TF']]
            adjacencyMatrix[1][index] = geneToIndex[row['target']]
        return torch.from_numpy(adjacencyMatrix).type(torch.LongTensor)

    def _get_edge_features(self, mol):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())
            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, mol):
        """
        We could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features
        """
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.metadata.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data
