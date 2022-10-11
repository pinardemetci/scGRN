import torch
from torch.nn import Linear, Softmax
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv, GatedGraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, data, output_size):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(data.num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_size)
        self.embedding_size = hidden_channels

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None: # No batch given
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
        # 1. Obtain node embeddings 
        #edge_index = edge_index.type(torch.cuda.LongTensor)
        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weights)

        # 2. Readout layer
        x = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)
        
        return x
    
    def embedding(self, x, edge_index, edge_weights=None):
        # 1. Obtain node embeddings 
        edge_index = edge_index.type(torch.LongTensor)
        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weights)
        
        return x
