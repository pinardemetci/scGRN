import torch
import torch.nn.functional as F
from torch.nn import Linear, Softmax
# from torch_geometric.nn import GCNConv, SAGEConv, ChebConv, GatedGraphConv
from torch_geometric.nn import global_max_pool, global_mean_pool

from layers.spgraphattention import SpGraphAttentionLayer


class GCN_Sparse(torch.nn.Module):
    def __init__(self, hidden_channels, data, output_size):
        super(GCN_Sparse, self).__init__()
        torch.manual_seed(12345)
        self.attentions = [SpGraphAttentionLayer(1, 
                                                 64, 
                                                 dropout=.6, 
                                                 alpha=.2, 
                                                 concat=False) for _ in range(1)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
#         self.out_att = SpGraphAttentionLayer(128, 
#                                              7, 
#                                              dropout=.6, 
#                                              alpha=.2, 
#                                              concat=False)
       
        self.lin = Linear(64, output_size)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        # 1. Obtain node embeddings 
        #edge_index = edge_index.type(torch.cuda.LongTensor)
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(x, .6, training=self.training)
        #x = F.elu(self.out_att(x, edge_index))
        # 2. Readout layer
        x = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin(x)
        
        return x
    