import torch
import sys
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros, reset
from torch_geometric.utils import softmax, add_remaining_self_loops
from torch_scatter import scatter_add
from torch_sparse import fill_diag, sum, mul
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

class SparsityLearning(torch.nn.Module):
    def __init__(self, input_dim, sparse, emb_type='1_hop', threshold=0.5, temperature=0.5):
        super(SparsityLearning, self).__init__()
        self.att = Parameter(torch.Tensor(1, input_dim * 2))
        glorot(self.att.data)
        self.emb_layer = int(emb_type.split('_')[0]) - 1
        if self.emb_layer > 0:
            self.gnns = torch.nn.ModuleList()
            for i in range(self.emb_layer):
                self.gnns.append(Sparse_GCN(input_dim, input_dim))

        self.threshold = threshold
        self.temperature = temperature
        self.sparse = sparse

    def forward(self, x, edge_index, edge_weight, edge_mask, layer):

        raw_edges = edge_weight[edge_weight==1].shape[0]
        _, edge_mask = add_remaining_self_loops(edge_index, edge_mask, -1, x.size(0))
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0, x.size(0))
        '''
            embedding node using emb_type
        '''
        if self.emb_layer > 0:
            for i in range(self.emb_layer):
                x = torch.nn.functional.relu(self.gnns[i](x, edge_index[:, edge_weight!=0], edge_weight[edge_weight!=0]))

        weights = (torch.cat([x[edge_index[0]], x[edge_index[1]]], 1) * self.att).sum(-1)
        weights = torch.nn.functional.leaky_relu(weights) + edge_weight

        row, col = edge_index

        col, col_id = col.sort()
        weights = weights[col_id]
        row = row[col_id]
        edge_index = torch.stack([row, col])
        edge_weight = edge_weight[col_id]

        edge_mask = edge_mask[col_id]

        if self.sparse == 'soft':
            edge_weight = softmax(weights, col)
        elif self.sparse == 'hard':
            weights = softmax(weights, col)
            sample_prob = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(self.temperature, probs=weights)
            y = sample_prob.rsample()
            y_soft = y
            y_hard = (y>self.threshold).to(y.dtype)
            y = (y_hard - y).detach() + y
            intra_edges = y[edge_weight == 1]
            inter_edges = y[edge_weight == 0]
            edge_weight[edge_weight==0] = inter_edges
            edge_mask[(edge_mask==0) & (y==1)] = layer+1
            intra_soft_edge = y_soft[edge_mask==-1]
        else:
            print('sparse operation is not found...')

        assert edge_index.size(1) == edge_weight.size(0)
        torch.cuda.empty_cache()

        return edge_index, edge_weight, y_soft, edge_mask, intra_soft_edge


class SparseGraph(torch.nn.Module):
    def __init__(self, input_dim, output_dim, func, dropout=0, act=torch.nn.ReLU(), bias=True, num_layer=2,  threshold=0.5, temperature=0.5):
        super(SparseGraph, self).__init__()
        self.dropout = dropout
        self.act = act
        self.num_layer = num_layer
        self.func = func
        self.threshold = threshold
        self.temperature = temperature
        self.weight = Parameter(torch.Tensor(input_dim, output_dim))
        if bias:
            self.bias = Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.convs = torch.nn.ModuleList()
        self.sls = torch.nn.ModuleList()
        for _ in range(self.num_layer):
            if 'lin' in self.func:
                self.Lin = True
            else:
                self.Lin = False
            self.convs.append(Sparse_GCN(output_dim, output_dim, Lin=self.Lin))
            if 'attn' in self.func:
                sparse = ''
                if 'soft' in self.func:
                    sparse = 'soft'
                elif 'gumbel' in self.func:
                    sparse = 'hard'
                self.sls.append(SparsityLearning(output_dim, sparse, threshold=self.threshold, temperature=self.temperature))
                torch.cuda.empty_cache()
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, input, **kwargs):
        x = input
        batch = kwargs['batch']
        edge_index = kwargs['edge_index']

        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.act(torch.matmul(x, self.weight) + self.bias)
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)

        if 'attn' in self.func:
            edge_mask = kwargs['edge_mask']
            # regard self as intra nodes! --> -1
            edge_weight = torch.ones((edge_mask.size(0), ), dtype=torch.float, device=edge_mask.device)
            edge_weight[edge_mask!=-1] = 0

            _, edge_mask = add_remaining_self_loops(edge_index, edge_mask, -1, x.size(0))
            edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0, x.size(0))

            raw_edge_intra_weight = torch.ones((edge_weight[edge_mask==-1].size(0), ), dtype=edge_weight.dtype, device=edge_mask.device)
            raw_edge_intra_weight = softmax(raw_edge_intra_weight, edge_index[0][edge_mask==-1])

            raw_size = edge_weight[edge_weight!=0].size(0)
            self.kl_terms = []
            soft_weights = []
            inter_edge_indexs = []
            edge_masks = []
            for i in range(self.num_layer):
                x = self.act(self.convs[i](x, edge_index[:, edge_weight!=0], edge_weight=edge_weight[edge_weight!=0]))
                if i != self.num_layer-1:
                    edge_index, edge_weight, soft_weight, edge_mask, intra_soft_edge = self.sls[i](x, edge_index, edge_weight, edge_mask, layer=i)
                    soft_weights.append(soft_weight[edge_mask==i+1])
                    inter_edge_indexs.append(edge_index[:, edge_mask==i+1])
                    edge_masks.append(edge_mask[edge_mask==i+1])

                if 'reg' in self.func:
                    assert intra_soft_edge.size() == raw_edge_intra_weight.size()
                    log_p = torch.log(raw_edge_intra_weight + 1e-12)
                    log_q = torch.log(intra_soft_edge+ 1e-12)
                    self.kl_terms.append(torch.mean(raw_edge_intra_weight * (log_p - log_q)))

            if 'explain' in kwargs:
                soft_weights = torch.cat(soft_weights)
                inter_edge_indexs = torch.cat(inter_edge_indexs, 1)
                edge_masks = torch.cat(edge_masks)
                assert inter_edge_indexs.shape[1] == soft_weights.shape[0] == edge_masks.shape[0]
                exp_dict = {'edge_index': inter_edge_indexs, 'edge_weight': soft_weights, 'edge_mask': edge_masks}
                return x, exp_dict
            else:
                return x

        else:
            # gnn
            for i in range(self.num_layer):
                x = self.act(self.convs[i](x, edge_index))
            return x


class Dense(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0, act=torch.nn.ReLU(), bias=False):
        super(Dense, self).__init__()
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.Tensor(input_dim, output_dim))
        if bias:
            self.bias = Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)


    def forward(self, input, **kwargs):
        x = input
        # dropout
        x = torch.nn.functional.dropout(x, 0.7, training=self.training)
        # dense encode
        x = self.act(torch.matmul(x, self.weight) + self.bias)

        return x

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, func, aggr, dropout=0, act=torch.nn.ReLU()):
        super(MLP, self).__init__()

        self.func = func
        self.dropout = dropout
        self.act = act
        self.aggr = aggr
        self.emb_weight = Parameter(torch.Tensor(input_dim, input_dim))
        self.emb_bias = Parameter(torch.Tensor(input_dim))


        self.mlp_weight = Parameter(torch.Tensor(input_dim, output_dim))
        self.mlp_bias = Parameter(torch.Tensor(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.emb_weight)
        glorot(self.mlp_weight)
        zeros(self.emb_bias)
        zeros(self.mlp_bias)

    def global_pooling(self, x, batch):
        x_emb = self.act(torch.matmul(x, self.emb_weight)+self.emb_bias)
        # global pooling
        if self.aggr == 'sum':
            x = global_add_pool(x_emb, batch)
        elif self.aggr == 'mean':
            x = global_mean_pool(x_emb, batch)
        elif self.aggr == 'max':
            x = global_max_pool(x_emb, batch)

        return x

    def forward(self, input=None, **kwargs):
        x = input
        batch = kwargs['batch']
        x = self.global_pooling(x, batch)
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.matmul(x, self.mlp_weight) + self.mlp_bias
        return x


class Sparse_GCN(MessagePassing):
    def __init__(self, in_channels, out_channels, score=False, improved=False, cached=False,
                 bias=False, normalize=True, Lin=False, **kwargs):
        super(Sparse_GCN, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.Lin = Lin

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight1 = Parameter(torch.Tensor(in_channels, out_channels))


        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if score:
            self.score = Parameter(torch.Tensor(in_channels, 1))
        else:
            self.register_parameter('score', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.weight1)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)
        if self.score is not None:
            edge_weight = torch.matmul(x, self.score).squeeze()

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        if self.Lin:
            return self.propagate(edge_index, x=x, norm=norm) + torch.matmul(x,self.weight1)
        else:
            return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)