import math
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, normal_
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    # @article{kipf2016semi,
    #   title={Semi-Supervised Classification with Graph Convolutional Networks},
    #   author={Kipf, Thomas N and Welling, Max},
    #   journal={arXiv preprint arXiv:1609.02907},
    #   year={2016}
    # }
    """
    Reference code:
        https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class FactorizationMachine(nn.Module):
    def __init__(self, n_users, n_items, u_input_features, i_input_features, embedding_size):
        super(FactorizationMachine, self).__init__()
        # define first fm embedding table and parameters
        self.first_user_embedding = nn.Embedding(n_users, 1)
        self.first_item_embedding = nn.Embedding(n_items, 1)
        self.user_W = nn.Parameter(torch.empty(size=(u_input_features, 1)))
        self.item_W = nn.Parameter(torch.empty(size=(i_input_features, 1)))

        # define second fm embedding table and parameters
        self.second_user_embedding = nn.Embedding(n_users, embedding_size)
        self.second_item_embedding = nn.Embedding(n_items, embedding_size)
        self.user_V = nn.Parameter(torch.empty(size=(u_input_features, embedding_size)))
        self.item_V = nn.Parameter(torch.empty(size=(i_input_features, embedding_size)))

        # define bias
        self.bias = nn.Parameter(torch.zeros(1, ))

        # parameters initialization
        self._init_weights()
    
    def _init_weights(self):
        xavier_normal_(self.first_user_embedding.weight.data)
        xavier_normal_(self.first_item_embedding.weight.data)
        xavier_normal_(self.user_W.data)
        xavier_normal_(self.item_W.data)

        xavier_normal_(self.second_user_embedding.weight.data)
        xavier_normal_(self.second_item_embedding.weight.data)
        xavier_normal_(self.user_V.data)
        xavier_normal_(self.item_V.data)

    def first_linear_layer(self, first_embeddings):
        """
        Args:
            first_embeddings: batch_size * (filed_num * embedding_size)
        """
        return torch.sum(first_embeddings, dim=1)  # batch_size
    
    def second_layer(self, second_embeddings):
        """
        Args:
            second_embeddings: batch_size * filed_num * embeding_size
        """
        square_of_sum = torch.sum(second_embeddings, dim=1) ** 2  # batch_size * embedding_size
        sum_of_square = torch.sum(second_embeddings ** 2, dim=1)  # batch_size * embedding_size
        output = square_of_sum - sum_of_square
        output = torch.sum(output, dim=1)  # batch_size
        output = 0.5 * output
        return output  

    def forward(self, first_embeddings, second_embeddings):
        return self.bias + self.first_linear_layer(first_embeddings) + self.second_layer(second_embeddings)
    
    def get_first_embedding(self, user, item, user_feat_mat, item_feat_mat):
        user_feat_embedding = torch.mm(user_feat_mat, self.user_W)  # batch_sizer * 1
        item_feat_embedding = torch.mm(item_feat_mat, self.item_W)  # batch_sizer * 1

        first_embedding = []
        first_embedding.append(self.first_user_embedding(user))
        first_embedding.append(self.first_item_embedding(item))
        first_embedding.append(user_feat_embedding[user])
        first_embedding.append(item_feat_embedding[item])
        return torch.cat(first_embedding, dim=1)  # batch_size * 4

    def get_second_embedding(self, user, item, user_feat_mat, item_feat_mat):
        user_feat_embedding = torch.mm(user_feat_mat, self.user_V)  # batch_sizer * emb_size
        item_feat_embedding = torch.mm(item_feat_mat, self.item_V)  # batch_sizer * emb_size
        second_embedding = torch.stack([self.second_user_embedding(user), 
                                       self.second_item_embedding(item),
                                       user_feat_embedding[user],
                                       item_feat_embedding[item]], dim=1)
        return second_embedding  # batch_size * 4 * emb_size 


class MLPLayers(nn.Module):
    """
    Reference code:
        https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/layers.py
    """
    def __init__(self, layers, dropout, use_bn=False):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.use_bn = use_bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            mlp_modules.append(nn.ReLU())

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)  
    

class LightGCNLayer(nn.Module):
    """
    Reference code:
        https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/general_recommender/lightgcn.py
    """
    def __init__(self, n_users, n_items, embedding_size):
        super(LightGCNLayer, self).__init__()
        # load parameters
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embedding_size

        # define layer
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Embedding):
            xavier_normal_(m.weight.data)

    def forward(self, n_layers, adj):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings_list = [all_embeddings]
        for layer_idx in range(n_layers):
            all_embeddings = torch.sparse.mm(adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_lightgcn_embeddings, item_lightgcn_embeddings = torch.split(lightgcn_all_embeddings,
                                                               [self.n_users, self.n_items])
        return user_lightgcn_embeddings, item_lightgcn_embeddings


class StackGCN(nn.Module):
    def __init__(self, input_shape, output_shape, n_layers):
        super(StackGCN, self).__init__()
        self.n_layers = n_layers
        hidden_shape = (input_shape + output_shape) // 2
        self.layers = [input_shape] + [hidden_shape] * (self.n_layers - 1) + [output_shape]

        self.GCNLayer = nn.Sequential()
        for i, (input_feat, output_feat) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.GCNLayer.add_module(f'GCN{i + 1}', GraphConvolution(input_feat, output_feat))
    
    def forward(self, input, adj, dropout):
        for i in range(self.n_layers - 1):
            input = self.GCNLayer[i](input, adj)
            input = F.relu(input)
            input = F.dropout(input, p=dropout, training=self.training)
        output = self.GCNLayer[i + 1](input, adj)
        return output
