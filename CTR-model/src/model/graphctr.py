from numpy import argsort
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

from layers import LightGCNLayer, StackGCN, FactorizationMachine
from utils import get_norm_adj_mat
from .basemodel import BaseModel


class GraphCTR1(BaseModel):
    def __init__(self, args, dataset, inter_mat, user_feat_mat, item_feat_mat):
        super(GraphCTR1, self).__init__(args, dataset, user_feat_mat, item_feat_mat)

        # load parameters
        self.n_layers = args.n_layers
        self.dropout = args.dropout

        # define layer
        self.lightgcn = LightGCNLayer(self.n_users, self.n_items, self.embedding_size)

        self.user_GCNLayer = StackGCN(user_feat_mat.shape[1], self.embedding_size, self.n_layers)
        self.item_GCNLayer = StackGCN(item_feat_mat.shape[1], self.embedding_size, self.n_layers)

        self.u_attention = nn.Parameter(torch.empty(size=(self.n_users, 1)))
        self.i_attention = nn.Parameter(torch.empty(size=(self.n_items, 1)))

        # get adj matrix
        self.adj, self.adj_u, self.adj_i= get_norm_adj_mat(inter_mat, self.n_users, self.n_items)
        self.adj = self.adj.to(self.args.device)
        self.adj_u = self.adj_u.to(self.args.device)
        self.adj_i = self.adj_i.to(self.args.device)

        # define loss
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

        # parameters initialization
        self._init_weights()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

    def _init_weights(self):
        xavier_normal_(self.u_attention.data)
        xavier_normal_(self.i_attention.data)

    def forward(self):
        user_lightgcn_embeddings, item_lightgcn_embeddings = self.lightgcn(self.n_layers, self.adj)
        user_feature_embeddings = self.user_GCNLayer(self.user_feat_mat, self.adj_u, dropout=self.dropout)
        item_feature_embeddings = self.item_GCNLayer(self.item_feat_mat, self.adj_i, dropout=self.dropout)

        user_embeddings = self.u_attention * user_lightgcn_embeddings + (1 - self.u_attention) * user_feature_embeddings
        item_embeddings = self.i_attention * item_lightgcn_embeddings + (1 - self.i_attention) * item_feature_embeddings

        return user_embeddings, item_embeddings
    
    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        user, item, label = interaction[0], interaction[1], interaction[2]
        user = user.to(self.args.device)
        item = item.to(self.args.device)
        label = label.to(self.args.device)

        # get enhanced embedding
        user_embeddings, item_embeddings = self.forward()
        u_embeddings = user_embeddings[user]
        i_embeddings = item_embeddings[item]

        # get loss
        logits = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        output = self.sigmoid(logits)
        loss = self.loss(output, label)
        return loss

    def predict(self, user, item):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_item_e[item]

        output = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return output


class GraphCTR2(BaseModel):
    def __init__(self, args, dataset, inter_mat, user_feat_mat, item_feat_mat):
        super(GraphCTR2, self).__init__(args, dataset, user_feat_mat, item_feat_mat)

        # load parameters
        self.n_layers = args.n_layers
        self.dropout = args.dropout

        # define layer
        self.user_GCNLayer = StackGCN(user_feat_mat.shape[1], self.embedding_size, self.n_layers)
        self.item_GCNLayer = StackGCN(item_feat_mat.shape[1], self.embedding_size, self.n_layers)

        self.u_attention = nn.Parameter(torch.empty(size=(self.n_users, 1)))
        self.i_attention = nn.Parameter(torch.empty(size=(self.n_items, 1)))

        self.FM_layer = FactorizationMachine(self.n_users, 
                                             self.n_items, 
                                             u_input_features=user_feat_mat.shape[1],
                                             i_input_features=item_feat_mat.shape[1],
                                             embedding_size=self.embedding_size)

        # get adj matrix
        self.adj, self.adj_u, self.adj_i= get_norm_adj_mat(inter_mat, self.n_users, self.n_items)
        self.adj = self.adj.to(self.args.device)
        self.adj_u = self.adj_u.to(self.args.device)
        self.adj_i = self.adj_i.to(self.args.device)

        # define loss
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

        # parameters initialization
        self._init_weights()

        # storage variables for full sort evaluation acceleration
        self.restore_user_fm = None
        self.restore_item_fm = None
        self.restore_user_gcn = None
        self.restore_item_gcn = None

    def _init_weights(self):
        xavier_normal_(self.u_attention.data)
        xavier_normal_(self.i_attention.data)
    
    def get_embedding(self, user, item):
        # first_fm_embedding： batch_size * 4
        first_fm_embedding = self.FM_layer.get_first_embedding(user, item, self.user_feat_mat, self.item_feat_mat)
        # second_fm_embedding： batch_size * 4 * embedding_size
        second_fm_embedding = self.FM_layer.get_second_embedding(user, item, self.user_feat_mat, self.item_feat_mat)

        # mlp_input_embedding: batch_size * (4 * embedding_size)
        batch_size = second_fm_embedding.shape[0]
        input_embedding = second_fm_embedding.view(batch_size, -1)
        
        # split interaction and feature embedding
        user_inter_embedding = input_embedding[:, :self.embedding_size]
        item_inter_embedding = input_embedding[:, self.embedding_size: 2 * self.embedding_size]
        user_feat_embedding = input_embedding[:, 2 * self.embedding_size: 3 * self.embedding_size]
        item_feat_embedding = input_embedding[:, 3 * self.embedding_size:]

        user_lightgcn_embeddings, item_lightgcn_embeddings = self.get_lightgcn_embedding(user_inter_embedding, item_inter_embedding)
        user_feature_embeddings = self.user_GCNLayer(user_feat_embedding, self.adj_u, dropout=self.dropout)
        item_feature_embeddings = self.item_GCNLayer(item_feat_embedding, self.adj_i, dropout=self.dropout)

        user_embeddings = self.u_attention * user_lightgcn_embeddings + (1 - self.u_attention) * user_feature_embeddings
        item_embeddings = self.i_attention * item_lightgcn_embeddings + (1 - self.i_attention) * item_feature_embeddings
        return first_fm_embedding, second_fm_embedding, user_embeddings, item_embeddings

    def get_lightgcn_embedding(self, user_inter_embedding, item_inter_embedding):
        all_embeddings = torch.cat([user_inter_embedding, item_inter_embedding], dim=0)
        embeddings_list = [all_embeddings]
        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_lightgcn_embeddings, item_lightgcn_embeddings = torch.split(lightgcn_all_embeddings,
                                                               [self.n_users, self.n_items])
        return user_lightgcn_embeddings, item_lightgcn_embeddings
    
    def forward(self, user, item):
        first_fm_embedding, second_fm_embedding, user_embeddings, item_embeddings = self.get_embedding(user, item)

        # fm output: batch_size * 1
        y_fm = self.FM_layer(first_fm_embedding, second_fm_embedding)

        u_embeddings = user_embeddings[user]
        i_embeddings = item_embeddings[item]

        # get loss
        logits = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        y_gcn = self.sigmoid(logits)

        y = self.sigmoid(y_fm + y_gcn)
        return y.squeeze(-1)

    def calculate_loss(self, interaction):
        # if self.restore_user_gcn is not None or self.restore_item_gcn is not None or \
        #     self.restore_user_fm is not None or self.restore_item_fm is not None:
        #     self.restore_user_gcn, self.restore_item_gcn = None, None
        #     self.restore_user_fm, self.restore_item_fm = None, None

        user, item, label = interaction[0], interaction[1], interaction[2]
        user = user.to(self.args.device)
        item = item.to(self.args.device)
        label = label.to(self.args.device)

        output = self.forward(user, item)
        loss = self.loss(output, label)
        return loss

    def predict(self, user, item):
        user = user.to(self.args.device)
        item = item.to(self.args.device)
        return self.forward(user, item)
