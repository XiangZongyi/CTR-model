"""
Reference code:
    https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/deepfm.py
"""

from .basemodel import BaseModel
from layers import FactorizationMachine, MLPLayers

import torch
import torch.nn as nn


class DeepFM(BaseModel):
    def __init__(self, args, dataset, user_feat_mat, item_feat_mat):
        super(DeepFM, self).__init__(args, dataset, user_feat_mat, item_feat_mat)
        # load parameters
        self.n_layers = args.n_layers
        self.dropout = args.dropout

        # define layer
        self.FM_layer = FactorizationMachine(self.n_users, 
                                             self.n_items, 
                                             u_input_features=user_feat_mat.shape[1],
                                             i_input_features=item_feat_mat.shape[1],
                                             embedding_size=self.embedding_size)
        
        self.layers = self._create_mlp_layers()
        self.mlp_layer = MLPLayers(self.layers, self.dropout)

        # define loss
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
    
    def _create_mlp_layers(self):
        # (4 * emb_size - emb_size) // 2 + emb_size
        hidden_size = 3 * self.embedding_size - self.embedding_size // 2
        layers = [4 * self.embedding_size] + [hidden_size] * (self.n_layers - 2) + [self.embedding_size] + [1]
        return layers
    
    def forward(self, user, item):
        # fm output: batch_size * 1
        # first_fm_embedding： batch_size * 4
        first_fm_embedding = self.FM_layer.get_first_embedding(user, item, self.user_feat_mat, self.item_feat_mat)
        # second_fm_embedding： batch_size * 4 * embedding_size
        second_fm_embedding = self.FM_layer.get_second_embedding(user, item, self.user_feat_mat, self.item_feat_mat)
        y_fm = self.FM_layer(first_fm_embedding, second_fm_embedding)

        # mlp output: batch_size * 1
        batch_size = second_fm_embedding.shape[0]
        # mlp_input_embedding: batch_size * (4 * embedding_size)
        mlp_input_embedding = second_fm_embedding.view(batch_size, -1)
        y_mlp = self.mlp_layer(mlp_input_embedding).reshape(-1)

        y = self.sigmoid(y_fm + y_mlp)
        return y.squeeze(-1)
    
    def calculate_loss(self, interaction):
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
