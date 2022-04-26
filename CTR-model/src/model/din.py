"""
Reference code:
    https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/din.py
"""

import torch
import torch.nn as nn

from .basemodel import BaseModel
from layers import MLPLayers


class DIN(BaseModel):
    def __init__(self, args, dataset, user_feat_mat, item_feat_mat):
        super(DIN, self).__init__(args, dataset, user_feat_mat, item_feat_mat)
        # load parameters
        self.n_layers = args.n_layers
        self.dropout = args.dropout

        # define layer
        self.layers = self._create_mlp_layers()
        self.dnn_predict_layers = MLPLayers()

        # define loss
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
    
    def _create_mlp_layers(self):
        # (4 * emb_size - emb_size) // 2 + emb_size
        hidden_size = 3 * self.embedding_size - self.embedding_size // 2
        layers = [4 * self.embedding_size] + [hidden_size] * (self.n_layers - 2) + [self.embedding_size] + [1]
        return layers

    def calculate_loss(self, interaction):
        user, item, label = interaction[0], interaction[1], interaction[2]
        user = user.to(self.args.device)
        item = item.to(self.args.device)
        label = label.to(self.args.device)
        return loss

    def predict(self, user, item):
        user = user.to(self.args.device)
        item = item.to(self.args.device)
        return scores
