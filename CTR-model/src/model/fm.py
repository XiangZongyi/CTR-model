"""
Reference code:
    https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/fm.py
"""

from .basemodel import BaseModel
from layers import FactorizationMachine

import torch.nn as nn
import torch


class FM(BaseModel):
    def __init__(self, args, dataset, user_feat_mat, item_feat_mat):
        super(FM, self).__init__(args, dataset, user_feat_mat, item_feat_mat)
        
        # define layer
        self.FM_layer = FactorizationMachine(self.n_users, 
                                             self.n_items, 
                                             u_input_features=user_feat_mat.shape[1],
                                             i_input_features=item_feat_mat.shape[1],
                                             embedding_size=self.embedding_size)
        # define loss
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        first_fm_embedding = self.FM_layer.get_first_embedding(user, item, self.user_feat_mat, self.item_feat_mat)
        second_fm_embedding = self.FM_layer.get_second_embedding(user, item, self.user_feat_mat, self.item_feat_mat)
        y = self.sigmoid(self.FM_layer(first_fm_embedding, second_fm_embedding))
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
