import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, args, dataset, user_feat_mat, item_feat_mat):
        super(BaseModel, self).__init__()
        self.args = args
        self.user_feat_mat = user_feat_mat.to(args.device)
        self.item_feat_mat = item_feat_mat.to(args.device)

        # load dataset info
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num

        # load parameters
        self.embedding_size = args.embedding_size
    
    def calculate_loss(self, interaction):
        return NotImplementedError
    
    def predict(self, user, item):
        return NotImplementedError

