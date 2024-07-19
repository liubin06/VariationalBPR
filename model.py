import sys
import torch
import torch.nn as nn


class MF(nn.Module):
    def __init__(self, num_users, num_items, dim):
        '''
        :param num_users: number of users
        :param num_items: number of items
        :param dim: latent dimension
        '''
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.User_Emb = nn.Embedding(self.num_users, self.dim)
        nn.init.normal_(self.User_Emb.weight,mean=0.0, std=1e-2)
        self.Item_Emb = nn.Embedding(self.num_items, self.dim)
        nn.init.normal_(self.Item_Emb.weight,mean=0.0, std=1e-2)

    def forward(self, users, positives, negatives):
        user_embs = self.User_Emb.weight[users]           # bs * d
        pos_item_embs = self.Item_Emb.weight[positives]   # bs * M * d
        neg_item_embs = self.Item_Emb.weight[negatives]   # bs * N * d
        return user_embs, pos_item_embs, neg_item_embs

    def predict(self):
        return torch.mm(self.User_Emb.weight, self.Item_Emb.weight.t())
