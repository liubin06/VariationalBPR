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

    def forward(self, user_idx, item_idx):
        user_embs = self.User_Emb.weight[user_idx]
        item_embs = self.Item_Emb.weight[item_idx]
        return user_embs, item_embs

    def predict(self):
        return torch.mm(self.User_Emb.weight, self.Item_Emb.weight.t())


class LightGCNAgg(nn.Module):
    def __init__(self, hidden_size):
        super(LightGCNAgg, self).__init__()
        self.dim = hidden_size

    def forward(self, A, x):
        '''
            A: n \times n
            x: n \times d
        '''
        return torch.sparse.mm(A, x)

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, dim,  g_laplace, g_adj, args):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.g_laplace = g_laplace
        self.g_adj = g_adj
        self.args = args
        self.hop = args.hop

        self.User_Emb = nn.Embedding(self.num_users, self.dim)
        nn.init.xavier_normal_(self.User_Emb.weight)
        self.Item_Emb = nn.Embedding(self.num_items, self.dim)
        nn.init.xavier_normal_(self.Item_Emb.weight)

        # LightGCN Agg
        self.global_agg = []
        for i in range(self.hop):
            agg = LightGCNAgg(self.dim)
            self.add_module('Agg_LightGCN_{}'.format(i), agg)
            self.global_agg.append(agg)

    def computer(self):
        users_emb = self.User_Emb.weight
        items_emb = self.Item_Emb.weight
        all_emb = torch.cat((users_emb, items_emb), dim=0)
        embs = [all_emb]
        for i in range(self.hop):
            aggregator = self.global_agg[i]
            x = aggregator(A=self.g_laplace, x=embs[i])
            embs.append(x)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def forward(self, user_ids, item_ids):
        # Fetch Emb
        all_users_emb, all_items_emb = self.computer()
        user_embs = all_users_emb[user_ids]
        item_embs = all_items_emb[item_ids]
        return user_embs, item_embs

    def predict(self):
        all_users_emb, all_items_emb = self.computer()      # |U| * d, |V| * d
        rating_matrix = torch.mm(all_users_emb, all_items_emb.t())
        return rating_matrix

