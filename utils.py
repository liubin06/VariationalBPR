import random
import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader


def load_data(data_path):
    data = pd.read_csv(data_path, header=0, sep='\t', encoding='utf-8')
    num_users = len(data['user'].unique())
    num_items = len(data['item'].unique())
    print('Number of users:{} Number of users:{} '.format(num_users, num_items))
    return num_users, num_items


def load_train(train_path, num_users, num_items):
    '''
    :param file: data path
    :return: interacted items, un-interacted items, interactions [[u1,i1], ..., [um,in]], and a boolean tenosr (|U|*|I|) indicating whether the user has interacted with item.
    '''
    item_set = {i for i in range(num_items)}
    pos_items = dict()
    neg_items = dict()
    data = pd.read_csv(train_path, header=0, sep='\t', encoding='utf-8')
    train_tensor = torch.zeros(num_users, num_items)
    datapair = []
    quality = np.zeros(num_items)
    for i in data.itertuples():
        user, item, rating = int(getattr(i, 'user')), int(getattr(i, 'item')), int(getattr(i,'rating'))
        datapair.append([user, item])
        pos_items.setdefault(user, list())
        pos_items[user].append(item)
        train_tensor[user, item] = 1
        quality[item] += rating
    for user in pos_items.keys():
        neg_items[user] = list(item_set - set(pos_items[user]))
    print('Number of Interactions for Training:{}'.format(len(datapair)))
    pop = torch.sum(train_tensor, dim=0)+1
    quality = quality/pop
    return pos_items, neg_items, datapair, train_tensor, pop, quality


def load_test(test_path, num_users, num_items):
    '''
    :param test_path:
    :param num_users:
    :param num_items:
    :return: bool tensor of test data [|U| * |I|]
    '''
    data = pd.read_csv(test_path, header=0,sep='\t', encoding='utf-8')
    test_tensor = torch.zeros(num_users, num_items)
    for i in data.itertuples():
        user, item = int(getattr(i, 'user')), int(getattr(i, 'item'))
        test_tensor[user, item] = 1
    print('Number of Interactions for Testing:{}'.format((test_tensor).sum().item()))
    return test_tensor.bool()

class MyData(Dataset):
    def __init__(self, datapair, num_users, num_items, pos_items, neg_items, args):
        self.data = datapair
        self.num_users = num_users
        self.num_items = num_items
        self.pos_items = pos_items
        self.neg_items = neg_items
        self.args = args

        if args.backbone == 'LightGCN':
            self.train_user = [pair[0] for pair in self.data]
            self.train_item = [pair[1] for pair in self.data]
            self.UserItemNet = csr_matrix((np.ones(len(self.train_user)), (self.train_user, self.train_item)),
                                          shape=(self.num_users, self.num_items))
            self.Lap_mat, self.Adj_mat = self.build_graph()

    def build_graph(self):
        print('building graph adjacency matrix')
        st = time.time()
        adj_mat = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items),
                                dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.UserItemNet.tolil()
        adj_mat[:self.num_users, self.num_users:] = R
        adj_mat[self.num_users:, :self.num_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))

        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.

        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        end = time.time()
        print(f"costing {end - st}s, obtained norm_mat...")
        return norm_adj, adj_mat

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        if self.args.loss == 'BPR':
            new_data = []
            for entry in batch:
                u, i = entry[0], entry[1]
                neg = random.sample(self.neg_items[u], k=1)
                data_entry = [u] + [i] + neg
                new_data.append(data_entry)
            return torch.tensor(new_data)

        else:
            new_data = []
            for entry in batch:
                u, i = entry[0], entry[1]
                extra_pos = random.choices(self.pos_items[u], k=self.args.M - 1)
                neg = random.sample(self.neg_items[u], k=self.args.N)
                data_entry = [u] + [i] + extra_pos + neg
                new_data.append(data_entry)
            return torch.tensor(new_data)



'''
### For Variational BPR, a mini-batch data is orgnized as follows:  #[bs, 1+M+N]

[[u1, pos1,...,posM, neg1,...,negN],
 [u2, pos1,...,posM, neg1,...,negN],
           ...
 [uBS, pos1,...,posM,neg1,...,negN]]
'''


def convert_spmat_to_sptensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
