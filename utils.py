import random
import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset,DataLoader

def load_data(data_path):
    data = pd.read_csv(data_path, header=0, sep=',')
    num_users = len(data['user'].unique())
    num_items = len(data['item'].unique())
    print('Number of users:{} Number of users:{} '.format(num_users, num_items))
    return num_users, num_items
def load_train(train_path, num_users, num_items):
    '''
    :param file: data path
    :return: total number of users, total number of items, interactions set [[u1,i1], ..., [um,in]], and a boolean tenosr (|U|*|I|) indicating whether the user has interacted with item.
    '''
    pos_dict = dict()
    neg_dict = dict()
    data = pd.read_csv(train_path, header=0, sep=',')
    train_tensor = torch.zeros(num_users, num_items)
    datapair = []
    for i in data.itertuples():
        user, item = int(getattr(i, 'user')), int(getattr(i, 'item'))
        datapair.append([user,item])
        pos_dict.setdefault(user, list())
        pos_dict[user].append(item)
        train_tensor[user, item] = 1
    item_set = {i for i in range(num_items)}
    for user in pos_dict.keys():
        neg_item = list(item_set - set(pos_dict[user]))
        neg_dict[user] = neg_item
    print('Number of Interactions for Training:{}'.format(len(datapair)))
    return pos_dict, neg_dict, datapair, train_tensor

def load_test(test_path, num_users, num_items):
    '''
    :param test_path:
    :param num_users:
    :param num_items:
    :return: bool tensor of test data [|U| * |I|]
    '''
    data = pd.read_csv(test_path, header=0, sep=',')
    test_tensor = torch.zeros(num_users, num_items)
    for i in data.itertuples():
        user, item = int(getattr(i, 'user')), int(getattr(i, 'item'))
        test_tensor[user, item] = 1
    return test_tensor.bool()


class MyData(Dataset):
    def __init__(self, datapair, num_users, num_items, pos_dict, neg_dict, args):
        self.data = datapair
        self.num_users = num_users
        self.num_items = num_items
        self.pos_dict = pos_dict
        self.neg_dict = neg_dict
        self.args = args

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def collate_fn(self,batch):
        new_data = []
        for entry in batch:
            u,i = entry[0],entry[1]
            extra_pos = random.choices(self.pos_dict[u], k= self.args.M-1 )
            neg = random.sample(self.neg_dict[u], k= self.args.N)
            data_entry = [u] + [i] + extra_pos + neg
            new_data.append(data_entry)
        return torch.tensor(new_data)

'''
### A mini-batch data is orgnized as:  #[bs, 1+M+N]

[[u1, pos1,...,posM, neg1,...,negN],
 [u2, pos1,...,posM, neg1,...,negN],
           ...
 [uBS, pos1,...,posM,neg1,...,negN]]
'''

# def convert_spmat_to_sptensor(X):
#     coo = X.tocoo().astype(np.float32)
#     row = torch.Tensor(coo.row).long()
#     col = torch.Tensor(coo.col).long()
#     index = torch.stack([row, col])
#     data = torch.FloatTensor(coo.data)
#     return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))