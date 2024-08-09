#-*- coding: utf-8 -*-
import argparse
import os
import datetime
import random
import torch
import numpy as np
from torch import sparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
import model
import evaluation
print(torch.__version__)
USE_CUDA = torch.cuda.is_available()

device = torch.device('cuda' if USE_CUDA else 'cpu')

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def attention(query, key, value, scaling_factor):
    '''
    :param query: [bs*dim]
    :param key:   [bs* M *dim]
    :param value: [bs* M *dim]
    :return:      [bs*dim]
    '''
    scores = torch.sum(query.unsqueeze(1)* key,dim=-1)                       # [bs*M]
    attention_weights = torch.softmax(scores/scaling_factor, dim=1)          # [bs*M]
    attended_values = torch.sum(value*attention_weights.unsqueeze(-1),dim=1) # [bs*dim]
    return attended_values
def cretirion(batch):
    '''
    :param batch: A mini batch of data, for BPR  batch.shape = [bs*3], for VBPR batch.shape = [bs*(1+M+N)]
    :return:
    '''
    if args.loss == 'BPR':
        user_idx, item_idx = batch[:, 0], batch[:, 1:]
        user_embs, item_embs = model.forward(user_idx, item_idx) # [bs,dim], [bs,2,dim]
        pos_item_embs, neg_item_embs = item_embs[:, 0, :], item_embs[:, 1, :]  # [bs,dim], [bs, dim]

        xui = torch.sum(user_embs * pos_item_embs,dim=-1) #[bs,]
        xuj = torch.sum(user_embs * neg_item_embs,dim=-1) #[bs,]
        BPR = -torch.log(torch.sigmoid(xui-xuj)).mean()
        return BPR



    elif args.loss == 'VBPR':
        user_idx, item_idx = batch[:,0], batch[:,1:]               #[bs,],  [bs,M+N]
        user_embs, item_embs = model.forward(user_idx, item_idx)   #[bs,dim],  [bs,M+N, dim]
        pos_item_embs, neg_item_embs = item_embs[:, 0:args.M, :], item_embs[:, args.M:, :] #[bs,M, dim],  [bs, N, dim]

        pos_prototype = attention(user_embs,pos_item_embs,pos_item_embs,args.cpos)
        neg_prototype = attention(user_embs,neg_item_embs,neg_item_embs,args.cneg)
        xui = torch.sum(user_embs * pos_prototype, dim=-1)  # [bs,]
        xuj = torch.sum(user_embs * neg_prototype, dim=-1)  # [bs,]
        VBPR = -torch.log(torch.sigmoid(xui - xuj)).mean()
        return VBPR



def train(model, data_loader, train_optimizer):
    model.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for batch in train_bar:
        batch = batch.to(device, non_blocking=True)
        loss = cretirion(batch)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += args.batch_size
        total_loss += loss.item() * args.batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, args.epochs, total_loss / total_num))
    return total_loss / total_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--root', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='1m', help='Dataset name, choose from [100k,1m, gowalla, yelp2018]')

    parser.add_argument('--backbone', default='LightGCN', type=str, help='Backbone model, choose from [MF, LightGCN]')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--topk', default=20, type=int, help='Top-k ranking list for evaluation')
    parser.add_argument('--feature_dim', default=64, type=int, help='Feature dim for latent vector')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='Weight_decay')
    parser.add_argument('--hop', default=1, type=int, help='Hop')

    parser.add_argument('--loss', default='BPR', type=str, help='Choose Loss Function, choose from [BPR, Variational-BPR]')
    parser.add_argument('--M', default=2, type=int, help='Number of positive samples')
    parser.add_argument('--N', default=4, type=int, help='Number of negative samples')
    parser.add_argument('--cpos', default=10, type=float, help='Positive scalling factor')
    parser.add_argument('--cneg', default=0.5, type=float, help='Negitive scalling factor')
    init_seed(2024)
    args = parser.parse_args()
    print(args)

    # Data Prepare
    data_path = args.root + '/' + args.dataset + '.txt'
    train_path = args.root + '/' + args.dataset + '_train.txt'
    test_path = args.root + '/' + args.dataset + '_test.txt'

    num_users, num_items = utils.load_data(data_path)
    pos_items, neg_items, datapair, train_tensor = utils.load_train(train_path,num_users, num_items)

    train_data = utils.MyData(datapair, num_users, num_items, pos_items, neg_items, args)
    train_loader = DataLoader(train_data, collate_fn=train_data.collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    test_tensor = utils.load_test(test_path, num_users, num_items)

    # Model Setup
    if args.backbone == 'MF':
        model = model.MF(num_users, num_items, args.feature_dim).to(device)
    elif args.backbone == 'LightGCN':
        G_Lap_tensor = utils.convert_spmat_to_sptensor(train_data.Lap_mat).to(device)
        G_Adj_tensor = utils.convert_spmat_to_sptensor(train_data.Adj_mat).to(device)
        model = model.LightGCN(num_users, num_items, args.feature_dim, G_Lap_tensor, G_Adj_tensor, args)
        model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Optimizer Config
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    results = []
    for epoch in range(1, args.epochs + 1):
        # Model Training
        train_loss = train(model, train_loader, optimizer)

        # Model Evaluation
        if epoch % 5 == 0:
            model.eval()
            rating_matrix = model.predict() - train_tensor.to(device) * 1000 # erase the training data for evaluation
            result = evaluation.topk_eval(rating_matrix, args.topk, test_tensor.to(device))
            results.append(result)
            print('Evaluation Epoch[{}/{}]: [Precision@{}: {:.4f}, Recall@{}: {:.4f}, F1@{}: {:.4f}, NDCG@{}: {:.4f}]'.format(epoch,args.epochs,args.topk, result[0],args.topk, result[1],args.topk, result[2],args.topk, result[3]))
            print('\n')
    best_result,best_epoch = torch.tensor(results).max(dim=0)
    print('Best Results: Precision:[{:.4f}/Epoch{}], Recall:[{:.4f}/Epoch{}], F1:[{:.4f}/Epoch{}], NDCG:[{:.4f}/Epoch{}]'.format(best_result[0],(best_epoch[0]+1)*5,best_result[1],(best_epoch[1]+1)*5,best_result[2],(best_epoch[2]+1)*5,best_result[3],(best_epoch[3]+1)*5))

