# -*- coding: utf-8 -*-
import argparse
import os
import time
import random
import torch
import numpy as np
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def prior_plus(pos_scores, rarity, positem_id):
    hardness = torch.sigmoid((pos_scores.mean(dim=-1, keepdim=True) - pos_scores) / 0.1)  # [bs,M]
    batch_rarity = rarity[positem_id]  # [bs,M]
    quality = item_quality[positem_id]  # [bs,M]
    return batch_rarity ** args.pos_rarity * hardness ** args.pos_hardness * quality ** args.pos_quality  # [bs,M]


@torch.no_grad()
def prior_minus(neg_scores, pop, negitem_id):
    hardness = torch.sigmoid((neg_scores - neg_scores.mean(dim=-1, keepdim=True)) / 0.1)  # [bs,N]
    batch_pop = pop[negitem_id]  # [bs,N]
    quality_bad = (1 - item_quality)[negitem_id]  # [bs,N]
    return batch_pop ** args.neg_popularity * hardness ** args.neg_hardness * quality_bad ** args.neg_badquality  # [bs,N]


@torch.no_grad()
def VarInference(scores, prior, scaling_factor, eps=1e-12):
    """
    query: [bs, dim]
    key:   [bs, M, dim]
    prior: [bs, M]  (Normalization of the prior does not affect the variational distribution， since prior normalization merely adds a global constant to the logits, which is eliminated during softmax normalization and does not alter the relative ratios of the prior elements that dictate the output.)
    return: variational distribution [bs, M]
    """
    logits = scores / scaling_factor + torch.log(prior.clamp_min(eps))
    VarDist = torch.softmax(logits, dim=-1)  # [bs, M]
    return VarDist


def VarBPR(pos_score, neg_score, alpha, beta):
    '''
    :param
    :return:
    '''
    xui = torch.sum(pos_score * alpha, dim=-1)  # [bs,]
    xuj = torch.sum(neg_score * beta, dim=-1)  # [bs,]
    margin = xui - xuj  # [bs,]
    loss = torch.nn.functional.softplus(-margin).mean()
    return loss


def ELBO(pos_score, neg_score, alpha, beta):
    margin = (pos_score.unsqueeze(2) - neg_score.unsqueeze(1)).reshape(pos_score.size(0), -1)  # [bs, M*N]
    logistics = torch.nn.functional.softplus(-margin)  # [bs, M*N]
    dist = (alpha.unsqueeze(2) * beta.unsqueeze(1)).reshape(pos_score.size(0), -1)  # [bs, M*N]
    loss = torch.sum(logistics * dist, dim=-1).mean()
    return loss


def train(model, data_loader, train_optimizer, rarity, pop):
    model.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for batch in train_bar:
        batch = batch.to(device, non_blocking=True)
        user_id, item_id = batch[:, 0], batch[:, 1:]  # [bs,],  [bs,M+N]
        positem_id, negitem_id = item_id[:, 0:args.M], item_id[:, args.M:]
        user_embs, item_embs = model.forward(user_id, item_id)  # [bs,dim],  [bs,M+N, dim]
        score = torch.sum(user_embs.unsqueeze(1) * item_embs, dim=-1)  # [bs, M+N]
        pos_score, neg_score = score[:, 0:args.M], score[:, args.M:]  # [bs, M], [bs, N]
        if args.loss == 'BPR':
            assert args.M == 1
            assert args.N == 1
            loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
        else:
            # _________________________Encode quality/ popularity/hardness with prior______________________________
            pi_plus = prior_plus(pos_score, rarity, positem_id)  # [bs, M]
            pi_minus = prior_minus(neg_score, pop, negitem_id)  # [bs, N]

            # _________________________             Variational Inference            ______________________________
            alpha = VarInference(pos_score, pi_plus, args.cpos)  # [bs, M]
            beta = VarInference(-neg_score, pi_minus, args.cneg)  # [bs, N]

            # _________________________             Variational Learning             ______________________________
            if args.loss == 'VarBPR':
                loss = VarBPR(pos_score, neg_score, alpha, beta)

            elif args.loss == 'ELBO':
                loss = ELBO(pos_score, neg_score, alpha, beta)

            else:
                raise NotImplementedError
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += args.batch_size
        total_loss += loss.item() * args.batch_size
        train_bar.set_description(
            'Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, args.epochs, total_loss / total_num))
    return total_loss / total_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--root', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='100k',
                        help='Dataset name, choose from [100k,1m, gowalla, yelp2018]')

    parser.add_argument('--backbone', default='MF', type=str, help='Backbone model, choose from [MF, LightGCN]')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--topk', default=20, type=int, help='Top-k ranking list for evaluation')
    parser.add_argument('--feature_dim', default=64, type=int, help='Feature dim for latent vector')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='Weight_decay')
    parser.add_argument('--hop', default=1, type=int, help='Hop')

    parser.add_argument('--loss', default='VarBPR', type=str,
                        help='Choose Loss Function, choose from [BPR, VarBPR, ELBO]')
    parser.add_argument('--M', default=4, type=int, help='Number of positive samples')
    parser.add_argument('--N', default=4, type=int, help='Number of negative samples')
    parser.add_argument('--cpos', default=10, type=float, help='Positive scalling factor')
    parser.add_argument('--cneg', default=10, type=float, help='Negitive scalling factor')
    # build prior
    parser.add_argument('--pos_hardness', default=0, type=float, help='Hardness of positive samples')
    parser.add_argument('--pos_rarity', default=0, type=float, help='Rarity of positive samples')
    parser.add_argument('--pos_quality', default=1, type=float, help='Rarity of positive samples')
    parser.add_argument('--neg_hardness', default=0.5, type=float, help='Hardness of negative samples')
    parser.add_argument('--neg_popularity', default=0, type=float, help='Rarity of negative samples')
    parser.add_argument('--neg_badquality', default=0.5, type=float, help='Rarity of negative samples')

    init_seed(2024)
    args = parser.parse_args()
    print(args)

    # Data Prepare
    data_path = args.root + '/' + args.dataset + '.txt'
    train_path = args.root + '/' + args.dataset + '_train.txt'
    test_path = args.root + '/' + args.dataset + '_test.txt'

    num_users, num_items = utils.load_data(data_path)
    pos_items, neg_items, datapair, train_tensor, pop, quality = utils.load_train(train_path, num_users, num_items)

    ################################################# Long Tail mask for evaluation
    tail_ratio = 0.85
    long_tail_mask = torch.zeros(pop.numel(), dtype=torch.bool)
    long_tail_mask[torch.argsort(pop)[:int(pop.numel() * tail_ratio)]] = True
    #################################################

    item_pop = torch.log1p(pop) / torch.log1p(pop.max())
    item_rarity = (1.0 - item_pop).clamp_min(1e-6)
    item_pop, item_rarity = item_pop.to(device), item_rarity.to(device)

    quality = torch.sigmoid((quality - (quality.mean())) / 0.2)
    item_quality = quality.to(device)

    train_data = utils.MyData(datapair, num_users, num_items, pos_items, neg_items, args)
    train_loader = DataLoader(train_data, collate_fn=train_data.collate_fn, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    test_tensor = utils.load_test(test_path, num_users, num_items)

    # Model Setup
    if args.backbone == 'MF':
        model = model.MF(num_users, num_items, args.feature_dim).to(device)
    elif args.backbone == 'LightGCN':
        G_Lap_tensor = utils.convert_spmat_to_sptensor(train_data.Lap_mat).to(device)
        G_Adj_tensor = utils.convert_spmat_to_sptensor(train_data.Adj_mat).to(device)
        model = model.LightGCN(num_users, num_items, args.feature_dim, G_Lap_tensor, G_Adj_tensor, args)
        model = model.to(device)

    # Optimizer Config
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    results = []
    for epoch in range(1, args.epochs + 1):
        # Model Training
        start = time.time()
        train_loss = train(model, train_loader, optimizer, item_rarity, item_pop)
        end = time.time()

        # Model Evaluation
        if epoch % 5 == 0:
            print('\n')
            model.eval()
            rating_matrix = model.predict() - train_tensor.to(
                device) * 1000  # erase the training data from predicted rating matrix for evaluation
            result5 = evaluation.topk_eval(rating_matrix, 5, test_tensor.to(device), long_tail_mask=long_tail_mask)
            print('Evaluation Epoch[{}/{}]: [Precision@{}: {:.4f}, Recall@{}: {:.4f}, F1@{}: {:.4f}, NDCG@{}: {:.4f}, APLT@{}: {:.4f}]'.format(epoch, args.epochs, 5, result5[0], 5, result5[1], 5, result5[2], 5, result5[3], 5, result5[4]))
            result20 = evaluation.topk_eval(rating_matrix, 20, test_tensor.to(device), long_tail_mask=long_tail_mask)
            print('Evaluation Epoch[{}/{}]: [Precision@{}: {:.4f}, Recall@{}: {:.4f}, F1@{}: {:.4f}, NDCG@{}: {:.4f}, APLT@{}: {:.4f}]'.format(epoch, args.epochs, 20, result20[0], 20, result20[1], 20, result20[2], 20, result20[3], 20,result20[4]))
            results.append(result5 + result20)
            print('\n')
    best_result, best_epoch = torch.tensor(results).max(dim=0)
    print('Best Results@Top-5: Precision:[{:.4f}], Recall:[{:.4f}], F1:[{:.4f}], NDCG:[{:.4f}], APLT:[{:.4f}]]'.format(
        best_result[0], best_result[1], best_result[2], best_result[3], best_result[4]))
    print('Best Results@Top-20: Precision:[{:.4f}], Recall:[{:.4f}], F1:[{:.4f}], NDCG:[{:.4f}], APLT:[{:.4f}]]'.format(
        best_result[5], best_result[6], best_result[7], best_result[8], best_result[9]))
    print('Best Epoch:[{}]'.format(best_epoch * 5))
