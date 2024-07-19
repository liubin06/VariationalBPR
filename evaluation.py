import math
import torch
from cuda import *
# evaluation
mc = MemCache()
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


def get_rec_tensor(k, topn_rec_index, num_items):
    '''
    :param k: Top-k
    :param topn_rec_index: [|U|*k]recommended item id
    :param num_items: The total number of numbers
    :return:
    rec_tensor: [|U|*|I|] with 0/1 elements ,1 indicates the item is recommended to the user
    index_dim0:[|U|*k] dim0 index for slicing
    '''
    index_dim0 = torch.arange(topn_rec_index.shape[0]).to(device)
    index_dim0 = index_dim0.unsqueeze(-1).expand(topn_rec_index.shape[0], k)
    rec_tensor = torch.zeros(topn_rec_index.shape[0],num_items).to(device)
    rec_tensor[index_dim0, topn_rec_index] = 1
    return rec_tensor.bool()


def get_idcg(discountlist, test_count):
    idcg = torch.zeros(len(test_count)).to(device)
    label_count_list = test_count.tolist()
    for i in range(len(test_count)):
        idcg[i] = discountlist[0:int(label_count_list[i])].sum()
    return idcg


# Pure CPU eva
def topk_eval(score,  k, test_tensor):
    '''
    :param score: prediction
    :param k: number of top-k
    '''
    evaluation = [0, 0, 0, 0]
    topk_tensor = torch.topk(score, k=k, dim=1).indices
    rec_tensor = get_rec_tensor(k, topk_tensor, score.shape[1])


    hit_tensor = rec_tensor * test_tensor
    label_count = test_tensor.sum(dim=-1)
    discountlist = torch.tensor([1 / math.log(i + 1, 2) for i in range(1, k + 1)]).to(device)   # Discount list to calculate dcg
    dcg = ((hit_tensor).gather(1, topk_tensor) * discountlist).sum(dim=-1)
    idcg = get_idcg(discountlist, label_count) + 1e-8


    pre = ((hit_tensor).sum(dim=-1)).sum(dim=-1) / k
    recall = (((hit_tensor).sum(dim=-1)) / (label_count + 1e-8)).sum(dim=-1)
    f1 = (2 * ((hit_tensor).sum(dim=-1)) / label_count.add(k)).sum(dim=-1)
    ndcg = (dcg / idcg).sum(dim=-1)


    # mc.show_cuda_info()
    # evaluation[0], evaluation[1], evaluation[2], evaluation[3], evaluation[4], evaluation[5], evaluation[6], evaluation[7], evaluation[8], evaluation[9] = pre.item(), recall.item(), f1.item(), ndcg.item(), 1, 1, 1, 1, 1, 1
    evaluation[0], evaluation[1], evaluation[2], evaluation[3]= pre.item(), recall.item(), f1.item(), ndcg.item()
    return [i / torch.count_nonzero(label_count).item() for i in evaluation]


