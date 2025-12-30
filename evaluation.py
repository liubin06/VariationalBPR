import math
import torch

@torch.no_grad()
def topk_eval(score, k, test_tensor, long_tail_mask):
    dev = score.device
    test_tensor = test_tensor.to(dev)
    long_tail_mask = long_tail_mask.to(dev)

    U, I = score.shape
    topk = torch.topk(score, k=k, dim=1).indices              # [U,k]
    hits = test_tensor.gather(1, topk).float()                # [U,k] 命中(0/1)

    label_count = test_tensor.sum(dim=1).float()              # [U]
    user_mask = label_count > 0
    denom = user_mask.sum().item()
    if denom == 0:
        return [0.0] * 5

    hit_cnt = hits.sum(dim=1)                                 # [U]

    # precision / recall / f1（只对有效用户平均）
    pre = (hit_cnt[user_mask] / k).mean()
    recall = (hit_cnt[user_mask] / (label_count[user_mask] + 1e-8)).mean()
    f1 = (2 * hit_cnt[user_mask] / (k + label_count[user_mask] + 1e-8)).mean()

    # ndcg
    discount = (1.0 / torch.log2(torch.arange(2, k + 2, device=dev).float()))  # [k]
    dcg = (hits * discount).sum(dim=1)                                         # [U]

    # IDCG: 用折扣前缀和 + clamp(min(label,k))
    ideal_len = torch.clamp(label_count.long(), max=k)                          # [U]
    cumsum_discount = discount.cumsum(dim=0)                                    # [k]
    idcg = torch.zeros(U, device=dev)
    nonzero = ideal_len > 0
    idcg[nonzero] = cumsum_discount[ideal_len[nonzero] - 1]
    ndcg = (dcg[user_mask] / (idcg[user_mask] + 1e-8)).mean()

    # APLT@k：top-k 中长尾比例
    aplt = long_tail_mask[topk].float().mean(dim=1)[user_mask].mean()
    return [pre.item(), recall.item(), f1.item(), ndcg.item(), aplt.item()]
