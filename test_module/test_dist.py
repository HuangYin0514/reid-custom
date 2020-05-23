import sys
sys.path.append('/home/hy/vscode/reid-custom')

from metrics import rank
from metrics import distance
import torch
import numpy as np


print(sys.path)

if __name__ == "__main__":

    inp1 = torch.randn(3, 100)
    inp2 = torch.randn(4, 100)
    inp1 = np.array(inp1)
    inp2 = np.array(inp2)
    dist = distance.cosine_dist(inp1, inp2)
    rank_results = np.argsort(dist)[:, ::-1]

    query_camids = np.array([1, 2, 3])
    query_pids = np.array([1, 2, 3, 4])

    gallery_camids = np.array([4, 5, 6])
    gallery_pids = np.array([1, 2, 3, 4])

    APs, CMC = [], []
    for idx, data in enumerate(zip(rank_results, query_camids, query_pids)):
        a_rank, query_camid, query_pid = data
        ap, cmc = rank.compute_AP(a_rank, query_camid, query_pid, gallery_camids, gallery_pids)
        APs.append(ap), CMC.append(cmc)

    '''compute CMC and mAP'''
    MAP = np.array(APs).mean()
    min_len = min([len(cmc) for cmc in CMC])
    CMC = [cmc[:min_len] for cmc in CMC]
    CMC = np.mean(np.array(CMC), axis=0)

    print('complete check!')

