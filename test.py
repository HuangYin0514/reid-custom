import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from utils import util
from dataloader import getDataLoader
from models import build_model

import metrics

# ---------------------- Extract features ----------------------


def _parse_data_for_eval(data):
    imgs = data[0]
    pids = data[1]
    camids = data[2]
    return imgs, pids, camids


def _extract_features(model, input):
    model.eval()
    return model(input)


# ---------------------- Start testing ----------------------
@torch.no_grad()
def test(model, dataset, dataset_path, batch_size, device, args, normalize_feature=False,
         dist_metric='cosine'):
    model.eval()

    # test dataloader------------------------------------------------------------
    query_dataloader = getDataLoader(dataset, batch_size, dataset_path, 'query',   args, shuffle=False, augment=False)
    gallery_dataloader = getDataLoader(dataset, batch_size, dataset_path, 'gallery', args, shuffle=False, augment=False)

    # image information------------------------------------------------------------
    print('Extracting features from query set ...')
    qf, q_pids, q_camids = [], [], []  # query features, query person IDs and query camera IDs
    q_score = []
    for batch_idx, data in enumerate(query_dataloader):
        imgs, pids, camids = _parse_data_for_eval(data)
        imgs = imgs.to(device)
        features = _extract_features(model, imgs)
        qf.append(features)
        q_pids.extend(pids)
        q_camids.extend(camids)
    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

    print('Extracting features from gallery set ...')
    gf, g_pids, g_camids = [], [], []  # gallery features, gallery person IDs and gallery camera IDs
    g_score = []
    for batch_idx, data in enumerate(gallery_dataloader):
        imgs, pids, camids = _parse_data_for_eval(data)
        imgs = imgs.to(device)
        features = _extract_features(model, imgs)
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
    gf = torch.cat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    if normalize_feature:
        print('Normalzing features with L2 norm ...')
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)

    print('Computing distance matrix with metric={} ...'.format(dist_metric))
    distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
    distmat = distmat.cpu().numpy()

    print('Computing CMC and mAP ...')
    cmc, mAP = metrics.evaluate_rank(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        use_metric_cuhk03=False
    )

    return cmc, mAP


if __name__ == "__main__":
    # -----------------------------parameters setting --------------------------------
    parser = argparse.ArgumentParser(description='Testing arguments')

    parser.add_argument('--experiment', type=str, default='PCB_p6')
    parser.add_argument('--save_path', type=str, default='./experiments')
    parser.add_argument('--which_epoch', default='final', type=str, help='0,1,2,3...or final')
    parser.add_argument('--checkpoint', type=str, default='/home/hy/vscode/reid-custom/experiments/Market1501')

    parser.add_argument('--dataset', type=str, default='Occluded_REID')
    parser.add_argument('--dataset_path', type=str, default='/home/hy/vscode/reid-custom/data/Occluded_REID')
    parser.add_argument('--height', type=int, default=384, help='height of the input image')
    parser.add_argument('--width', type=int, default=128, help='width of the input image')

    parser.add_argument('--batch_size', default=3, type=int, help='batchsize')
    parser.add_argument('--share_conv', default=False, action='store_true')

    args = parser.parse_args()

    # devie---------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model------------------------------------------------------------------------------------
    model = build_model(args.experiment, num_classes=1, share_conv=args.share_conv)
    model = util.load_network(model, args.checkpoint, args.which_epoch)
    model = model.to(device)

    # save_dir_path-----------------------------------------------------------------------------------
    save_dir_path = os.path.join(args.save_path, args.dataset)
    os.makedirs(save_dir_path, exist_ok=True)

    # logger------------------------------------------------------------------------------------
    logger = util.Logger(save_dir_path)
    logger.info(vars(args))

    # test -----------------------------------------------------------------------------------
    CMC, mAP = test(model, args.dataset, args.dataset_path, args.batch_size, device, args)
    logger.info('Testing: top1:%.4f top5:%.4f top10:%.4f mAP:%.4f' % (CMC[0], CMC[4], CMC[9], mAP))

    # torch.cuda.empty_cache()
