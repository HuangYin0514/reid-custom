import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from dataloader import *
from model import *
import utils
from test import test


# ---------------------- Settings ----------------------
parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--experiment', type=str, default='PCB_p6')
parser.add_argument('--save_path', type=str, default='./experiments')
parser.add_argument('--dataset', type=str, default='market1501',
                    choices=['market1501', 'cuhk03', 'duke'])
parser.add_argument('--dataset_path', type=str,
                    default='/home/hy/vscode/pcb_custom/datasets/Market1501')
parser.add_argument('--batch_size', default=64,
                    type=int, help='batch_size')
parser.add_argument('--learning_rate', default=0.1, type=float,
                    help='FC params learning rate')
parser.add_argument('--epochs', default=60, type=int,
                    help='The number of epochs to train')
parser.add_argument('--share_conv', default=False, action='store_true')
parser.add_argument('--stripes', type=int, default=6)
args = parser.parse_args()

print(args)

# Fix random seed
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

# Make saving directory
save_dir_path = os.path.join(args.save_path, args.dataset)
os.makedirs(save_dir_path, exist_ok=True)


# ---------------------- Train function ----------------------
# Schedule learning rate
def adjust_lr(epoch):
    step_size = 40
    lr = args.learning_rate * (0.1 ** (epoch // step_size))
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)


def train(model, criterion, optimizer, scheduler, dataloader, num_epochs, device):
    '''
        train
    '''
    start_time = time.time()

    # Logger instance
    logger = utils.Logger(save_dir_path)
    logger.info('-' * 10)
    logger.info(vars(args))

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

        model.train()
        adjust_lr(epoch)

        # Training
        running_loss = 0.0
        batch_num = 0
        for inputs, labels in dataloader:
            batch_num += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # with torch.set_grad_enabled(True):
            outputs = model(inputs)

            # Sum up the stripe softmax loss
            loss = 0
            if isinstance(outputs, (list,)):
                for logits in outputs:
                    stripe_loss = criterion(logits, labels)
                    loss += stripe_loss
            elif isinstance(outputs,(torch.Tensor,)):
                loss = criterion(outputs, labels)
            else:
                raise Exception('outputs type is error !')

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset.imgs)
        logger.info('Training Loss: {:.4f}'.format(epoch_loss))

        # Save result to logger
        logger.x_epoch_loss.append(epoch + 1)
        logger.y_train_loss.append(epoch_loss)

        if (epoch + 1) % 20 == 0 or epoch + 1 == num_epochs:
            # Testing / Validating
            torch.cuda.empty_cache()
            CMC, mAP = test(model, args.dataset, args.dataset_path, 512)
            logger.info('Testing: top1:%.2f top5:%.2f top10:%.2f mAP:%.2f' %
                        (CMC[0], CMC[4], CMC[9], mAP))

            logger.x_epoch_test.append(epoch + 1)
            logger.y_test['top1'].append(CMC[0])
            logger.y_test['mAP'].append(mAP)
            if epoch + 1 != num_epochs:
                utils.save_network(model, save_dir_path, str(epoch + 1))
        logger.info('-' * 10)

    # Save the loss curve
    logger.save_curve()

    time_elapsed = time.time() - start_time
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Save final model weights
    utils.save_network(model, save_dir_path, 'final')


if __name__ == "__main__":

    train_dataloader = getDataLoader(
        args.dataset, args.batch_size, args.dataset_path, 'train', shuffle=True, augment=True)

    model = build_model(args.experiment, num_classes=len(train_dataloader.dataset.classes),
                        share_conv=args.share_conv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    base_param_ids = set(map(id, model.backbone.parameters()))
    new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
    param_groups = [{'params': model.backbone.parameters(), 'lr_mult': 0.1},
                    {'params': new_params, 'lr_mult': 1.0}]
    optimizer = torch.optim.SGD(param_groups, lr=args.learning_rate, momentum=0.9, weight_decay=5e-4,
                                nesterov=True)

    scheduler = None

    # ---------------------- Start training ----------------------
    train(model, criterion, optimizer, scheduler,
          train_dataloader, args.epochs, device)
