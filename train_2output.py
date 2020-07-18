import time
from test import test

import torch
from utils import util


# ---------------------- Train function ----------------------
def train(model, criterion, optimizer, scheduler, dataloader, num_epochs, device, save_dir_path, args):
    '''
        train
    '''
    start_time = time.time()

    # Logger instance--------------------------------------------
    logger = util.Logger(save_dir_path)
    # logger.info('-' * 10)
    logger.info(vars(args))
    # logger.info(model)

    # +++++++++++++++++++++++++++++++++start++++++++++++++++++++++++++++++++++++++++
    for epoch in range(num_epochs):
        

        model.train()
        scheduler.step(epoch)

        # # train open_specified_layers--------------------------------------------------
        # if (epoch+1) <= args.fixbase_epoch and args.open_layers is not None:
        #     logger.info('* Only train {} (epoch: {}/{})'.format(args.open_layers, epoch+1, fixbase_epoch))
        #     torchtool.open_specified_layers(model, args.open_layers)
        # else:
        #     # logger.info('open all layers.')
        #     torchtool.open_all_layers(model)

        # ===================one epoch====================
        # Training
        running_loss = 0.0
        for data in dataloader:
            inputs, labels, _ = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # with torch.set_grad_enabled(True):-------------
            parts_outputs, gloab_outputs = model(inputs)
            gloab_loss = criterion(gloab_outputs, labels)
            # Sum up the stripe softmax loss-------------------
            part_loss = 0
            for logits in parts_outputs:
                stripe_loss = criterion(logits, labels)
                part_loss += stripe_loss
            loss = part_loss+gloab_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        # ===================one epoch end================

        epoch_loss = running_loss / len(dataloader.dataset)

        # logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # logger.info('Training Loss: {:.4f}'.format(epoch_loss))
        # time_remaining = (num_epochs - epoch)*(time.time() - start_time)/(epoch+1)
        # logger.info('time remaining  is {:.0f}h : {:.0f}m'.format(time_remaining//3600, time_remaining/60 % 60))

        # Save result to logger---------------------------------
        logger.x_epoch_loss.append(epoch + 1)
        logger.y_train_loss.append(epoch_loss)

        # Testing / Validating-----------------------------------
        if (epoch + 1) % args.test_every == 0 or epoch + 1 == num_epochs:
            torch.cuda.empty_cache()
            CMC, mAP = test(model, args.dataset, args.dataset_path, args.test_batch_size, device, args)
            logger.info('Testing: top1:%.4f top5:%.4f top10:%.4f mAP:%.4f' % (CMC[0], CMC[4], CMC[9], mAP))

            logger.x_epoch_test.append(epoch + 1)
            logger.y_test['top1'].append(CMC[0])
            logger.y_test['mAP'].append(mAP)
            if epoch + 1 != num_epochs:
                util.save_network(model, save_dir_path, str(epoch + 1))
        logger.info('-' * 10)

    # +++++++++++++++++++++++++++++++++start end+++++++++++++++++++++++++++++++++

    # Save the loss curve-----------------------------------
    logger.save_curve()

    time_elapsed = time.time() - start_time
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # Save final model weights-----------------------------------
    util.save_network(model, save_dir_path, 'final')
