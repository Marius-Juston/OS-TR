import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.data import datasets
from utils.data.augmentation import Transformation
from utils.evaluate import Evaluator
from utils.loss import myloss
from utils.model import models

torch.cuda.set_per_process_memory_fraction(1.)


def main(seed=2018, epoches=1000):
    parser = argparse.ArgumentParser(description='my_trans')

    # dataset option
    parser.add_argument('--dataset_name', type=str, default='dtd', choices=['dtd'], help='dataset name (default: my)')
    parser.add_argument('--model_name', type=str, default='dtd', choices=['baseline', 'attention', 'tex'],
                        help='model name (default: my)')
    parser.add_argument('--loss_name', type=str, default='weighted_bce', choices=['weighted_bce', 'DF'],
                        help='model name (default: my)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: auto)')
    parser.add_argument('--checkname', type=int, default=0, help='set the checkpoint name')
    parser.add_argument('--train_batch_size', type=int, default=8,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=8,
                        metavar='N', help='input batch size for testing (default: auto)')

    args = parser.parse_args()

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    if args.dataset_name == 'dtd':
        # transform_zk = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Lambda(lambda x: (x * 255).type(torch.uint8)),
        #     transforms.RandomEqualize(p=1),
        #     transforms.Lambda(lambda x: x.type(torch.FloatTensor) / 255.),
        #     transforms.RandomHorizontalFlip(p=.5),
        #     transforms.RandomVerticalFlip(p=.5),
        #     transforms.TrivialAugmentWide(),
        #     transforms.RandomAffine(45, (.1 , .1), (0.5, 1)),
        # ])

        # transform_zk = Compose([
        #     transforms.ToTensor(),
        #
        #     transforms.Normalize((0.5355, 0.4852, 0.4441), std=(0.2667, 0.2588, 0.2667)),
        #
        #     # transforms.Lambda(lambda x: (x * 255).type(torch.uint8)),
        #     transforms.ColorJitter(.1, .1, .1),
        #     transforms.RandomHorizontalFlip(p=.5),
        #     transforms.RandomVerticalFlip(p=.5),
        #     # transforms.TrivialAugmentWide(),
        #     transforms.RandomAffine(30, (.1, .1), (0.75, 1), 30),
        #     # transforms.Lambda(lambda x: x.type(torch.FloatTensor) / 255.),
        # ], )
        transform = Transformation()
        evaluator = Evaluator(num_class=6)

    mydataset_embedding = datasets[args.dataset_name]
    data_val1 = mydataset_embedding(split='test1', transform=None, checkpoint=args.checkname)
    loader_val1 = DataLoader(data_val1, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    data_train = mydataset_embedding(split='train', transform=transform, checkpoint=args.checkname)
    # data_train = mydataset_embedding(split='train', transform=None, checkpoint=args.checkname)
    loader_train = DataLoader(data_train, batch_size=args.train_batch_size, shuffle=True, num_workers=0,
                              pin_memory=True)

    dir_name = 'log/' + str(args.dataset_name) + '_' + str(args.model_name) + '_' + str(args.loss_name) + '_' + \
               data_val1.test[0] + '_' + str(args.lr)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    now_time = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    logging.basicConfig(level=logging.INFO,
                        filename=dir_name + '/output_' + now_time + '.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info('dataset_name: %s, model_name: %s, loss_name: %s', args.dataset_name, args.model_name, args.loss_name)
    logging.info('test with: %s', data_val1.test)

    model = models[args.model_name]()

    model.load_state_dict(
        torch.load('./log/dtd_dtd_weighted_bce_banded_0.001/snapshot-epoch_2021-11-26-02:09:42_texture.pth'))

    # model = torch.jit.script(model)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    model.train()

    criterion = myloss[args.loss_name]()

    optim_para = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = torch.optim.SGD(optim_para, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    optimizer = Adam(optim_para, lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    plat_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=20, factor=.5)

    IoU_final = 0
    epoch_final = 0
    losses = 0
    iteration = 0

    print("Num steps per epoch:", len(data_train) / args.train_batch_size)

    for epoch in range(epoches):
        train_loss = 0
        logging.info('epoch:' + str(epoch))
        start = time.time()
        np.random.seed(epoch)
        for i, data in enumerate(loader_train):
            _, _, inputs, target, patch, _ = data[0], data[1], data[2], data[3], data[4], data[5]

            # im = inputs[0, :, :].detach().cpu().numpy().transpose([1, 2, 0])
            # tar = target[0, :, :].detach().cpu().numpy().transpose([1, 2, 0])
            # pat = patch[0, :, :].detach().cpu().numpy().transpose([1, 2, 0])
            # #
            # print(pat.shape)
            #
            # fig, axes = plt.subplots(1, 3)
            #
            # axes[0].imshow(np.uint8(im * 255))
            # axes[1].imshow(np.uint8(tar[..., 0] * 255))
            # axes[2].imshow(np.uint8(pat * 255))
            # plt.show()

            inputs = inputs.float()
            iteration += 1
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                target = target.cuda(non_blocking=True)
                # target = target.cuda()
                patch = patch.cuda()

            output = model(inputs, patch)

            output = output.unsqueeze(1)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()
            losses += loss.detach().item()

            if iteration % 20 == 0:
                run_time = time.time() - start
                start = time.time()
                losses = losses / 20
                logging.info(
                    'iter:' + str(iteration) + " time:" + str(run_time) + " train loss = {:02.5f}".format(losses))

                losses = 0
            # break

        # scheduler.step()
        print("Finished epoch")

        snapshot_path = dir_name + '/snapshot-epoch_{epoches}_texture.pth'.format(epoches=now_time)
        model.eval()

        with torch.no_grad():
            evaluator.reset()
            torch.cuda.empty_cache()
            np.random.seed(2019)
            for i, data in enumerate(loader_val1):
                _, _, inputs, target, patch, image_class = data[0], data[1], data[2], data[3], data[4], data[5]
                inputs = inputs.float()
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    target = target.cuda(non_blocking=True)
                    patch = patch.cuda()

                scores = model(inputs, patch)
                scores[scores >= 0.5] = 1
                scores[scores < 0.5] = 0
                seg = scores[:, 0, :, :].long()
                pred = seg.data.cpu().numpy()
                target = target.cpu().numpy()
                # Add batch sample into evaluator
                evaluator.add_batch(target, pred, image_class)

            mIoU, mIoU_d = evaluator.Mean_Intersection_over_Union()
            FBIoU = evaluator.FBIoU()

            logging.info("{:10s} {:.3f}".format('IoU_mean', mIoU))
            logging.info("{:10s} {}".format('IoU_mean_detail', mIoU_d))
            logging.info("{:10s} {:.3f}".format('FBIoU', FBIoU))
            if mIoU > IoU_final:
                epoch_final = epoch
                IoU_final = mIoU
                torch.save(model.state_dict(), snapshot_path)
            logging.info('best_epoch:' + str(epoch_final))
            logging.info("{:10s} {:.3f}".format('best_IoU', IoU_final))
        plat_scheduler.step(mIoU)
        logging.info(f"LR: {optimizer.param_groups[0]['lr']}")
        model.train()

    logging.info(epoch_final)
    logging.info(IoU_final)


if __name__ == '__main__':
    main()
