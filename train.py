# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import random
import shutil
from threading import local
import time
from collections import OrderedDict
import numpy as np

import paddle
import paddle.nn.functional as F
import paddle.optimizer as optim

from paddle.optimizer.lr import LambdaDecay
from paddle.io import DataLoader, RandomSampler, SequenceSampler, BatchSampler, DistributedBatchSampler

# 第1处改动 导入分布式训练所需的包
import paddle.distributed as dist

from tqdm import tqdm

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy

best_acc = 0
global local_master, logger


def get_logger(args, name=__name__, verbosity=2):
    log_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=log_levels[verbosity] if args.local_rank in [-1, 0] else logging.INFO,
                        filename=f'{args.log_out}/train@{args.num_labeled}.log',
                        filemode='a')

    msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                   log_levels.keys())
    assert verbosity in log_levels, msg_verbosity
    logger = logging.getLogger(name)
    chlr = logging.StreamHandler()  # 输出到控制台的handler
    logger.addHandler(chlr)
    return logger


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pdparams'):
    filepath = os.path.join(checkpoint, filename)
    paddle.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pdparams'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def get_cosine_schedule_with_warmup(learning_rate, num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaDecay(learning_rate=learning_rate,
                       lr_lambda=_lr_lambda,
                       last_epoch=last_epoch,
                       # verbose=True,
                       )


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose([1, 0, 2, 3, 4]).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose([1, 0, 2]).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='Paddle FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2 ** 20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--val-iter', default=64, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--log-out', default='logs',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--data-file', default='./data/cifar-10-python.tar.gz', type=str,
                        help='path to cifar10 dataset')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    args = parser.parse_args()

    global local_master, logger
    local_master = (args.local_rank == -1 or dist.get_rank() == 0)

    if local_master:
        os.makedirs(args.out, exist_ok=True)
        os.makedirs(args.log_out, exist_ok=True)
        print(f"out: {args.out}, log_out: {args.log_out}")

    logger = get_logger(args) if local_master else None

    logger.info(
        f'[Process {os.getpid()}] world_size = {dist.get_world_size()}, '
        + f'rank = {dist.get_rank()}'
    ) if local_master else None
    print(
        f'[Process {os.getpid()}] world_size = {dist.get_world_size()}, '
        + f'rank = {dist.get_rank()}'
    )

    global best_acc

    paddle.set_device('gpu') if paddle.is_compiled_with_cuda() else paddle.set_device('cpu')
    args.n_gpu = len(paddle.static.cuda_places()) if paddle.is_compiled_with_cuda() else 0

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M".format(
            (sum(p.numel() for p in model.parameters()) / 1e6).numpy()[0])) if local_master else None
        return model

    if args.local_rank == -1:
        args.device = paddle.get_device()
        args.world_size = 1
        args.n_gpu = 1
    else:
        args.device = paddle.get_device()
        # 第2处改动，初始化并行环境
        dist.init_parallel_env()
        args.world_size = dist.get_world_size()
        args.n_gpu = len(paddle.static.cuda_places())

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}", ) if local_master else None

    logger.info(dict(args._get_kwargs())) if local_master else None

    if args.seed is not None:
        set_seed(args)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    if args.local_rank not in [-1, 0]:
        # Use a barrier() to make sure that all process have finished above code
        dist.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, args.data_file)

    if args.local_rank == 0:
        dist.barrier()

    train_sampler = BatchSampler if args.local_rank == -1 else DistributedBatchSampler

    labeled_batch_sampler = train_sampler(dataset=labeled_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          drop_last=True)
    labeled_trainloader = DataLoader(
        labeled_dataset,
        batch_sampler=labeled_batch_sampler,
        num_workers=args.num_workers)

    unlabeled_batch_sampler = train_sampler(dataset=unlabeled_dataset,
                                            batch_size=args.batch_size * args.mu,
                                            shuffle=True,
                                            drop_last=True)
    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        batch_sampler=unlabeled_batch_sampler,
        num_workers=args.num_workers)

    test_sampler = SequenceSampler(test_dataset)
    test_batch_sampler = BatchSampler(sampler=test_sampler,
                                      batch_size=args.batch_size,
                                      drop_last=True)
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        dist.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        dist.barrier()

    no_decay = ['bias', 'bn']

    scheduler_1 = get_cosine_schedule_with_warmup(args.lr, args.warmup, args.total_steps)
    scheduler_2 = get_cosine_schedule_with_warmup(args.lr, args.warmup, args.total_steps)

    model_params_1 = [p for n, p in model.named_parameters() if not any(
        nd in n for nd in no_decay)]
    model_params_2 = [p for n, p in model.named_parameters() if any(
        nd in n for nd in no_decay)]

    optimizer_1 = optim.Momentum(learning_rate=scheduler_1, momentum=0.9, weight_decay=args.wdecay,
                                 parameters=model_params_1, use_nesterov=args.nesterov)
    optimizer_2 = optim.Momentum(learning_rate=scheduler_2, momentum=0.9, weight_decay=0.0,
                                 parameters=model_params_2, use_nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..") if local_master else None
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        # args.out = os.path.dirname(args.resume)
        checkpoint = paddle.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.set_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.set_state_dict(checkpoint['ema_state_dict'])
        optimizer_1.set_state_dict(checkpoint['optimizer_1'])
        optimizer_2.set_state_dict(checkpoint['optimizer_2'])

    if local_master:
        logger.info("***** Running training *****")
        logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
        logger.info(f"  Num Epochs = {args.epochs}")
        logger.info(f"  Batch size per GPU = {args.batch_size}")
        logger.info(
            f"  Total train batch size = {args.batch_size * args.world_size}")
        logger.info(f"  Total optimization steps = {args.total_steps}")

    optimizer_1.clear_grad()
    optimizer_2.clear_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer_1, optimizer_2, ema_model, scheduler_1, scheduler_2)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer_1, optimizer_2, ema_model, scheduler_1, scheduler_2):
    if args.amp:
        from apex import amp

    global best_acc
    global local_master, logger

    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    if args.local_rank != -1:
        # 第3处改动，增加paddle.DataParallel封装
        model = paddle.DataParallel(model, find_unused_parameters=True)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                paddle.concat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1)

            logits = model(inputs)
            logits = de_interleave(logits, 2 * args.mu + 1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = F.softmax(logits_u_w.detach() / args.T, axis=-1)

            max_probs, targets_u = paddle.max(pseudo_label, axis=-1), paddle.argmax(pseudo_label, axis=-1)
            mask = paddle.greater_equal(max_probs, paddle.to_tensor(args.threshold)).astype(paddle.float32)

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            loss = Lx + args.lambda_u * Lu

            if args.amp:
                with amp.scale_loss(loss, optimizer_1) as scaled_loss:
                    scaled_loss.backward()
                with amp.scale_loss(loss, optimizer_2) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer_1.step()
            scheduler_1.step()
            optimizer_2.step()
            scheduler_2.step()
            if args.use_ema:
                ema_model.update(model)
            optimizer_1.clear_grad()
            optimizer_2.clear_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if (batch_idx + 1) % args.val_iter == 0:
                logger.info(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler_1.get_lr(),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        mask=mask_probs.avg)) if local_master else None

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if local_master:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema._layers if hasattr(
                    ema_model.ema, "_layers") else ema_model.ema

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer_1': optimizer_1.state_dict(),
                'optimizer_2': optimizer_2.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))


def test(args, test_loader, model, epoch):
    global local_master, logger

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with paddle.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                if local_master:
                    logger.info(
                        "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                            batch=batch_idx + 1,
                            iter=len(test_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                        ))

    if local_master:
        logger.info("top-1 acc: {:.2f}".format(top1.avg))
        logger.info("top-5 acc: {:.2f}".format(top5.avg))

    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
