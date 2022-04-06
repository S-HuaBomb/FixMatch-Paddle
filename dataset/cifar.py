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

from paddle.vision import datasets, transforms
import logging
import math

import numpy as np
from PIL import Image
from paddle.vision import datasets

from dataset.randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

# 全局取消证书验证
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class CIFAR10SSL(datasets.Cifar10):
    def __init__(self, data_file, indexs, mode='train',
                 transform=None, download=True, backend='pil'):
        super().__init__(data_file=data_file, mode=mode,
                         transform=transform, download=download, backend=backend)
        if indexs is not None:
            # print(f"indexs: {indexs}")
            self.data = np.asarray(self.data)[indexs]
            # self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        image, label = self.data[index]
        # print(f"index {index}, image: {image.shape}")
        image = np.reshape(image, [3, 32, 32])
        image = image.transpose([1, 2, 0])  # HWC

        if self.backend == 'pil':
            image = Image.fromarray(image.astype('uint8'))
        if self.transform is not None:
            image = self.transform(image)

        if self.backend == 'pil':
            return image, np.array(label).astype('int64')

        return image, label


def get_cifar10(args, data_file):
    """
    返回有/无 标签训练集和测试集 dataset
    """

    # 有标签训练数据采用常规弱增强
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    # 测试集转成 Tensor 以及标准归一化
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    base_dataset = datasets.Cifar10(
        data_file=data_file,
        mode='train',
        download=True,
        transform=transform_val)

    # 划分指定数量的有/无标签训练集
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, np.asarray(base_dataset.data)[:, 1])  # 取标签列

    train_labeled_dataset = CIFAR10SSL(
        data_file, train_labeled_idxs,
        mode='train',
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        data_file, train_unlabeled_idxs,
        mode='train',
        # 无标签训练数据采用强增强
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.Cifar10(data_file=data_file,
                                    mode='test',
                                    transform=transform_val,
                                    download=True)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // 10
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    """
    调用 RandAugment 作为 Transform
    """
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


DATASET_GETTERS = {'cifar10': get_cifar10}
