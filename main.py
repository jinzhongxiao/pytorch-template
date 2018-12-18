from data.dataset import DogCat

import torch as t
import torch.nn as nn
from torch.nn import Module
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from config import DefaultConfig
import models

import os
import sys
import time
from torchnet import meter
opt = DefaultConfig()


def train(**kwargs):

    opt.parse(kwargs)

    #############################
    #  model define
    model = getattr(models, opt.model)(opt.num_classes)

    model.train()
    if opt.use_gpu:
        model.cuda()
    if opt.load_model_path is not None:
        model.load(opt.load_model_path)

    #############################
    #  dataset preparation
    transform_train = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225))
    ])

    transform_val = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = DogCat(opt.train_data_root, transforms=transform_train)
    train_dataset.describe()
    val_dataset = DogCat(opt.test_data_root,
                         transforms=transform_val, test=False, train=False)

    val_dataset.describe()
    train_dataloader = DataLoader(train_dataset, opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers,
                                  )
    val_dataloader = DataLoader(train_dataset, opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers,
                                )

    #############################
    #  loss define
    criterion = t.nn.CrossEntropyLoss()

    #############################
    #  optimizer define   !!! SGD will be better
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=opt.weight_decay)

    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    count = 0
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        start_time = time.time()
        for ii, (data, label) in enumerate(train_dataloader):

            input, target = data, label
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            jj = time.time()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            # debug  grad  explosion
            # for name, param in model.feature_layers.named_parameters():
            #     print(name, param.grad[:10])
            #     break

            #  update measure
            loss_meter.add(loss.data.item())
            confusion_matrix.add(score.data, target.data)

            batch_time = time.time() - start_time
            if count == 0:
                print("One Batch Time is {}".format(batch_time))
                print("One epoch Time is {}".format(batch_time * (1. * len(train_dataset)/opt.batch_size)))
                count += 1
        model.save()

        val_cm, val_accuracy = val(model, val_dataloader)
        print("#"*30)
        print("epoch:{epoch},lr:{lr},loss:{loss},acc:{acc},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch,
            loss=loss_meter.value()[0],
            val_cm=str(val_cm.value()),
            acc=val_accuracy,
            train_cm=str(confusion_matrix.value()),
            lr=lr))

        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        previous_loss = loss_meter.value()[0]


def val(model, dataloader):
    model.eval()

    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data

        if opt.use_gpu:
            input = input.cuda()
            label = label.cuda()
        score = model(input)
        confusion_matrix.add(score.data.squeeze(), label)

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


def test(**kwargs):
    opt.parse(kwargs)

    # model define
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    pass


def help():
    """
     python file.py help
    """

    print("""
    usage : python {0} <function> [--args=value,]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire
    fire.Fire()
