import os
import glob
import json

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from model.classifier import Classifier
from modules.dataset import CustomDataset
from modules.utils import *
from modules.trainer import Trainer


class CFG:

    seed = 525
    epoch = 100
    n_splits = 5
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lr = 5e-4
    batch_size = 32


def load_dataset(dataset_path):

    X_train, y_train, max_size, min_size = [], [], [], []
    label_encoder = LabelEncoder()

    json_list = sorted(glob.glob(os.path.join(dataset_path, '*/*.json')), key=lambda x: int(os.path.basename(x).split('.')[0]))
    for json_file in json_list:
        fd = json.load(open(json_file))
        annotations = fd['annotations']
        for annotation in annotations:
            image_path = os.path.join('dataset/train', f'{annotation["id"]}', f'{annotation["image_id"]}.png')
            if os.path.exists(image_path):  # check the png file exist
                X_train.append(image_path)
                y_train.append(annotation['category_id'])
                # get min & max point in the annoation['data']
                image_max = np.max(np.array(annotation['data']), axis=0).astype(int) + 100
                image_min = np.min(np.array(annotation['data']), axis=0).astype(int) - 100
                image_max, image_min = image_max[:-1], image_min[:-1]  # remove z order
                max_size.append(image_max)
                min_size.append(image_min)

    y_train = label_encoder.fit_transform(y_train).tolist()
    dataset = CustomDataset(X_train, y_train, image_size=256, max_size=max_size, min_size=min_size)

    return dataset, label_encoder


def load_model(num_classes):

    model = Classifier(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion


def train(model, optimizer, criterion, train_loader, valid_loader, fold):

    best_score = 0
    trainer = Trainer(model, CFG.device, criterion, optimizer)
    early_stop = EarlyStop(patience=10)

    for epoch_index in range(CFG.epoch):

        trainer.train_epoch(train_loader, epoch_index)
        trainer.valid_epoch(valid_loader, epoch_index)

        early_stop(trainer.valid_mean_loss)

        if early_stop.early_stop:
            break

        if trainer.valid_mean_acc > best_score:
            best_score = trainer.valid_mean_acc
            torch.save(model, f'result/{fold}_fold.pt')


def main():

    print(f'Pytorch version:[{torch.__version__}]')
    print(f"device:[{CFG.device}]")
    print(f"GPU : {torch.cuda.get_device_name(0)}")

    fix_seed(CFG.seed)

    train_dataset, label_encoder = load_dataset('dataset/train')

    kfold = KFold(n_splits=CFG.n_splits)

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_dataset)):

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(dataset=train_dataset, batch_size=CFG.batch_size, sampler=train_subsampler)
        valid_loader = DataLoader(dataset=train_dataset, batch_size=CFG.batch_size, sampler=valid_subsampler)

        model, optimizer, criterion = load_model(len(label_encoder.classes_))
        model = model.to(CFG.device)
        # print(model)
        train(model, optimizer, criterion, train_loader, valid_loader, fold)
        torch.cuda.empty_cache()


if __name__ == '__main__':

    main()