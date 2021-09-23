import os
import glob
import json
import pickle

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from modules.dataset import CustomDataset


def load_dataset(folder):

    X_test, max_size, min_size = [], [], []

    json_file = glob.glob(os.path.join(folder, '*.json'))[0]
    fd = json.load(open(json_file))
    annotations = fd['annotations']
    for annotation in annotations:
        image_path = os.path.join('dataset/test', f'{annotation["id"]}', f'{annotation["image_id"]}.png')
        if os.path.exists(image_path):  # check the png file exist
            X_test.append(image_path)
            # get min & max point in the annoation['data']
            image_max = np.max(np.array(annotation['data']), axis=0).astype(int) + 100
            image_min = np.min(np.array(annotation['data']), axis=0).astype(int) - 100
            image_max, image_min = image_max[:-1], image_min[:-1]  # remove z order
            max_size.append(image_max)
            min_size.append(image_min)
    dataset = CustomDataset(X_test, image_size=380, max_size=max_size, min_size=min_size)

    return dataset


def predict():

    # test_dataset = load_dataset('dataset/test')
    # label_encoder = pickle.load(open('result/label_encoder.pt', 'rb'))
    #
    # # print(models[0])
    #
    # test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    # y_preds_mean = []
    # for i in range(len(models)):
    #     print(f'model {i} predict start')
    #     y_preds = []
    #     models[i].cuda()
    #     models[i].eval()
    #
    #     with torch.no_grad():
    #         for batch_index, image in enumerate(test_loader):
    #             image = image.cuda()
    #             output = models[i](image)
    #             output = softmax(output)
    #             y_preds.extend(output.detach().cpu().numpy().tolist())
    #             # print(output)
    #             # y_pred = np.argmax(output.data.cpu().numpy(), axis=1)
    #             # y_preds.extend(y_pred.tolist())
    #             # y_trues.extend(label.cpu().numpy().tolist())
    #     y_preds_mean.append(y_preds)
    #
    # y_preds_mean = np.array(y_preds_mean)
    # y_preds_mean = np.mean(y_preds_mean, axis=0)
    # pickle.dump(y_preds_mean, open('result/result_pred', 'wb'))
    # print(y_preds_mean.shape)


    # folder -> model -> submission

    submission = pd.read_csv('dataset/sample_submission.csv')

    models = [torch.load(f"result/{fold}_fold.pt") for fold in range(5)]
    softmax = nn.Softmax(dim=1)

    folder_list = sorted(glob.glob(os.path.join('dataset/test/**')))

    total_y_preds = []
    for index, folder in enumerate(folder_list):
        print(os.path.basename(folder))

        # define dataset and dataloader here (glob all image in one folder)
        dataset = load_dataset(folder)
        data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        y_preds = []
        for i in range(len(models)):
            print(f'\t{i} model predict start')

            models[i].cuda()
            models[i].eval()
            y_pred = []

            with torch.no_grad():
                for batch_index, image in enumerate(data_loader):
                    image = image.cuda()
                    output = models[i](image)
                    output = softmax(output)
                    y_pred.extend(output.detach().cpu().numpy().tolist())

            y_pred = np.array(y_pred)
            y_preds.append(np.mean(y_pred, axis=0))  # model 별 폴더 내의 모든 이미지의 평균 softmax 값

        y_preds = np.array(y_preds)
        total_y_preds.append(np.mean(y_preds, axis=0))  # cv model의 전체 평균 softmax 값
    total_y_preds = np.array(total_y_preds)
    print(total_y_preds.shape)
    pickle.dump(total_y_preds, open('result/result_pred', 'wb'))

    submission.iloc[:, 1:] = total_y_preds
    submission.to_csv('result/submit.csv', index=False)

if __name__ == '__main__':

    predict()