import tqdm

import torch
import numpy as np
import albumentations as A

from utils import rand_bbox

augmenter = A.Compose(
    [
        A.Cutout(num_holes=30, max_h_size=30, max_w_size=30, fill_value=0)
    ]
)


class Trainer:

    def __init__(self, model, device, criterion, optimizer):

        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def train_epoch(self, train_loader, epoch_index):

        self.model.train()
        self.train_loss = 0
        sample_num = 0

        y_preds = []
        y_trues = []

        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)

        for step, (image, label) in pbar:
            self.optimizer.zero_grad()

            image = image.to(self.device)
            label = label.to(self.device)

            # cut mix
            if np.random.random() > 0.5:
                lam = np.random.beta(1., 1.)
                rand_index = torch.randperm(image.size()[0]).to(self.device)
                target_a = label
                target_b = label[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
                output = self.model(image)
                loss = self.criterion(output, target_a) * lam + self.criterion(output, target_b) * (1. - lam)

            else:
                output = self.model(image)
                loss = self.criterion(output, label)

            self.train_loss += loss
            sample_num += label.shape[0]
            y_pred = np.argmax(output.data.cpu().numpy(), axis=1)
            y_preds.extend(y_pred.tolist())
            y_trues.extend(label.cpu().numpy().tolist())

            description = f'loss: {self.train_loss / sample_num:.4f}'
            pbar.set_description(description)

            loss.backward()
            # if (step + 1) % 4 == 0 or step == len(train_loader) - 1:
            self.optimizer.step()

        self.train_mean_loss = self.train_loss / len(train_loader)
        self.train_mean_acc = np.mean(np.array(y_preds) == np.array(y_trues))
        msg = f'Epoch {epoch_index}, Train, loss: {self.train_mean_loss}, Score: {self.train_mean_acc}'
        print(msg)

    def valid_epoch(self, valid_loader, epoch_index):

        self.model.eval()
        self.valid_loss = 0
        y_preds = []
        y_trues = []

        pbar = tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader), position=0, leave=True)

        with torch.no_grad():

            for step, (image, label) in pbar:

                image = image.to(self.device)
                label = label.to(self.device)

                output = self.model(image)
                loss = self.criterion(output, label)
                self.valid_loss += loss

                y_pred = np.argmax(output.data.cpu().numpy(), axis=1)
                y_preds.extend(y_pred.tolist())
                y_trues.extend(label.cpu().numpy().tolist())

        self.valid_mean_loss = self.valid_loss / len(valid_loader)
        self.valid_mean_acc = np.mean(np.array(y_preds) == np.array(y_trues))
        msg = f'Epoch {epoch_index}, Valid, loss: {self.valid_mean_loss}, Score: {self.valid_mean_acc}'
        print(msg)