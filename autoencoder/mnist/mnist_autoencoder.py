import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
import argparse

import os
import sys

class ConvAutoencoder(nn.Module):
    def __init__(self, channels):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
                    nn.Conv2d(channels[0], channels[1], 4, stride=2, padding=1),
                    nn.ELU(),
                    nn.BatchNorm2d( channels[1]),
                    nn.Conv2d(channels[1], channels[2], 4, stride=2, padding=1),
                    nn.ELU(),
                    )

        self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(channels[2], channels[1], 4, stride=2, padding=1),
                    nn.ELU(),
                    nn.ConvTranspose2d(channels[1], channels[0], 4, stride=2, padding=1),
                    nn.Sigmoid(),
                    )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


parser = argparse.ArgumentParser(description='mnist_autoencoder')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--pretrain', action='store_true')


args = parser.parse_args()

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
if args.pretrain:
    workers = 1
    shuf = False
else:
    workers = 4
    shuf = True

x_train = torch.tensor(np.load("x_train.npy"))
x_test = torch.tensor(np.load("x_test.npy"))

train = torch.utils.data.TensorDataset(x_train,x_train)
test = torch.utils.data.TensorDataset(x_test,x_test)

train_loader = torch.utils.data.DataLoader(train, batch_size = 64, shuffle = shuf, num_workers=workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test, batch_size = 64, shuffle = shuf, num_workers=workers, pin_memory=True)

channels = [1, 8, 8]
model = ConvAutoencoder(channels).to(device)
if not args.pretrain:
    criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                            milestones=[200, 350], gamma=0.1)
    for epoch in range(400):
        model.train()
        train_loss = 0.
        train_cnt = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            reconstruction = model(images)
            loss = criterion(reconstruction, labels)
            train_loss += loss.detach()
            train_cnt += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        model.eval()
        test_loss = 0.
        test_cnt = 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            reconstruction = model(images)
            test_loss += criterion(reconstruction, labels)
            test_cnt += 1
        print(f"Epoch {epoch} | Train Loss {train_loss/train_cnt} | Test Loss {test_loss/test_cnt}")

    torch.save(model.state_dict(), "pretrained.pth")

else:
    model.load_state_dict(torch.load('pretrained.pth'))
    model.eval()
    new_train = []
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        feats = model.encoder(images).detach().cpu().numpy()
        new_train.append(feats)
        '''
        reconstruction = model(images)
        image = images[:10,:,:,:].detach().cpu().view(10,1,28,28)
        re_image = reconstruction[:10,:,:,:].detach().cpu().view(10,1,28,28)
        save_image(image, 'img.png', nrow=10)
        save_image(re_image, 're_img.png', nrow=10)
        sys.exit()
        '''
    new_train = np.concatenate(new_train, axis=0)
    np.save("encoded_train.npy", new_train.reshape(new_train.shape[0], -1))
