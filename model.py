model.py
Без спільного доступу
Тип
Текст
Розмір
7 КБ (6 857 байтів)
Використовується
12 КБ (12 424 байти)
Розташування
lightning
Власник
я
Змінено
17 квіт. 2021 р. (мною)
Відкрито
16:43 (мною)
Створено
15 квіт. 2021 р. у додатку Google Drive Web
Додати опис
Користувачі з правами перегляду можуть завантажувати файл

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from piq import psnr


import pytorch_lightning as pl
from .dataset import NoisyDataset


class ConvBlock(nn.Module):
    def __init__(self, ni, no, ks, stride=1, pad=1, use_act=True):
        super(ConvBlock, self).__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(ni, no, ks, stride=stride, padding=pad)
        self.bn = nn.BatchNorm2d(no)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        op = self.bn(self.conv(x))
        return self.act(op) if self.use_act else op


class ResBlock(nn.Module):
    def __init__(self, ni, no, ks):
        super(ResBlock, self).__init__()
        self.block1 = ConvBlock(ni, no, ks)
        self.block2 = ConvBlock(ni, no, ks, use_act=False)

    def forward(self, x):
        return x + self.block2(self.block1(x))


class Net(pl.LightningModule):
    def __init__(
        self,
        input_channels,
        output_channels,
        lr,
        batch_size,
        train_dataset,
        val_dataset,
        test_dataset,
        train_crop_size,
        val_crop_size,
        test_crop_size,
        train_noise_model,
        val_noise_model,
        test_noise_model,
        train_clean_target=True,
        val_clean_target=True,
        test_clean_target=True,
        res_layers=16,
        num_workers=4,
    ):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        # self.act = nn.LeakyReLU(0.2, inplace=True)

        # _resl = [ResBlock(output_channels, output_channels, 3) for i in range(res_layers)]
        # self.resl = nn.Sequential(*_resl)

        # self.conv2 = ConvBlock(output_channels, output_channels, 3, use_act=False)
        # self.conv3 = nn.Conv2d(output_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.conv0 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, stride=1, padding=1
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

        _resl = [
            ResBlock(output_channels, output_channels, 3)
            for i in range(res_layers)
        ]
        self.resl = nn.Sequential(*_resl)

        _resl = [
            ResBlock(output_channels, output_channels, 3)
            for i in range(res_layers // 2)
        ]
        self.resl2 = nn.Sequential(*_resl)

        self.conv2 = ConvBlock(
            output_channels, output_channels, 3, use_act=False
        )
        self.conv3 = ConvBlock(
            output_channels, output_channels, 3, use_act=False
        )

        self.conv4 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv6 = nn.Conv2d(
            output_channels, input_channels, kernel_size=3, stride=1, padding=1
        )
        self.lr = lr
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.train_crop_size = train_crop_size
        self.train_noise_model = train_noise_model
        self.val_dataset = val_dataset
        self.val_crop_size = val_crop_size
        self.val_noise_model = val_noise_model
        self.test_dataset = test_dataset
        self.test_crop_size = test_crop_size
        self.test_noise_model = test_noise_model
        self.train_clean_target = train_clean_target
        self.val_clean_target = val_clean_target
        self.test_clean_target = test_clean_target
        self.num_workers = num_workers
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, input):
        # _op1 = self.act(self.conv1(input))
        # _op2 = self.conv2(self.resl(_op1))
        # op = self.conv3(torch.add(_op1, _op2))
        _op0 = self.act(self.conv0(input))
        _op1 = self.act(self.conv1(input))
        _op2 = self.conv2(self.resl(_op1))
        _op3 = self.conv3(self.resl2(_op1))

        op1 = self.act(self.bn1(self.conv4(torch.add(_op1, _op2))))
        op2 = self.act(self.bn2(self.conv5(torch.add(_op1, _op3))))

        op = self.conv6(torch.add(op1, op2))
        return op + input

    def step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # if mask is not None:
        #     mask = 1 - mask
        #     y = y * mask
        #     y_hat = y_hat * mask

        loss_mse = F.mse_loss(y_hat, y, reduction='mean')
        psnr_loss = psnr(y_hat.clip(0, 1), y.clip(0, 1))

        return (
            loss_mse,
            {"loss_mse": loss_mse, "psnr_loss": psnr_loss},
        )

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(
            NoisyDataset(
                root_dir=self.train_dataset,
                crop_size=self.train_crop_size,
                train_noise_model=self.train_noise_model,
                clean_target=self.train_clean_target,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            NoisyDataset(
                root_dir=self.val_dataset,
                crop_size=self.val_crop_size,
                train_noise_model=self.val_noise_model,
                clean_target=self.val_clean_target,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            NoisyDataset(
                root_dir=self.test_dataset,
                crop_size=self.test_crop_size,
                train_noise_model=self.test_noise_model,
                clean_target=self.test_clean_target,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
