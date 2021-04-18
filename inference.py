import json
import os

import albumentations as A
import numpy as np
import torch
from piq import psnr

from model import Net


class Cleaner:
    def __init__(self, version, log_dir="lightning_logs", params="params.json"):
        self.version = version
        self.log_dir = log_dir
        self.params = params
        self.checkpoints_dir = f"{self.log_dir}/{self.version}/checkpoints"
        self.models = []
        self.load_checkpoints()

    def run_one(self, image, inx=-1):
        image = torch.Tensor([image.transpose(2, 0, 1)])
        res = self.models[inx](image)
        return res[0].detach().numpy().transpose(1, 2, 0)

    def run_one_d4(self, image, inx=-1):
        rotate = A.Rotate((90, 90), p=1)
        de_rotate = A.Rotate((-90, -90), p=1)

        x0 = image
        x90 = rotate(image=x0)["image"]
        x180 = rotate(image=x90)["image"]
        x270 = rotate(image=x180)["image"]

        x0 = x0.transpose(2, 0, 1)
        x90 = x90.transpose(2, 0, 1)
        x180 = x180.transpose(2, 0, 1)
        x270 = x270.transpose(2, 0, 1)

        x = np.array([x0, x90, x180, x270])

        res_d4 = self.models[inx](torch.Tensor(x))
        res_d4n = [x.transpose(1, 2, 0) for x in res_d4.detach().numpy()]

        dex0 = res_d4n[0]
        dex90 = de_rotate(image=res_d4n[1])["image"]
        dex180 = de_rotate(image=de_rotate(image=res_d4n[2])["image"])["image"]
        dex270 = de_rotate(
            image=de_rotate(image=de_rotate(image=res_d4n[3])["image"])["image"]
        )["image"]
        return (dex0 + dex90 + dex180 + dex270) / 4

    def run_all(self, image):
        return np.mean([model(image) for model in self.models], axis=0)

    def run_all_d4(self, image):
        return np.mean(
            [self.run_one_d4(image, inx) for inx in range(len(self.models))],
            axis=0,
        )

    def load_checkpoints(self):
        with open(self.params) as file:
            params = json.load(file)
        for name in os.listdir(self.checkpoints_dir):
            self.models.append(Net(**params))
            self.models.load_state_dict(
                torch.load(os.path.join(self.checkpoints_dir, name))[
                    "state_dict"
                ]
            )


class MultiplicativeBernoulliCleaner(Cleaner):
    def __init__(self, gaussian_model=None, **kwargs):
        super(MultiplicativeBernoulliCleaner, self).__init__(**kwargs)
        self.gaussian_model = gaussian_model

        self.checkpoints_dir = f"{self.log_dir}/{self.version}/checkpoints"
        self.load_checkpoints()

    def __call__(self, image):
        cleaned = self.replace(image, self.run_all_d4(image))
        if self.gaussian_model is not None:
            cleaned = self.replace(image, self.gaussian_model(cleaned))
        return cleaned

    @staticmethod
    def replace(noisy, res):
        noisy_copy = noisy.copy()
        noisy_copy[noisy == 0] = res[noisy == 0]
        return noisy_copy


class GaussianCleaner(Cleaner):
    def __init__(self, add_noise=False, psnr_level=25, **kwargs):
        super(GaussianCleaner, self).__init__(**kwargs)
        self.add_noise = add_noise
        self.psnr_level = psnr_level

        self.checkpoints_dir = f"{self.log_dir}/{self.version}/checkpoints"
        self.load_checkpoints()

    def __call__(self, image):
        cleaned = self.run_all_d4(image)
        if self.add_noise:
            cleaned = self.add_input_noise(image, cleaned)
        return cleaned

    def add_input_noise(self, noisy, res):
        if (
            psnr(torch.Tensor(noisy.clip(0, 1)), torch.Tensor(res.clip(0, 1)))
            > self.psnr_level
        ):
            res = (res + noisy) / 2
        return res
