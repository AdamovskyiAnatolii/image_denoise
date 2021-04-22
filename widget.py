import io
import os

import ipywidgets as widgets
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from piq import psnr

from dataset import NoisyDataset


class Widget:
    def __init__(self, bernoulli_model, gaussian_model):
        self.bernoulli_model = bernoulli_model
        self.gaussian_model = gaussian_model
        self.noise_type = widgets.RadioButtons(
            options=["multiplicative_bernoulli", "gaussian"],
            description="Noise type:",
            disabled=False,
        )
        self.speed_radio = widgets.RadioButtons(
            options=["fast", "medium", "long"],
            description="Speed:",
            disabled=False,
        )
        self.bernoulli_noise_level = widgets.FloatSlider(
            value=0.3,
            min=0,
            max=1.0,
            step=0.05,
            description="Bernoulli Noise:",
            disabled=False,
            continuous_update=False,
            readout=True,
            readout_format=".2f",
        )
        self.gaussian_noise_level = widgets.IntSlider(
            value=30,
            min=0,
            max=100,
            step=1,
            description="Gaussian Noise:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        self.dataset_name = widgets.RadioButtons(
            options=[
                os.path.join("datasets", path)
                for path in os.listdir("datasets")
            ],
            description="Dataset name:",
            disabled=False,
        )
        self.image_index = widgets.IntSlider(
            value=0,
            min=0,
            max=len(os.listdir(self.dataset_name.value)),
            step=1,
            description="Index:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        self.crop_seed = widgets.Checkbox(
            value=False, description="Random crop", disabled=False, indent=False
        )
        self.file = widgets.FileUpload(accept="", multiple=False)
        with open("rand_fname.png", "rb") as file:
            self.img_widget = widgets.Image(
                value=file.read(), format="png", width=600, height=1200,
            )

        self.noise_type.observe(self.callback, names="value")
        self.bernoulli_noise_level.observe(self.callback, names="value")
        self.gaussian_noise_level.observe(self.callback, names="value")
        self.speed_radio.observe(self.callback, names="value")
        self.image_index.observe(self.callback, names="value")
        self.dataset_name.observe(self.callback, names="value")
        self.file.observe(self.callback_upload, names="value")

        self.accordion = widgets.Accordion(
            children=[
                self.noise_type,
                self.image_index,
                self.speed_radio,
                self.bernoulli_noise_level,
                self.gaussian_noise_level,
                self.img_widget,
                self.crop_seed,
                self.dataset_name,
                self.file,
            ],
        )
        self.accordion.set_title(0, "Noise type")
        self.accordion.set_title(1, "Index")
        self.accordion.set_title(2, "Speed")
        self.accordion.set_title(3, "Bernoulli Noise")
        self.accordion.set_title(4, "Gaussian Noise")
        self.accordion.set_title(5, "Image")
        self.accordion.set_title(6, "Random crop")
        self.accordion.set_title(7, "Dataset name")
        self.accordion.set_title(8, "Upload file")

    def show(self):
        return self.accordion

    def run_model(self, image):
        model = (
            self.bernoulli_model
            if self.noise_type.value == "multiplicative_bernoulli"
            else self.gaussian_model
        )
        if self.speed_radio.value == "long":
            res = model(image)
        elif self.speed_radio.value == "medium":
            res = model.run_one_d4(image)
        else:
            res = model.run_one(image)
        return res

    def run_model_upload(self, image):
        model = (
            self.bernoulli_model
            if self.noise_type.value == "multiplicative_bernoulli"
            else self.gaussian_model
        )
        return model.run_one(image)

    def callback(self, _):
        dataset = self._make_dataset()

        noisy, clean = dataset[self.image_index.value]
        noisy = noisy.numpy().transpose((1, 2, 0))
        clean = clean.numpy().transpose((1, 2, 0))

        res = self.run_model(noisy)
        self._save_image(noisy, clean, res)
        self._reload_image()

        print(psnr(torch.Tensor(np.array(res).clip(0, 1)), torch.Tensor(clean)))

    def callback_upload(self, _):
        print("Upload file")
        dataset = self._make_dataset()

        clean = Image.open(
            io.BytesIO(
                self.file.value[list(self.file.value.keys())[0]]["content"]
            )
        )
        print("Add noise")
        noisy = dataset.corrupt_image(clean)
        print("Start clean")
        res = self.run_model_upload(noisy)
        print("End clean")
        self._save_image(noisy, clean, res)
        self._reload_image()

        print(psnr(torch.Tensor(np.array(res).clip(0, 1)), torch.Tensor(clean)))

    def _reload_image(self):
        with open("rand_fname.png", "rb") as file:
            self.img_widget.value = file.read()

    def _make_dataset(self):
        noise_level = (
            self.bernoulli_noise_level.value
            if self.noise_type.value == "multiplicative_bernoulli"
            else self.gaussian_noise_level.value
        )
        print(
            f"Run: inx= {self.image_index.value}, speed = {self.speed_radio.value}, noise = {noise_level}"
        )
        return NoisyDataset(
            self.dataset_name.value,
            crop_size=128,
            clean_target=True,
            train_noise_model=(self.noise_type.value, noise_level,),
            noise_static=True,
            crop_seed=42 if not self.crop_seed.value else None,
        )

    @staticmethod
    def _save_image(noisy, clean, pred):
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))
        ax1.imshow(noisy)
        ax1.set_title("Noisy")
        ax2.imshow(clean)
        ax2.set_title("Clean")
        ax3.imshow(pred.clip(0, 1))
        ax3.set_title("Predicted")
        f.savefig("rand_fname.png")
        plt.close(f)
