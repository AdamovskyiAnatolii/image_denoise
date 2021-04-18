import os

import ipywidgets as widgets
import torch
from matplotlib import pyplot as plt
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
        self.noise_level = widgets.FloatSlider(
            value=0.3,
            min=0,
            max=1.0,
            step=0.05,
            description="Noise:",
            disabled=False,
            continuous_update=False,
            readout=True,
            readout_format=".2f",
        )
        self.image_index = widgets.IntSlider(
            value=0,
            min=0,
            max=len(os.listdir("dataset/valid/")),
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
        with open("rand_fname.png", "rb") as file:
            self.img_widget = widgets.Image(
                value=file.read(), format="png", width=600, height=1200,
            )

        self.noise_type.observe(self.callback, names="value")
        self.noise_level.observe(self.callback, names="value")
        self.speed_radio.observe(self.callback, names="value")
        self.image_index.observe(self.callback, names="value")

        self.accordion = widgets.Accordion(
            children=[
                self.noise_type,
                self.image_index,
                self.speed_radio,
                self.noise_level,
                self.img_widget,
                self.crop_seed,
            ],
        )
        self.accordion.set_title(0, "Noise type")
        self.accordion.set_title(1, "Index")
        self.accordion.set_title(2, "Speed")
        self.accordion.set_title(3, "Noise")
        self.accordion.set_title(4, "Image")
        self.accordion.set_title(5, "Random crop")

    def show(self):
        return self.accordion

    def run_model(self, image):
        model = (
            self.bernoulli_model
            if self.noise_type.value == "bernoulli_model"
            else self.gaussian_model
        )
        if self.speed_radio.value == "long":
            res = model(image)
        elif self.speed_radio.value == "medium":
            res = model.run_one_d4(image)
        else:
            res = model.run_one(image)
        return res

    def callback(self, _):
        print(
            f"Run: inx= {self.image_index.value}, speed = {self.speed_radio.value}, noise = {self.noise_level.value}"
        )
        dataset = NoisyDataset(
            "dataset/valid/",
            crop_size=128,
            clean_target=True,
            train_noise_model=(self.noise_type.value, self.noise_level.value,),
            noise_static=True,
            crop_seed=42 if not self.crop_seed.value else None,
        )
        noisy, clean = dataset[self.image_index.value]
        noisy = noisy.numpy().transpose((1, 2, 0))
        clean = clean.numpy().transpose((1, 2, 0))
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 16))
        ax1.imshow(noisy)
        ax1.set_title("Noisy")
        ax2.imshow(clean)
        ax2.set_title("Clean")
        res = self.run_model(noisy)
        ax3.imshow(res.clip(0, 1))
        ax3.set_title("Predicted")
        f.savefig("rand_fname.png")
        plt.close(f)

        with open("rand_fname.png", "rb") as file:
            self.img_widget.value = file.read()
        print(psnr(torch.Tensor(res.clip(0, 1)), torch.Tensor(clean)))
