import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import random


class NoisyDataset(Dataset):
    def __init__(
        self,
        root_dir,
        crop_size=128,
        train_noise_model=("gaussian", 50),
        clean_target=False,
        noise_static=False,
        crop_seed=None
    ):
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.clean_target = clean_target
        self.noise = train_noise_model[0]
        self.noise_param = train_noise_model[1]
        self.image_name_list = os.listdir(root_dir)
        self.noise_static = noise_static
        self.crop_seed = crop_seed

    def _random_crop_to_size(self, imgs):
        if self.crop_seed is not None:
            np.random.seed(self.crop_seed)

        w, h = imgs[0].size
        assert (
            w >= self.crop_size and h >= self.crop_size
        ), "Cannot be croppped. Invalid size"

        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 2)
        j = np.random.randint(0, w - self.crop_size + 2)

        for img in imgs:
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))
            cropped_imgs.append(
                tvF.crop(img, i, j, self.crop_size, self.crop_size)
            )

        # cropped_imgs = cv2.resize(np.array(imgs[0]), (self.crop_size, self.crop_size))
        return cropped_imgs

    def _add_gaussian_noise(self, image):
        """
        Added only gaussian noise
        """
        w, h = image.size
        c = len(image.getbands())

        if self.noise_static:
            std = self.noise_param
        else:
            std = np.random.uniform(0, self.noise_param)
        _n = np.random.normal(0, std, (h, w, c))
        noisy_image = np.array(image) + _n

        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return {
            "image": Image.fromarray(noisy_image),
            "mask": None,
            "use_mask": False,
        }

    def _add_m_bernoulli_noise(self, image):
        """
        Multiplicative bernoulli
        """
        sz = np.array(image).shape[0]
        if self.noise_static:
            prob_ = self.noise_param
        else:
            prob_ = random.uniform(0, self.noise_param)
        mask = np.random.choice([0, 1], size=(sz, sz), p=[prob_, 1 - prob_])
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        return {
            "image": np.multiply(image, mask).astype(np.uint8),
            "mask": mask.astype(np.uint8),
            "use_mask": True,
        }

    def corrupt_image(self, image):

        if self.noise == "gaussian":
            return self._add_gaussian_noise(image)
        elif self.noise == "multiplicative_bernoulli":
            return self._add_m_bernoulli_noise(image)
        else:
            raise ValueError("No such image corruption supported")

    def __getitem__(self, index):
        """
        Read a image, corrupt it and return it
        """
        img_path = os.path.join(self.root_dir, self.image_name_list[index])
        image = Image.open(img_path).convert("RGB")

        if self.crop_size > 0:
            image = self._random_crop_to_size([image])[0]

        source_img_dict = self.corrupt_image(image)
        source_img_dict["image"] = tvF.to_tensor(source_img_dict["image"])

        if source_img_dict["use_mask"]:
            source_img_dict["mask"] = tvF.to_tensor(source_img_dict["mask"])

        if self.clean_target:
            target = tvF.to_tensor(image)
        else:
            _target_dict = self.corrupt_image(image)
            target = tvF.to_tensor(_target_dict["image"])
        return source_img_dict["image"], target

    def __len__(self):
        return len(self.image_name_list)
