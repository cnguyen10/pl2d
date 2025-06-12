from typing import Any

import grain.python as grain

import numpy as np
import albumentations as A


INT_MAX = np.iinfo(np.int32).max


class Resize(grain.MapTransform):
    def __init__(self, resize_shape: tuple[int, int]) -> None:
        super().__init__()
        self.resize_shape = resize_shape

    def map(self, element: dict[str, Any]) -> dict[str, np.ndarray]:
        """
        """
        resize_fn = A.Resize(height=self.resize_shape[0], width=self.resize_shape[1])

        element['image'] = resize_fn(image=element['image'])['image']

        return element


class RandomCrop(grain.RandomMapTransform):
    def __init__(self, crop_size: tuple[int, int]) -> None:
        super().__init__()
        self.crop_size = crop_size

    def random_map(self, element: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        seed = rng.integers(low=0, high=INT_MAX)
        rand_crop = A.Compose(
            transforms=[
                A.RandomCrop(
                    height=self.crop_size[0],
                    width=self.crop_size[1]
                )
            ],
            seed=seed
        )
        element['image'] = rand_crop(image=element['image'])['image']

        return element


class RandomHorizontalFlip(grain.RandomMapTransform):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def random_map(self, element: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        seed = rng.integers(low=0, high=INT_MAX)
        random_hflip = A.Compose(
            transforms=[A.HorizontalFlip(p=self.p),],
            seed=seed
        )
        element['image'] = random_hflip(image=element['image'])['image']

        return element


class RandomVerticalFlip(grain.RandomMapTransform):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def random_map(self, element: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        seed = rng.integers(low=0, high=INT_MAX)
        random_hflip = A.Compose(
            transforms=[A.VerticalFlip(p=self.p),],
            seed=seed
        )
        element['image'] = random_hflip(image=element['image'])['image']

        return element


class ToRGB(grain.MapTransform):
    def map(self, element: dict[str, Any]) -> dict[str, Any]:
        """convert a gray-scale image to a color one
        """
        if len(element['image'].shape) == 2:
            to_rgb = A.ToRGB(p=1.0)
            element['image'] = to_rgb(image=element['image'])['image']
        
        return element


class ToFloat(grain.MapTransform):
    def map(self, element: dict[str, Any]) -> dict[str, Any]:
        """convert int image to float image
        """
        to_float_fn = A.ToFloat()
        element['image'] = to_float_fn(image=element['image'])['image']
        
        return element


class Normalize(grain.MapTransform):
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def map(self, element: dict[str, Any]) -> dict[str, Any]:
        """
        """
        normalize_fn = A.Normalize(mean=self.mean, std=self.std)

        element['image'] = normalize_fn(image=element['image'])['image']

        return element