import numbers
import random

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates
from torchvision.transforms import transforms
from torchvision.transforms.functional import hflip, vflip


class RandomElastic(object):
    """Random Elastic transformation by CV2 method on image by alpha, sigma parameter.
        # you can refer to:  https://blog.csdn.net/qq_27261889/article/details/80720359
        # https://blog.csdn.net/maliang_1993/article/details/82020596
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html#scipy.ndimage.map_coordinates
    Args:
        alpha (float): alpha value for Elastic transformation, factor
        if alpha is 0, output is original whatever the sigma;
        if alpha is 1, output only depends on sigma parameter;
        if alpha < 1 or > 1, it zoom in or out the sigma's Relevant dx, dy.
        sigma (float): sigma value for Elastic transformation, should be \ in (0.05,0.1)
        mask (PIL Image) in __call__, if not assign, set None.
    """

    def __init__(self, alpha, sigma):
        assert isinstance(alpha, numbers.Number) and isinstance(sigma, numbers.Number), \
            "alpha and sigma should be a single number."
        assert 0.05 <= sigma <= 0.1, \
            "In pathological image, sigma should be in (0.05,0.1)"
        self.alpha = alpha
        self.sigma = sigma

    @staticmethod
    def RandomElasticCV2(img, alpha, sigma, mask=None):
        alpha = np.random.rand() * alpha
        sigma = np.random.rand() * (sigma - 0.05) + 0.05

        alpha = img.shape[1] * alpha
        sigma = img.shape[1] * sigma
        if mask is not None:
            mask = np.array(mask).astype(np.uint8)
            img = np.concatenate((img, mask[..., None]), axis=2)

        shape = img.shape

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        # dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        img = map_coordinates(img, indices, order=0, mode='reflect').reshape(shape)
        if mask is not None:
            return img[..., :3], img[..., 3]
        else:
            return img, None

    def __call__(self, img, mask=None):
        return self.RandomElasticCV2(img, self.alpha, self.sigma, mask)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(alpha value={0})'.format(self.alpha)
        format_string += ', sigma={0}'.format(self.sigma)
        format_string += ')'
        return format_string


class Transformation:
    def __init__(self, to_tensor=True):
        if to_tensor:
            self.to_tensor = transforms.ToTensor()
        else:
            self.to_tensor = None
        self.resize = transforms.Resize((256, 256))
        self.norm = transforms.Normalize((0.5355, 0.4852, 0.4441), std=(0.2667, 0.2588, 0.2667))
        self.color_jitter = transforms.ColorJitter(.3, .6, .2, .3)
        self.affine = transforms.RandomAffine(30, (.1, .1), (0.75, 1), 30)

        self.el = RandomElastic(1, 0.1)

    def transform(self, image, reference, target_mask=None):
        # image, target_mask = self.el(image, target_mask)

        if self.to_tensor is not None:
            image = self.to_tensor(image)
            reference = self.to_tensor(reference)

            if target_mask is not None:
                target_mask = self.to_tensor(target_mask)

        if image.max() > 1.:
            image = image.float()
            image /= 255
        if reference.max() > 1:
            reference = reference.float()
            reference /= 255

        if reference.shape[0] == 1:
            reference = reference.repeat((3, 1, 1))

        image = self.resize(image)
        reference = self.resize(reference)
        if target_mask is not None:
            target_mask = self.resize(target_mask)

        # image = self.norm(image)
        # reference = self.norm(reference)

        image = self.color_jitter(image)
        reference = self.color_jitter(reference)

        if random.random() > .5:
            image = hflip(image)

            if target_mask is not None:
                target_mask = hflip(target_mask)

        if random.random() > .5:
            reference = hflip(reference)

        if random.random() > .5:
            image = vflip(image)

            if target_mask is not None:
                target_mask = vflip(target_mask)

        if random.random() > .5:
            reference = vflip(reference)

        seed = random.randint(0, 10000)
        torch.manual_seed(seed)
        image = self.affine(image)

        if target_mask is not None:
            torch.manual_seed(seed)
            target_mask = self.affine(target_mask)

        reference = self.affine(reference)

        # image = image.unsqueeze(0)
        # reference = reference.unsqueeze(0)

        if target_mask is not None:
            # target_mask = target_mask.unsqueeze(0)
            # target_mask = target_mask[0]
            return image, target_mask, reference

        return image, reference

    def __call__(self, *args, **kwargs):
        return self.transform(*args)
