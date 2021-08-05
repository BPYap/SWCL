import random

import numpy as np
import torch
import torchvision.transforms as transforms

from retinal_fundus_encoder.transforms_utils import GaussianBlur, Solarize

EYEPACS_DATASET_MEAN = (0.1633, 0.2259, 0.3219)
EYEPACS_DATASET_STD = (0.1756, 0.2195, 0.3038)


class TransformsDataset(torch.utils.data.Dataset):
    """ Dataset wrapper for image transformation. Useful for wrapping ConcatDataset, Subset or other Dataset object.
    """

    def __init__(self, dataset, image_transform, target_transform=None):
        self.dataset = dataset
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __bool__(self):
        return self.dataset is not None

    def __getitem__(self, index):
        sample = self.dataset[index]
        if isinstance(sample, tuple):
            image, target = sample
        else:
            image = sample
            target = -1  # used to denote unlabeled dataset

        if self.target_transform:
            if isinstance(image, list):
                raise TypeError("Image list is not supported when `target_transform` is not None.")

            # same seed to ensure random transformations are applied consistently on both image and target
            seed = random.randint(0, 2147483647)

            random.seed(seed)
            image = self.image_transform(image)

            if isinstance(target, dict):  # multi-task targets
                for name, target_ in target.items():
                    random.seed(seed)
                    target[name] = self.target_transform(target_)
            else:
                random.seed(seed)
                target = self.target_transform(target)
        else:
            image = [self.image_transform(img) for img in image] if isinstance(image, list) \
                else self.image_transform(image)

        return image, target

    def __len__(self):
        return len(self.dataset)


DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor()
])

TRANSFORM_TRAIN_CONTRASTIVE = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.RandomApply([GaussianBlur(23, sigma=(0.1, 2.0))], p=0.5),
    transforms.RandomApply([Solarize(threshold=0.5)], p=0.1),
    transforms.Normalize(EYEPACS_DATASET_MEAN, EYEPACS_DATASET_STD)
])

TRANSFORM_TRAIN_CLS = transforms.Compose([
    # transforms.Resize(350),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(EYEPACS_DATASET_MEAN, EYEPACS_DATASET_STD)
])

TRANSFORM_EVAL_CLS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(EYEPACS_DATASET_MEAN, EYEPACS_DATASET_STD)
])


class ToMask:
    def __call__(self, image):
        if image.mode == 'P':
            np_image = np.array(image)
            if np_image.ndim == 2:
                np_image = np_image[:, :, None]

            tensor = torch.from_numpy(np_image.transpose((2, 0, 1)))
        else:
            tensor = transforms.functional.to_tensor(image)

        return tensor.squeeze().long()


TRANSFORM_TRAIN_SEG = transforms.Compose([
    transforms.RandomCrop(384),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(EYEPACS_DATASET_MEAN, EYEPACS_DATASET_STD)
])

TRANSFORM_TRAIN_SEG_MASK = transforms.Compose(
    TRANSFORM_TRAIN_SEG.transforms[0:2] + [ToMask()]
)

TRANSFORM_EVAL_SEG = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(EYEPACS_DATASET_MEAN, EYEPACS_DATASET_STD)
])

TRANSFORM_EVAL_SEG_MASK = ToMask()
