import os
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor, PairCenterCrop
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = path

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(256),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(path, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    transform = PairCompose(
        [
            PairCenterCrop(128),
            PairToTensor()
        ]
    )
    dataloader = DataLoader(
        DeblurDataset(path, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.lq_dir = os.path.join(image_dir, 'Lq')
        self.gt_dir = os.path.join(image_dir, 'Gt')
        self.image_list = os.listdir(self.lq_dir)  # 只列 Lq 文件
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.lq_dir, self.image_list[idx])
        label_path = os.path.join(self.gt_dir, self.image_list[idx])
        
        image = Image.open(image_path)
        label = Image.open(label_path)

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)

        if self.is_test:
            return image, label, self.image_list[idx]

        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
