import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets



def generate_pseudo_labels(model, dataset, transform):
    model.eval()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    pseudo_labels = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.cuda()
            outputs = model(inputs)
            preds = outputs.detach().argmax(dim=1).cpu().numpy()

            for pred in preds:
                pred = pred.astype(np.uint8)
                pred = transforms.ToPILImage()(pred)
                pred = transforms.Resize((224, 224))(pred)
                pred = np.array(pred)
                pseudo_labels.append(transforms.ToTensor()(pred))

    return pseudo_labels
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class SemiVOCSegmentation(Dataset):
    def __init__(self, root, year='2012', image_set='train', pseudo_labels=None, transform=None, target_transform=None, download=False):
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.pseudo_labels = pseudo_labels

        base_dir = 'VOCdevkit/VOC{}'.format(year)
        image_dir = os.path.join(self.root, base_dir, 'JPEGImages')
        target_dir = os.path.join(self.root, base_dir, 'SegmentationClass')

        splits_dir = os.path.join(self.root, base_dir, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, image_set + '.txt')

        with open(os.path.join(split_f), "r") as f:
            self.ids = [x.strip() for x in f.readlines()]

        self.images = []
        self.targets = []
        for name in self.ids:
            img_path = os.path.join(image_dir, name + ".jpg")
            target_path = os.path.join(target_dir, name + ".png")
            self.images.append(img_path)
            self.targets.append(target_path)

        # 如果提供了伪标签，则添加到目标列表中
        if pseudo_labels is not None:
            self.targets += pseudo_labels

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)
