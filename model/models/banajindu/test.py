import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# from model.deeplabv3plus import DeepLabV3Plus
from models.model.deeplab_v3_plus import DeepV3Plus

import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 遍历所有的图像和标签文件，并将它们存储在self.images和self.labels中
        for filename in os.listdir(root_dir):
            if filename.endswith(".png"):
                image_path = os.path.join(root_dir, filename)
                label_path = os.path.join(root_dir, filename.replace(".jpg", ".png"))

                self.images.append(image_path)
                self.labels.append(label_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.labels[idx]

        # 加载图像和标签文件
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        # 对图像和标签进行变换
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

        # 返回图像和标签
        return image, label

# 定义训练参数
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

# 加载数据集
train_dataset = CustomDataset("train")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 定义模型
model = DeepV3Plus(in_channels=3,n_classes=21,backbone="resnet50",pretrained=False).cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义有监督损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 使用有标签数据进行有监督训练
for epoch in range(NUM_EPOCHS):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, NUM_EPOCHS, i+1, len(train_loader), loss.item()))

# 加载无标签数据
unlabeled_dataset = CustomDataset("unlabeled")
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 使用有限的无标签数据进行半监督训练
for epoch in range(NUM_EPOCHS):
    for i, (inputs, _) in enumerate(unlabeled_loader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        mask = predicted.cpu().numpy()
        mask = torch.from_numpy(mask).to(device)
        outputs = model(inputs, mask)
        loss = criterion(outputs, mask)
        loss.backward()
        optimizer.step()
        print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, NUM_EPOCHS, i+1, len(unlabeled_loader), loss.item()))
