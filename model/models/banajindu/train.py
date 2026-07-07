import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models.model.deeplab_v3_plus import DeepV3Plus
from utils import generate_pseudo_labels, SemiVOCSegmentation


# 设置随机数种子，以便实验的可重复性
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)

# 数据增强
transform = transforms.Compose([
    transforms.Resize((513, 513)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据加载
train_dataset = datasets.VOCSegmentation(root='./data', year='2012', image_set='train', download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

# 定义模型
model = DeepV3Plus(in_channels=3,n_classes=21,backbone="resnet50",pretrained=False).cuda()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练模型
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # 正向传播
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

    # 生成伪标签
    pseudo_labels = generate_pseudo_labels(model, train_dataset, transform)

    # 使用伪标签重新训练模型
    pseudo_dataset = SemiVOCSegmentation(root='./data', year='2012', image_set='train', pseudo_labels=pseudo_labels, download=True, transform=transform)
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    for i, (inputs, targets) in enumerate(pseudo_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # 正向传播
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, len(pseudo_loader), loss.item()))
