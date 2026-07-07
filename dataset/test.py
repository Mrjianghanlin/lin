import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log


# def test(model, loader, cfg,save_path):
#     model.eval()
#     intersection_meter = AverageMeter()
#     union_meter = AverageMeter()
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     with torch.no_grad():
#         for img, mask, id in loader:
#             img = img.cuda()
#
#             pred = model(img).argmax(dim=1)
#
#             intersection, union, target = \
#                 intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
#
#             intersection_meter.update(intersection)
#             union_meter.update(union)
#             # filename = valloader.dataset.ids[0] + '.tif'
#             filename = id[0].split('/')[-1]
#             save_file = os.path.join(save_path, filename)
#             pred_img = pred.cpu().numpy()[0]
#             pred_img = np.uint8(pred_img)
#             pred_img = np.squeeze(pred_img)
#             Image.fromarray(pred_img).save(save_file)
#
#     iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
#     mIoU = np.mean(iou_class)
#     return mIoU, iou_class #这份代码缺少一个计算总体Accuracy和每类的Accuracy
def test(model, loader, cfg, save_path):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    correct_pixels = 0
    total_pixels = 0
    class_correct_pixels = [0] * cfg['nclass']
    class_total_pixels = [0] * cfg['nclass']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        for img, mask, id in loader:
            img = img.cuda()

            pred = model(img).argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            intersection_meter.update(intersection)
            union_meter.update(union)
            # filename = valloader.dataset.ids[0] + '.tif'
            filename = id[0].split('/')[-1]
            save_file = os.path.join(save_path, filename)
            pred_img = pred.cpu().numpy()[0]
            pred_img = np.uint8(pred_img)
            pred_img = np.squeeze(pred_img)
            Image.fromarray(pred_img).save(save_file)

            # calculate accuracy
            correct_pixels += (pred == mask.cuda()).sum().item()
            total_pixels += mask.numel()
            for i in range(cfg['nclass']):
                class_correct_pixels[i] += ((pred == i) & (mask.cuda() == i)).sum().item()
                class_total_pixels[i] += (mask == i).sum().item()

    accuracy = correct_pixels / total_pixels
    class_accuracy = [class_correct_pixels[i] / class_total_pixels[i] for i in range(cfg['nclass'])]
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIoU = np.mean(iou_class)
    return mIoU, iou_class, accuracy, class_accuracy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test script for semantic segmentation model')
    parser.add_argument('--data-root', type=str, default='VOCdevkit_all/VOC2007')
    parser.add_argument('--model-path', type=str, default='logs/best_epoch_weights.pth')
    parser.add_argument('--save-path', type=str, default='predicted_images')

    args = parser.parse_args()

    # Set up the dataset and data loader
    valset = SemiDataset('pascal', args.data_root, 'val')
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    # print(id[0])
    print(valloader.dataset.ids[0])
    cfg = {
        'dataset': 'pascal',
        'data_root': 'VOCdevkit_all/VOC2007',
        'nclass': 6,
        'crop_size': 512,
        'pretrained': True,
        'epochs': 200,
        'batch_size': 10,
        'lr': 0.001,
        'lr_multi': 10.0,
        'criterion': {
            'name': 'CELoss',
            'kwargs': {
                'ignore_index': 255
            }
        },
        'conf_thresh': 0.95,
        'model': 'deeplabv3plus',
        'backbone': 'resnet18_ibn_b',
        'replace_stride_with_dilation': [False, False, True],
        'dilations': [6, 12, 18]
    }
    from models.model import DeepV3Plus, HRNet

    # model = DeepV3Plus(in_channels=3, n_classes=6, backbone="resnet50", pretrained=True)
    model = DeepLabV3Plus(cfg)
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()
    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    mIoU, iou_class, accuracy, class_accuracy = test(model, valloader,  cfg,args.save_path)
    # mIOU, iou_class = test(model, valloader, cfg)

    print('Mean IoU:', mIoU)
    print('Class-wise IoU:', iou_class)
    print('Class-wise class_accuracy:', class_accuracy)
    print(f"Mean Accuracy: {accuracy:.4f}")
