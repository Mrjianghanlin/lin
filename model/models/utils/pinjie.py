import os
import numpy as np
from osgeo import gdal

# 定义图像路径
# image_file = r'C:\Users\wang\Desktop\data\yuanwenjian\image\8_8_bands.tif'
image_file = r'C:\Users\wang\Desktop\Potsdam\test\images\top_potsdam_7_10_gray_mask.png'

# 定义裁剪后的图像的大小和步长
crop_size_x = 512
crop_size_y = 512
step_x = 512
step_y = 512

# 打开图像tif文件，获取其大小和数据类型
image_dataset = gdal.Open(image_file)
image_size_x = image_dataset.RasterXSize
image_size_y = image_dataset.RasterYSize
data_type = image_dataset.GetRasterBand(1).DataType

# 创建输出文件夹
# output_folder = r'C:\Users\wang\Desktop\VOCdevkit\VOC2007\JPEGImages'

# output_folder = r'C:\Users\wang\Desktop\VOCdevkit\VOC2007\JPEGImages'
output_folder = r'C:\Users\wang\Desktop\VOCdevkit\VOC2007\SegmentationClass'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 将图像裁剪，并将它们分开存放
for y in range(0, image_size_y - crop_size_y + 1, step_y):
    for x in range(0, image_size_x - crop_size_x + 1, step_x):
        output_image_file = os.path.join(output_folder, f'image_{x}_{y}_7_10.png')
        driver = gdal.GetDriverByName('GTiff')
        output_image_dataset = driver.Create(output_image_file, crop_size_x, crop_size_y, image_dataset.RasterCount, data_type)
        output_image_dataset.SetGeoTransform((x*image_dataset.GetGeoTransform()[1], image_dataset.GetGeoTransform()[1], 0, y*image_dataset.GetGeoTransform()[5], 0, image_dataset.GetGeoTransform()[5]))
        output_image_dataset.SetProjection(image_dataset.GetProjection())
        for i in range(1, image_dataset.RasterCount + 1):
            output_image_dataset.GetRasterBand(i).WriteArray(image_dataset.GetRasterBand(i).ReadAsArray(x, y, crop_size_x, crop_size_y))



# # 合并裁剪后的图像
# merged_image_file = os.path.join(output_folder, 'merged_image.tif')
# driver = gdal.GetDriverByName('GTiff')
# merged_image_dataset = driver.Create(merged_image_file, image_size_x, image_size_y, image_dataset.RasterCount, data_type)
# merged_image_dataset.SetGeoTransform(image_dataset.GetGeoTransform())
# merged_image_dataset.SetProjection(image_dataset.GetProjection())
# for y in range(0, image_size_y - crop_size_y + 1, step_y):
#     for x in range(0, image_size_x - crop_size_x + 1, step_x):
#         input_image_file = os.path.join(output_folder, f'image_{x}_{y}_8.tif')
#         input_image_dataset = gdal.Open(input_image_file)
#         for i in range(1, image_dataset.RasterCount + 1):
#             merged_image_dataset.GetRasterBand(i).WriteArray(input_image_dataset.GetRasterBand(i).ReadAsArray(), x, y)
#
