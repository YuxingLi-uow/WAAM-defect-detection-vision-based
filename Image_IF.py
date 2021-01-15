from Image_Preprocess import *
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
import os
from torchvision import transforms, models
import torch
import torch.nn as nn
from MultiScale_Bstwts_Load import predict_model
import matplotlib.patches as pch

################################################################################################################
# FOLLOWING:                                                                                               #####
# TODO: Here we get the three layers patches from a image,                                                 #####
# then we need input every patch into the model                                                            #####
# Base on the predict results, we need to demonstrate it on the image, by square or rectangular marks      #####
# some filters or other methods required to acquire accurate results                                       #####
################################################################################################################

# This is the part we need to be considered to be integrated into the interface software
# patches shape: [patches_num, *final_size, 3], np.array
first_layer = (64, 64)
second_layer = (128, 128)
final_size = (400, 400)

# random select a image to check the detection result
patches_final, image_cropped_clahe, image_cropped_raw = interface_process(
    'D:\Python Projects\Monitoring\Image_Data_1\Image_raw\yothing_photo381.jpg',
    first_layer_size=first_layer, second_layer_size=second_layer, final_size=final_size)

# # convert channel last to channel first
# patches_final = np.einsum('ijk->kij',patches_final)

################################################################################################################
# FOLLOWING:                                                                                               #####
# Here we get the three layers patches from a image, (done)                                                #####
# TODO: then we need input every patch into the model and get the scores of every class(done)              #####
# Base on the predict results, we need to demonstrate it on the image, by square or rectangular marks      #####
# some filters or other methods required to acquire accurate results                                       #####
################################################################################################################

defect_class = {0: 'edge', 1: 'lack of gas', 2: 'pores', 3: 'unfused', 4: 'void'}
input_size = 224
batch_size = 16
patches_size = patches_final.shape[0]
scores = torch.zeros(patches_size, len(defect_class))

patches_inputs = torch.zeros(patches_size, 3, input_size, input_size)

# Define image transforms
data_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(input_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.1398, 0.2588, 0.2459], [0.1513, 0.1281, 0.1062])
                                     ])



# patches_dataset =torch.utils.data.TensorDataset(patches_inputs)
# patches_loaders = torch.utils.data.DataLoader(patches_dataset, batch_size=16, num_workers=4)

# preprocess the image patches using pytorch data transform
for i in range(patches_size):

    image = Image.fromarray(patches_final[i, :, :, :])
    patches_inputs[i, :, :, :] = data_transform(image)  # patches_inputs is a torch.tensor, which shape as [N, C, H, W]


# Import the ResNet18 neural network
model_ft = models.resnet18(pretrained=True)  # Use resnet as the transfer learning model
num_ftrs = model_ft.fc.in_features  # number of full connect layer features number of ResNet

# Add a new full connect layer to the last part of ResNet neural network
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, len(defect_class))  # Replace the full connect layer

# Load best model weights from .pth file
model_ft.load_state_dict(torch.load('Resnet18_zero_bstwts.pth'))

# Variable scores is used to store the scores of each patch in every class
# Variable preds is used to store the prediction class of every patch
for i in range(patches_size//batch_size + 1):

    if batch_size * (i + 1) > patches_size:
        scores[batch_size * i:, :] = predict_model(model_ft, patches_inputs[batch_size * i:, :, :, :])
    else:
        scores[batch_size * i : batch_size * (i + 1), :] = \
            predict_model(model_ft, patches_inputs[batch_size * i : batch_size * (i + 1), :, :, :])

# for inputs in patches_loaders:
#     scores = predict_model(model_ft, inputs)

torch.cuda.empty_cache()

_, preds = torch.max(scores, 1)  # preds: class number, tensor, shape as [N, ]


################################################################################################################
# FOLLOWING:                                                                                               #####
# Here we get the three layers patches from a image, (done)                                                #####
# then we need input every patch into the model and get the scores of every class (done)                   #####
# TODO: Base on the predict results, we need to demonstrate it on the image, by square or rectangular marks (done)
# some filters or other methods required to acquire accurate results                                       #####
################################################################################################################

# 用 cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) → None 来实现
# pt1 为矩形左上角，pt2 为矩形右下角，都为 (x, y) tuple 坐标

# define the rectangle color as following: {edge: white(白), lackgas: yellow(黄), pores: magenta(洋红), unfused: cyan(蓝绿), void: blue(蓝)}
# # BGR format for cv2
# color_class = {0: (255, 255, 255), 1: (0, 255, 255), 2: (255, 0, 255), 3: (255, 255, 0), 4: (255, 0, 0)}

# color format for matplotlib, {edge: white(白), lackgas: yellow(黄), pores: magenta(洋红), unfused: cyan(蓝绿), void: blue(蓝)}
color_class = {0: 'white', 1: 'yellow', 2: 'magenta', 3: 'cyan', 4: 'blue'}

# Create figure and axes
fig, ax = plt.subplots(1)
plt.figure(figsize=(12, 8))

# Display the image in the axes
ax.imshow(image_cropped_raw)

# Create the rectangle patch
row = (image_cropped_raw.shape[0] - second_layer[0]) // first_layer[0]
col = (image_cropped_raw.shape[1] - second_layer[0]) // first_layer[0]

# 这里的画 patch 的程序不对，先把板子，没问题的类别加到数据库里面，再改正这里的画 patch 程序。
for i, result in enumerate(preds):

    # filter bad detection results
    if result == 5:
        continue

    # elif result == 4:
    #     continue

    else:
        # calculate the position of marking rectangle
        patch_row = i // col
        patch_col = i % col

        left_corner_pt1 = (patch_col * first_layer[0], patch_row * first_layer[0])
        # right_corner_pt2 = (patch_col * first_layer[0] + first_layer[0], patch_row * first_layer[0] + first_layer[0])

    # # cv2.rectangle(image_cropped_raw, left_corner_pt1, right_corner_pt2, color_class[int(result)], 3)
    # cv2.rectangle(image_cropped_raw, left_corner_pt1, right_corner_pt2, color_class[int(result)], thickness=1)

    # Create rectangle patches
    rects = pch.Rectangle(left_corner_pt1, first_layer[0], first_layer[0], linewidth=1, edgecolor=color_class[int(result)], facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rects)

# # 设置窗口的尺寸可调整并保持窗口比例
# cv2.namedWindow('marked image', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
#
# # # resize窗口大小
# # cv2.namedWindow('marked image', 0)
# # cv2.resizeWindow("marked image", int(image_cropped_raw.shape[1] / 2), int(image_cropped_raw.shape[0] / 2))
#
# cv2.imshow('marked image', image_cropped_raw)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.show()





























