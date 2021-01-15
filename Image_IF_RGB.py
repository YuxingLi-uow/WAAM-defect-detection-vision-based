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
import matplotlib.collections as clct

################################################################################################################
# FOLLOWING:                                                                                               #####
# TODO: Here we calculate the center crop image size and patch it                                          #####
# then we need input every patch into the model                                                            #####
# Base on the predict results, we need to demonstrate it on the image, by square or rectangular marks      #####
# some filters or other methods required to acquire accurate results                                       #####
################################################################################################################

# This is the part we need to be considered to be integrated into the interface software
# patches shape: [patches_num, *final_size, 3], np.array
patch_base = (64, 64, 3)
patch_size = (64, 64)
image_path = 'D:\Python Projects\Monitoring\Image_Data_1\Image_raw\yothing_photo111.jpg'
image = np.asarray(Image.open(image_path))
cropped_size = crop_size_cal_RGB(image, patch_size)
image = np.asarray(image_center_crop(image, cropped_size))

patches, patches_row, patches_column, patch_width, patch_height = image_patching_RGB(image, patch_base, patch_base)
patches = patches.reshape(patches_row * patches_column, patch_width, patch_height, 3)

# # convert channel last to channel first
# patches_final = np.einsum('ijk->kij',patches_final)

################################################################################################################
# FOLLOWING:                                                                                               #####
# Here we get the three layers patches from a image, (done)                                                #####
# TODO: then we need input every patch into the model and get the scores of every class(done)              #####
# Base on the predict results, we need to demonstrate it on the image, by square or rectangular marks      #####
# some filters or other methods required to acquire accurate results                                       #####
################################################################################################################

defect_class = {0: 'edge', 1: 'good', 2: 'plate', 3: 'pores', 4: 'unfuse'}
input_size = 224
batch_size = 64
patches_size = patches.shape[0]
scores = torch.zeros(patches_size, len(defect_class))

patches_inputs = torch.zeros(patches_size, 3, input_size, input_size)

# Define image transforms
data_transform = transforms.Compose([transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.4653, 0.4658, 0.5293], [0.0935, 0.0985, 0.0946])
                                     ])



# patches_dataset =torch.utils.data.TensorDataset(patches_inputs)
# patches_loaders = torch.utils.data.DataLoader(patches_dataset, batch_size=16, num_workers=4)

# preprocess the image patches using pytorch data transform
for i in range(patches_size):

    img = Image.fromarray(patches[i, :, :, :])
    patches_inputs[i, :, :, :] = data_transform(img)  # patches_inputs is a torch.tensor, which shape as [N, C, H, W]


# Import the ResNet18 neural network
model_ft = models.resnet18(pretrained=True)  # Use resnet as the transfer learning model
num_ftrs = model_ft.fc.in_features  # number of full connect layer features number of ResNet

# Add a new full connect layer to the last part of ResNet neural network
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, len(defect_class))  # Replace the full connect layer

# Load best model weights from .pth file
model_ft.load_state_dict(torch.load('Resnet18_RGB_bstwts.pth'))

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

# # Save the prediction scores to xlsx file
# data_file_name = image_path.rsplit('_', 1)[-1].rsplit('.', 1)[0] + ' rgb score data.xlsx'
# tensor_data_save(scores.data.cpu(), columns=['edge', 'good', 'plate', 'pores', 'unfuse'], file_name=data_file_name)
# tensor_data_save(preds.data.cpu(), columns=['edge', 'good', 'plate', 'pores', 'unfuse'], file_name='scores data.xlsx')

################################################################################################################
# FOLLOWING:                                                                                               #####
# Here we get the three layers patches from a image, (done)                                                #####
# then we need input every patch into the model and get the scores of every class (done)                   #####
# TODO: Base on the predict results, we need to demonstrate it on the image, by square or rectangular marks (done)
# some filters or other methods required to acquire accurate results                                       #####
################################################################################################################

# 用 cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) → None 来实现
# pt1 为矩形左上角，pt2 为矩形右下角，都为 (x, y) tuple 坐标

# # color format for matplotlib, {edge: white(白), good: yellow(黄), plate: magenta(洋红), pores: cyan(蓝绿), unfuse: blue(蓝)}
# color_class = {0: 'white', 1: 'yellow', 2: 'magenta', 3: 'cyan', 4: 'blue'}
# color format for matplotlib, {edge: yellow(黄), good: green(绿), plate: white(白), pores: cyan(蓝绿), unfuse: blue(蓝)}
color_class = {0: 'yellow', 1: 'green', 2: 'white', 3: 'cyan', 4: 'blue'}

# Create figure and axes
fig, ax = plt.subplots(1)
plt.figure(figsize=(12, 8))
plt.axis('off')

# Display the image in the axes
ax.imshow(image)
# ax.axis('off')

# Create the rectangle patch
row = image.shape[0]  // patch_size[0]
col = image.shape[1]  // patch_size[1]

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

        left_corner_pt1 = (patch_col * patch_size[0], patch_row * patch_size[0])
        # right_corner_pt2 = (patch_col * patch_size[0] + patch_size[0], patch_row * patch_size[0] + patch_size[0])

    # # cv2.rectangle(image_cropped_raw, left_corner_pt1, right_corner_pt2, color_class[int(result)], 3)
    # cv2.rectangle(image_cropped_raw, left_corner_pt1, right_corner_pt2, color_class[int(result)], thickness=1)

    # Create rectangle patches
    rects = pch.Rectangle(left_corner_pt1, patch_size[0], patch_size[0], linewidth=0, linestyle='--',
                          color=color_class[int(result)], alpha=0.35)

    ax.add_patch(rects)


#     rects = pch.Rectangle(left_corner_pt1, patch_size[0], patch_size[0], facecolor=color_class[int(result)])
#     boxes.append(rects)
#
# pachs = clct.PathCollection(boxes, alpha=0.5)
# # Add the collection to the Axes
# ax.add_collection(pachs)

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





























