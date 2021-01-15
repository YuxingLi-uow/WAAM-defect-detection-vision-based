from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from EmailSend import *
import sys
import random

plt.ion()  # interactive mode(交互模式)

# # Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))
#
# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)
#
# imshow(out, title=[class_names[x] for x in classes])


def val_model(model):

    since = time.time()

    # Each epoch has validation and test phase
    for phase in ['val', 'test']:

        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        class_correct = list(0. for i in range(5))
        class_total = list(0. for i in range(5))

        count = 0
        running_corrects = 0

        # Iterate over data, number of batch size is set in the dataloader
        # Number of for loops is determined by dataset_size / batch_size
        for inputs, labels in dataloaders[phase]:

            # Transfer data to GPU
            inputs = inputs.to(device)  # inputs shape as [batch_size, C, H, W], here is [16, 3, 224, 224]
            labels = labels.to(device)  # labels shape as [16, ]

            inputs_size = inputs.shape[0]

            # Get the scores through the pre-trained ResNet
            outputs = model(inputs)  # score of image on every class, shape as [batch_size, class number], here is [16, 5]
            _, preds = torch.max(outputs, 1)  # Select the max value in every tensor, return the index, or label, shape as [16, ]

            # Record the correct number of image in every class
            c = (preds == labels.data).squeeze()

            running_corrects += torch.sum(preds == labels.data)

            for i in range(inputs_size):
                label = labels.data[i]
                class_correct[label] += c[i]
                class_total[label] += 1

            count += 1

        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        print('batch running count in {} epoch: {}'.format(phase, count))
        print('Acc: {:.4f} '.format(epoch_acc))

        for i in range(5):
            print('Accuracy of {} : {} / {} = {:.4f} %'.format(defect_class[i],
                                                               class_correct[i], class_total[i],
                                                               100 * class_correct[i].item() /
                                                               class_total[i]))
        print()

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

#
# # Visualize a few images
# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = trainset_mean.numpy()
#     std = trainset_std.numpy()
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)  # clip (limit) the values in an array ((0, 1) here)
#     # plt.imshow(inp)
#
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated
#
#
# # Visualize the model prediction
# def visualize_model(model, num_images=6):
#
#     was_training = model.training
#     model.eval()
#     image_so_far = 0
#     fig = plt.figure()
#
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#
#             for j in range(inputs.size()[0]):
#                 image_so_far += 1
#                 ax = plt.subplot(num_images // 2, 2, image_so_far)
#                 ax.axis('off')
#                 ax.set_title('predict: {}'.format(class_names[preds[j]]))
#                 imshow(inputs.cpu().data[j])
#
#                 if image_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)


def predict_model(model, images):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():

        model = model.to(device)

        model.eval()  # Set model to evaluate mode

        # Iterate over data, number of batch size is set in the dataloader
        # Number of for loops is determined by dataset_size / batch_size

        # Transfer data to GPU
        inputs = images.to(device)  # inputs shape as [batch_size, C, H, W], here is [16, 3, 224, 224]

        # inputs_size = inputs.shape[0]

        # Get the scores through the pre-trained ResNet
        outputs = model(inputs)  # score of image on every class, shape as [batch_size, class number], here is [16, 5]
        # # _, preds = torch.max(outputs, 1)  # Select the max value in every tensor, return the index, or label, shape as [16, ]

        return outputs


if __name__ == '__main__':

    ####################################################################################################################
    # Some Initializations                                                                                      ########
    ####################################################################################################################
    defect_class = {0: 'edge', 1: 'lack of gas', 2: 'pores', 3: 'unfused', 4: 'void'}

    # Constants or other hyperparameters
    num_trial = 1

    param = {'batch_size': 16}

    trainset_mean = torch.tensor([0.1398, 0.2588, 0.2459])
    trainset_std = torch.tensor([0.1513, 0.1281, 0.1062])

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(336),  # random crop a patch at random size, and resize it to 336x336
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),  # left and right overturn (图片左右翻转)
            transforms.ToTensor(),  # Normalize to [0.0, 1.0]
            transforms.Normalize([0.1398, 0.2588, 0.2459], [0.1513, 0.1281, 0.1062])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.1398, 0.2588, 0.2459], [0.1513, 0.1281, 0.1062])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.1398, 0.2588, 0.2459], [0.1513, 0.1281, 0.1062])
        ]),
    }

    ####################################################################################################################
    # Load and read dataset                                                                                     ########
    ####################################################################################################################

    data_dir = 'ZeroPadding/WaamLayerDataset'

    # 先join出dataset的位置，然后在ImageFolder中设置transform
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }

    # 读取dataset，同时设置batchsize, 用torch.utils.data.DataLoader,用于输入到 train 或者 val functions
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=param['batch_size'], shuffle=True, num_workers=4)
        for x in ['val', 'test']
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Some constants
    dataset_sizes = {x: len(image_datasets[x]) for x in ['val', 'test']}

    class_names = image_datasets['val'].classes

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ####################################################################################################################
    # Start validation and test                                                                                 ########
    # Load model hyperparameters, and validate                                                                  ########
    ####################################################################################################################

    # Import the ResNet18 neural network
    model_ft = models.resnet18(pretrained=True)  # Use resnet as the transfer learning model
    num_ftrs = model_ft.fc.in_features  # number of full connect layer features number of ResNet

    # Add a new full connect layer to the last part of ResNet neural network
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))  # Replace the full connect layer

    # Load best model weights from .pth file
    model_ft.load_state_dict(torch.load('Resnet18_zero_bstwts.pth'))
    model_ft = model_ft.to(device)

    # # Redirect all print results to LogTrain txt file
    # sys.stdout = open('LogTrain_ZeroPadding/LogTest{}.txt'.format(num_trial), 'a')

    # Evaluate and test
    val_model(model_ft)

    torch.cuda.empty_cache()
    # visualize_model(model_ft)
    # plt.imshow()

    # # Send training results by email
    # msg = open('LogTrain/LogTrain{}.txt'.format(num_trial)).read()
    #
    # credentials, service = get_credentials()
    #
    # testMessage = CreateMessage('yothinglee@gmail.com', 'yothinglee@gmail.com', subject, msg)
    #
    # testSend = SendMessage(service, 'me', testMessage)
    #
    # # Close txt file
    # sys.stdout.close()

    ####################################################################################################################

