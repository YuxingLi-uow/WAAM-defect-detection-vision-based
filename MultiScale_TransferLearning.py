from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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


def train_model(model, criterion, optimizer, scheduler, defect_classes, num_epochs=25):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())  # Initialize the best model, copy the pretrained resnet
    best_acc = 0.0
    best_acc_sum = 0.0

    # acc_iter_history_train = []
    # acc_iter_history_val = []


    for epoch in range(num_epochs):  # Define how much of epoches run in the training (set manually)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # This is a place need to be modified if you use a new dataset
        class_correct = list(0. for i in range(5))
        class_total = list(0. for i in range(5))
        epoch_acc_sum = 0.0

        confusion_matrix = {defect_classes[0]: [0, 0, 0, 0, 0],
                            defect_classes[1]: [0, 0, 0, 0, 0],
                            defect_classes[2]: [0, 0, 0, 0, 0],
                            defect_classes[3]: [0, 0, 0, 0, 0],
                            defect_classes[4]: [0, 0, 0, 0, 0]}

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            count = 0

            # Iterate over data, number of batch size is set in the dataloader
            # Number of for loops is determined by dataset_size / batch_size
            # For example here, batch_size = 4, dataset_size['train'] = 244, 61 training for loop
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # Transfer data to GPU
                labels = labels.to(device)

                inputs_size = inputs.shape[0]

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                # NOTE: this command has no effect if we set param.requires_grad = False previously
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # Get the outputs through the pre-trained resnet
                    _, preds = torch.max(outputs, 1)  # Select the max value in every tensor, return the index, or label

                    loss = criterion(outputs, labels)  # Use this criterion calculate the loss of output and label

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # # add for acc_iter_history_train
                        # acc_iter_history_train.append(100 * torch.sum(preds == labels.data).data.double().cpu() / inputs_size)

                        loss.backward()
                        optimizer.step()

                    # Record the correct number of image in every class
                    if phase == 'val':
                        c = (preds == labels.data).squeeze()
                        # # add for acc_iter_history_val
                        # acc_iter_history_val.append(100 * torch.sum(preds == labels.data).data.double().cpu() / inputs_size)

                        for i in range(inputs_size):
                            label = labels.data[i]
                            pred = preds.data[i]
                            # count the correct prediction
                            class_correct[label] += c[i]
                            # count the total sample
                            class_total[label] += 1
                            # count for the confusion matrix
                            confusion_matrix[defect_classes[label.item()]][pred.item()] += 1
                            # add for acc_iter_history_val


                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                count += 1

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # print('batch running count in this epoch: {}'.format(count))

            # print val batch class correct in every class
            if phase == 'val':
                for i in range(5):
                    print('Accuracy of {} : {} / {} = {:.4f} %'.format(defect_class[i],
                                                                       class_correct[i], class_total[i],
                                                                       100 * class_correct[i].item() /
                                                                       class_total[i]))
                    epoch_acc_sum += 100 * class_correct[i].item() /class_total[i]
                else:
                    print('Sum of Accuracy: {:.4f} %'.format(epoch_acc_sum))
                    print()
                    print('Confusion Matrix: {}'.format(confusion_matrix))


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # deep copy the model
            if phase == 'val' and epoch_acc_sum > best_acc_sum:
                best_acc_sum = epoch_acc_sum
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f} %'.format(best_acc * 100))
    print('Best val Acc Sum: {:.4f} %'.format(best_acc_sum))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # return model, acc_iter_history_train, acc_iter_history_val
    return model


# Visualize a few images
def imshow(inp, title=None):
    """Imshow for Tensor."""

    inp = inp.transpose((1, 2, 0))
    mean = trainset_mean.numpy()
    std = trainset_std.numpy()
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)  # clip (limit) the values in an array ((0, 1) here)
    # plt.imshow(inp)

    if title is not None:
        plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated
    return inp

# Visualize the model prediction
def visualize_model(model, num_images=6):

    was_training = model.training
    model.eval()
    fig = plt.figure(figsize=(8, 8))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            image_so_far = 0
            inputs = inputs.to(device)
            labels = labels.to(device)

            # count pores number in this val set
            pores_num = torch.sum(labels==3).item()
            if pores_num > 2:
                num_images = pores_num // 2 + 1
                colum = 2
            else:
                num_images = 1
                colum = pores_num

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            pors_indx = (labels==3).nonzero()

            for j in range(pores_num):
                image_so_far += 1
                ax = plt.subplot(num_images, colum, image_so_far)
                im = np.asarray(inputs[pors_indx[j]].cpu())[0]
                # im = im.transpose((1,2,0))
                # ax.imshow(im)
                ax.axis('off')
                ax.set_title('predict: {}'.format(class_names[preds[pors_indx[j]]]))
                inp = imshow(im)
                plt.imshow(inp)

            plt.show()
                # if image_so_far == num_images * colum:
                #
                #     model.train(mode=was_training)
                #     # return

        model.train(mode=was_training)


if __name__ == '__main__':
    ####################################################################################################################

    defect_class = {0: 'edge', 1: 'good', 2: 'plate', 3: 'pores', 4: 'unfuse'}

    # Constants or other hyperparameters
    num_trial = 5
    # lr_tun = (np.random.random(3, ) * 2 + 6) * 1e-2  # 0.06 -- 0.08
    # lr_tun = lr_tun.tolist()
    # lr_tun.sort()
    # lr_tun = [0.0006, 0.00075, 0.0008]
    # lr_tun = [0.00689141944945379]
    # lr_tun = [0.007148321308663864]
    lr_tun = [0.06630369131902578]  # learning rate
    # moment = 0.9

    num_epoch = 30  # number of epoch

    # l2_coe = np.random.random(3, ) / 50  # 0 -- 0.02
    # l2_coe = l2_coe.tolist()
    # l2_coe.sort()

    # l2_coe = [0.001, 0.005, 0.01]

    # l2_coe = [0.001105477691618042]
    # l2_coe = [0.014546764623332382]
    l2_coe = [0.010973581237306296]  # L2 regularization coefficient
    param = {'step_size': 5, 'gamma': 0.4, 'batch_size': 64, 'momentum': 0.9}

    trainset_mean = torch.tensor([0.4653, 0.4658, 0.5293])
    trainset_std = torch.tensor([0.0935, 0.0985, 0.0946])

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomCrop(336),  # random crop a patch at size 336x336
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),  # left and right overturn (图片左右翻转)
            transforms.ToTensor(),  # Normalize to [0.0, 1.0]
            transforms.Normalize([0.4653, 0.4658, 0.5293], [0.0935, 0.0985, 0.0946])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4653, 0.4658, 0.5293], [0.0935, 0.0985, 0.0946])
        ]),
        'test': transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4653, 0.4658, 0.5293], [0.0935, 0.0985, 0.0946])
        ]),
    }

    # data_dir = 'ZeroPadding/WaamLayerDataset'
    data_dir = 'DatasetRGBraw'

    # 先join出dataset的位置，然后在ImageFolder中设置transform
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # # Redirect all print results to LogTrain txt file
    # sys.stdout = open('LogTrain_RGB/Logtrain{}.txt'.format(num_trial), 'a')

    for lr in lr_tun:

        param['lr'] = lr

        for wt_dcay in l2_coe:

            param['weight_decay'] = wt_dcay

            # 读取dataset，同时设置batchsize, 用torch.utils.data.DataLoader
            dataloaders = {
                x: torch.utils.data.DataLoader(image_datasets[x], batch_size=param['batch_size'], shuffle=True, num_workers=4)
                for x in ['train', 'val']
            }

            # This training will update and backward all parameters over the whole resnet.
            # Finetuning the convet
            model_ft = models.resnet18(pretrained=True)  # Use resnet as the transfer learning model
            num_ftrs = model_ft.fc.in_features  # number of features to resnet??

            # Here the size of each output sample is set to 5
            # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
            model_ft.fc = nn.Linear(num_ftrs, len(class_names))  # Replace the full connect layer
            model_ft = model_ft.to(device)

            # Evaluate the loss of softmax prediction result with ground-truth label
            criterion = nn.CrossEntropyLoss()

            # Optimize the neural network
            # Observe that all parameters are being optimized
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=param['lr'], momentum=param['momentum'], weight_decay=param['weight_decay'])

            # Decay LR by a factor of 0.1 every 7 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=param['step_size'], gamma=param['gamma'])

            # # Redirect all print results to LogTrain txt file
            # sys.stdout = open('LogTrain/LogTrain{}.txt'.format(num_trial), 'a')

            # This is the preparing messages
            print('\n-------------------------------------------\n')
            print('L2 regularization in the SGD optimizer, set weight_decay = {}'.format(param['weight_decay']))
            print('\n-------------------------------------------\n')
            print('Epoch number is {}, the batch size is {}'.format(num_epoch, param['batch_size']))
            # print('Step size is {}, gamma is {}'.format(param['step_size'], param['gamma']))
            # print('The initial learning rate to lr={} \n'.format(param['lr']))
            print('-------------------------------------------\n')
            print('criterion = nn.CrossEntropyLoss()')
            print('optimizer_ft = optim.SGD(model_ft.parameters(), lr={}, momentum={})'.format(param['lr'], param['momentum']))
            print('exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size={}, gamma={})'
                  .format(param['step_size'], param['gamma']))
            print('\n-------------------------------------------\n')

            subject = '{}th train and val result: resnet18, no freeze previous layers'.format(num_trial)
            print('-------------------------------------------\n')
            print(subject)
            print()

            # Train and evaluate
            model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, defect_class, num_epochs=num_epoch)
            # model_ft, acc_iter_history_train, acc_iter_history_val = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, defect_class, num_epochs=num_epoch)

            # # Save weights of the model, should be commented while training, when the parameter determined, uncommented.
            # torch.save(model_ft.state_dict(), 'Resnet18_RGB_bstwts.pth')

            torch.cuda.empty_cache()
            # visualize_model(model_ft)
            # plt.imshow()

    # # Send training results by email
    # msg = open('LogTrain_ZeroPadding/LogTrain{}.txt'.format(num_trial)).read()
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

    ####################################################################################################################
    # # This training will FREEZE all the network except the final layer.
    # # Finetuning the convet
    # model_conv = torchvision.models.resnet18(pretrained=True)
    # # This is the main freeze structure for previous weights
    # for param in model_conv.parameters():
    #     param.requires_grad = False
    #
    # # Parameters of newly constructed modules have requires_grad=True by default
    # num_ftrs = model_conv.fc.in_features
    # model_conv.fc = nn.Linear(num_ftrs, 2)
    #
    # model_conv = model_conv.to(device)
    #
    # criterion = nn.CrossEntropyLoss()
    #
    # # Observe that only parameters of final layer are being optimized as
    # # opposed to before.
    # optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    #
    # # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    #
    # model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)
    #
    # visualize_model(model_conv)
    #
    # plt.ioff()
    # plt.show()
    ####################################################################################################################
