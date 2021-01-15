import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import skimage
import random
from torchvision import transforms
import PIL.Image as Image
import os
import torch
import pandas as pd


def image_clahe(img, cliplimit=1.0, gridsize=(3, 3)):

    """
    Contrast Limited Adaptive Histogram Equalization for a single dark image

    :param image: Single dark image in BGR color type
    :param cliplimit: Threshold for contrast limiting
    :param gridsize: Size of grid for histogram equalization. Input image will be divided into
    equally sized rectangular tiles. tileGridSize defines the number of tiles in row and column.

    :return: Return image after CLAHE
    """

    # img_path = image_path
    # img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # image_nois = cv2.GaussianBlur(image_gray, (5, 5), 1)
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=gridsize)
    img_clahe = clahe.apply(img_gray)

    return img_clahe


def image_read_clahe(dark_image_path, image_save=False):

    """
    Read dark images from the director path, and process the dark image with CLAHE algorithm

    :param dark_image_path: Path of all dark images
    :param image_save: Determine if the CLAHE image is saved. If TRUE, the CLAHE image
    is saved to the same folder as dark images

    :return: None
    """

    img_path = glob.glob(dark_image_path + '/*.jpg')

    for path in img_path:

        print('Processing', path)

        img = cv2.imread(path)
        img_clahe = image_clahe(img, cliplimit=1.0, gridsize=(3, 3))

        if image_save:
            # extract image save file name
            img_title = path.rsplit('.', 1)[0] + '_clahe.jpg'
            plt.imsave(img_title, img_clahe, cmap='gray')
        else:
            plt.figure(figsize=(12, 8))
            plt.imshow(img_clahe, cmap='gray')
            plt.show()

        # return img_clahe


def image_padding(image_pad, padding_size=(32, 32, 32, 32), value=0):

    """
    Border padding of image with cv2 constant padding type.

    :param image_pad: single image require board padding
    :param padding_size: padding size, tuple required as sequence: top, bottom, left, right padding size

    :return: image after padding with border zero
    """

    # pad_top, pad_bottom, pad_left, pad_right = padding_size
    # img_reflect = cv2.copyMakeBorder(image_pad, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
    img_zero = cv2.copyMakeBorder(image_pad, *padding_size, cv2.BORDER_CONSTANT, value=value)

    return img_zero


def image_padding_reflect(image_pad, padding_size=(32, 32, 32, 32)):

    """
    Border padding of image with cv2 reflect padding type.

    :param image_pad: single image require board padding
    :param padding_size: padding size, tuple required as sequence: top, bottom, left, right padding size

    :return: image after padding with border reflection
    """

    # pad_top, pad_bottom, pad_left, pad_right = padding_size
    # img_reflect = cv2.copyMakeBorder(image_pad, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
    img_reflect = cv2.copyMakeBorder(image_pad, *padding_size, cv2.BORDER_REFLECT)

    return img_reflect


def image_padding_square(image, padding_size=0, value=0):

    """
    This function padding a rectangular image to square image, with constant bordering.

    :param image: rectangular image
    :param padding_size: padding size on the short image side
    :param value: padding constant value

    :return: square image
    """

    h, w = image.shape

    if w > h:

        patches_3 = cv2.copyMakeBorder(image, padding_size, padding_size, 0, 0, cv2.BORDER_CONSTANT,
                                       value=value)

    else:

        patches_3 = cv2.copyMakeBorder(image, 0, 0, padding_size, padding_size, cv2.BORDER_CONSTANT,
                                       value=value)

    return patches_3


def image_center_crop(image, crop_size):

    """
    :param image: image need to be center cropped.
    :param crop_size: center crop size, if not provided as (w, h), a square crop (cropsize, cropsize) is cropped.

    :return: image after cropped.
    """

    if type(image) is np.ndarray:
        image = Image.fromarray(np.uint8(image))

    # type(image) is PIL.JpegImagePlugin.JpegImageFile:
    try:
        image_transform = transforms.CenterCrop(crop_size)
        image_crop = image_transform(image)
        return image_crop

    except TypeError:
        print('Image type Error! Image type must be numpy array or PIL Image!')


def crop_size_cal(image, size_base):

    """
    Center crop size calculator, calculate the closest crop size of based size.

    :param image: image required to be croped, has attribute size().
    :param size_base: the smallest center crop size.

    :return: Crop size of image, as (w, h) shape.
    """

    image_w, image_h = image.shape
    w_base, h_base = size_base
    crop_size = ((image_w // w_base) * w_base, (image_h // h_base) * h_base)

    return crop_size


def crop_size_cal_RGB(image, size_base):

    """
    Center crop size calculator, calculate the closest crop size of based size.

    :param image: image required to be croped, has attribute size().
    :param size_base: the smallest center crop size.

    :return: Crop size of image, as (w, h) shape.
    """

    image_w, image_h = image.shape[0:2]
    w_base, h_base = size_base
    crop_size = ((image_w // w_base) * w_base, (image_h // h_base) * h_base)

    return crop_size


def image_patching_gray(image_patch, patching_size=(32, 32), patching_stride=(1, 1)):

    """
    2D image patching, with grayscale image only

    :param image_patch: image need to be patched
    :param patching_size: size of image patched
    :param patching_stride: stride of every patching

    :return: numpy array of patch images, and the shape of the patch array, patches_column, patch_width, patch_height
    """

    # img_h, img_w, img_c = image_patch.shape
    patches = skimage.util.view_as_windows(image_patch, patching_size, patching_stride)
    patches_row, patches_column, patch_width, patch_height = patches.shape
    return patches, patches_row, patches_column, patch_width, patch_height


def image_patching_RGB(image_patch, patching_size=(32, 32, 3), patching_stride=(1, 1, 3)):

    """
    2D image patching, with grayscale image only

    :param image_patch: image need to be patched
    :param patching_size: size of image patched
    :param patching_stride: stride of every patching

    :return: numpy array of patch images, and the shape of the patch array, patches_column, patch_width, patch_height
    """

    # img_h, img_w, img_c = image_patch.shape
    patches = skimage.util.view_as_windows(image_patch, patching_size, patching_stride)
    patches_row, patches_column, _, patch_width, patch_height, patches_deep = patches.shape
    return patches, patches_row, patches_column, patch_width, patch_height


def image_3layer_comb(image_layer_1, image_layer_2, image_layer_3, size):

    if not image_layer_1.shape == size:
        image_layer_1 = cv2.resize(image_layer_1, size)

    if not image_layer_2.shape == size:
        image_layer_2 = cv2.resize(image_layer_2, size)

    if not image_layer_3.shape == size:
        image_layer_3 = cv2.resize(image_layer_3, size)

    multi_image = cv2.merge([image_layer_1, image_layer_2, image_layer_3])
    # multi_image_gray = cv2.cvtColor(multi_image, cv2.COLOR_RGB2GRAY)

    return multi_image


def tensor_data_save(data, columns, file_name):
    data = np.asarray(data).transpose()
    df = pd.DataFrame(data, columns).transpose()
    df.to_excel(file_name)
    print('data saved!')
    # print(df)



def interface_process(image_path, first_layer_size=(64, 64), second_layer_size=(128, 128), final_size=(400, 400)):

    """
    This function is used to process a single image to 3 layers multi-scale image.

    :param image_path: Image director path, jpg or other forms.
    :param first_layer_size: first layer size of multi-scale image.
    :param second_layer_size: second layer size of multi-scale image.
    :param final_size: final size of scaled multi-scale image.

    :return: patches_final: numpy array of patches, shape as (N, final_size, 3)
             image_crop_np: clahed image after cropped, ready for future marks
             image_crop: raw image after cropped, ready for future marks
    """

    raw_image_np = np.asarray(Image.open(image_path))  # convert PIL Image to numpy array, (h, w, 3)
    # # uncomment this line to convert this numpy to PIL.Image
    # im =  PIL.Image.fromarray(numpy.uint8(raw_image))

    # process image using clahe method
    image_cla = image_clahe(raw_image_np)

    # calculate the center crop size of image
    crop_size = crop_size_cal(image_cla, second_layer_size)

    # Center crop the clahe image with crop_size
    # Convert to numpy array
    image_crop_np = np.asarray(image_center_crop(image_cla, crop_size))

    plt.imsave('tmp.jpg', image_crop_np, cmap='gray')
    image_crop_np = cv2.cvtColor(cv2.imread('tmp.jpg'), cv2.COLOR_BGR2GRAY)
    os.remove('tmp.jpg')

    # Patching to 128 x 128 pixels
    patches_2, patches_row, patches_column, patch_width, patch_height = \
        image_patching_gray(image_crop_np, patching_size=second_layer_size, patching_stride=first_layer_size)

    # reshape, 2nd layer patches matrix, (N, 128, 128)
    patches_2 = patches_2.reshape(patches_column * patches_row, patch_width, patch_height)

    # Final patches matrix
    patches_final = np.zeros((patches_2.shape[0], *final_size, 3))

    # 3rd layer preparing
    h, w = image_crop_np.shape
    padding_size = abs(int((w - h) / 2))

    patches_3 = image_padding_square(image_crop_np, padding_size=padding_size, value=0)

    # save all patches
    for i in range(patches_2.shape[0]):

        # center crop to 64 x 64 pixels
        l1 = np.asarray(image_center_crop(patches_2[i, :, :], first_layer_size))

        patches_final[i, :, :, :]= image_3layer_comb(l1, patches_2[i, :, :], patches_3, final_size)
    else:
        patches_final = patches_final.astype('uint8')

    # return patches_final, image_center_crop(Image.open(image_path), crop_size)
    return patches_final, image_crop_np, np.asarray(image_center_crop(Image.open(image_path), crop_size))





# if __name__ == '__main__':

    # # CLAHE all dark images of components
    # dark_image_path = '..\Image_Data_1'
    # image_read_clahe(dark_image_path, image_save=True)




    # # Repair some extremely dark image
    # dark_image_path = '..\Image_Data_1\yothing_photo380.jpg'
    # image = cv2.imread(dark_image_path)
    # img_clahe = image_clahe(image, cliplimit=1.0, gridsize=(1, 1))
    #
    # # plt.imshow(img_clahe, cmap='gray')
    # # plt.show()
    #
    # img_title = dark_image_path.rsplit('.', 1)[0] + '_clahe.jpg'
    # plt.imsave(img_title, img_clahe, cmap='gray')






    # # Image padding using cv2 reflection border type
    # image_clahe_path = 'F:\Monitoring_Image\Image_scale\image_padded'
    # # image_clahe_path = ['F:\Monitoring_Image\Image_scale\yothing_photo210_clahe.jpg']
    # img_path = glob.glob(image_clahe_path + '/*.jpg')
    #
    # for path in img_path:
    #
    #     image = cv2.imread(path)
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     # plt.imshow(image, cmap='gray')
    #     # plt.show()
    #
    #     image_reflect = image_padding(image, padding_size=(32, 32, 32, 32))
    #     # plt.imshow(image_reflect, cmap='gray')
    #     # plt.show()
    #     img_path = path.rsplit('.', 1)[0] + '_padded.jpg'
    #     print('saving {}'.format(path))
    #     plt.imsave(img_path, image_reflect, cmap='gray')





    # image_clahe_path = 'F:\Monitoring_Image\Image_scale\yothing_photo100_clahe.jpg'
    # image_pad_path = 'F:\Monitoring_Image\Image_scale\yothing_photo210_clahe_padded.jpg'
    # image_small = cv2.imread(image_clahe_path)
    # image_small = cv2.cvtColor(image_small, cv2.COLOR_RGB2GRAY)

    # image_large = cv2.imread(image_pad_path)
    # image_large = cv2.cvtColor(image_large, cv2.COLOR_RGB2GRAY)
    #
    # # Image patches with shape (32, 32), and larger patches with (96, 96)
    # # Small patches of image, and reshape it to (n, 32, 32)
    # patches_small, row_s, column_s, _, _ = image_patching_gray(image_small, patching_size=(32, 32), patching_stride=32)
    # patches_small = np.reshape(patches_small, (row_s * column_s, 32, 32))
    #
    # Large patches of image, and reshape it to (n, 96, 96)
    # patches_large, row_l, column_l, _, _ = image_patching_gray(image_large, patching_size=(128, 128), patching_stride=(64, 64))
    # patches_large = np.reshape(patches_large, (row_l * column_l, 128, 128))
    #
    # random_sv = random.sample(range(row_s * column_s), 5)
    #
    # for i in range(5):
    #
    #     plt.imsave('F:\Monitoring_Image\Image_scale\{}.jpg'.format(i), patches_small[random_sv[i], :, :], cmap='gray')
    #     plt.imsave('F:\Monitoring_Image\Image_scale\{}.jpg'.format(10 * i + 10), patches_large[random_sv[i], :, :], cmap='gray')







    ###################################################################################################################
    #                           Patch first, second layer of feature image.                                           #
    ###################################################################################################################

    # # Working for both first and second layers
    # # Find directories to find match the right padding image and patch images to patches
    # image_dir_path = 'F:\Monitoring_Image\Image_scale'
    # path = glob.iglob(image_dir_path + '/*')
    #
    # # Read all path for feature image patches picked already, stored as dictionary
    # image_feature_path = dict()
    #
    # for feature_path in path:
    #
    #     # Determine if this path is directory or image
    #     if feature_path.rsplit('.')[-1] == 'jpg':
    #         continue
    #     elif feature_path.rsplit('\\', 1)[-1] == 'image_padded':
    #         continue
    #     elif feature_path.rsplit('\\', 1)[-1] == 'image_clahe':
    #         continue
    #     else:
    #         image_feature_path[feature_path] = glob.glob(feature_path + '/*.jpg')





    # # Working for first layer, comment when trying to patch second layer patches
    # # Prepare for the padded image, ready to provide patches
    # img_org_path = 'F:\\Monitoring_Image\\Image_scale\\image_clahe'
    # image_org_path = {img_org_path: glob.glob(img_org_path + '/*.jpg')}
    # image_org_name = image_org_path[img_org_path]
    #
    #
    #
    # # Working for first layer, comment when trying to patch second layer patches
    # # Extract a dictionary to store all patches
    # patches = dict()
    # for image_org in image_org_name:
    #
    #     # Extract the padded image name
    #     image_name = image_org.rsplit('.', 1)[0]
    #     image_name = image_name.rsplit('\\', 1)[-1]
    #
    #     # Start patching padded image with size 128 x 128 pixels
    #     image_org_gray = cv2.imread(image_org)
    #     image_org_gray = cv2.cvtColor(image_org_gray, cv2.COLOR_RGB2GRAY)
    #     patches_temp, row_s, column_s, _, _ = image_patching_gray(image_org_gray, patching_size=(64, 64),
    #                                                               patching_stride=(64, 64))
    #     # Store the patches into dictionary for further matches
    #     patches[image_name] = np.reshape(patches_temp, (row_s * column_s, 64, 64))
    #
    #
    #
    # # Working for first layer, comment when trying to patch second layer patches
    # # Match padded image with feature images
    # for image_feature, images in image_feature_path.items():
    #
    #     # 'images' stores all the feature image path in list
    #     # Extract the image name need to be matched
    #     image_req_patched = image_feature.rsplit('_', 1)[0]
    #     image_req_patched = image_req_patched.rsplit('\\', 1)[-1]
    #
    #     # Extract the number of patch, and then match with padded image dictionary
    #     for img_patch in images:
    #
    #         print('processing {}'.format(img_patch))
    #
    #         patch_path = img_patch.rsplit('.', 1)[0]
    #
    #         try:
    #             patch_num = int(patch_path.rsplit('_', 1)[-1])
    #             if patch_num == 1:
    #                 print('First layer patch! Skipping ...')
    #                 continue
    #             if patch_num == 2:
    #                 print('Second layer patch! Skipping ...')
    #                 continue
    #         except ValueError:
    #             print('Cannot extract the patch number, check the image patch {}'.format(img_patch))
    #             continue
    #
    #         img_first_layer = patches[image_req_patched][patch_num - 1]
    #
    #         cv2.imwrite(patch_path + '_1.jpg', img_first_layer)


    # # Working for second layer, comment when trying to patch first layer patches
    # # Prepare for the padded image, ready to provide patches
    # img_padded_path = 'F:\\Monitoring_Image\\Image_scale\\image_padded'
    # image_padded_path = {img_padded_path: glob.glob(img_padded_path + '/*.jpg')}
    # image_padded_name = image_padded_path[img_padded_path]
    #
    #
    # # Working for second layer, comment when trying to patch first layer patches
    # # Extract a dictionary to store all patches
    # patches = dict()
    #
    # for image_padded in image_padded_name:
    #
    #     # Extract the padded image name
    #     image_name = image_padded.rsplit('.', 1)[0]
    #     image_name = image_name.rsplit('\\', 1)[-1]
    #     # Abandon 'padded' postfix in the file name
    #     image_name = image_name.rsplit('_', 1)[0]
    #
    #     # Start patching padded image with size 128 x 128 pixels
    #     image_padded_gray = cv2.imread(image_padded)
    #     image_padded_gray = cv2.cvtColor(image_padded_gray, cv2.COLOR_RGB2GRAY)
    #     patches_temp, row_s, column_s, _, _ = image_patching_gray(image_padded_gray, patching_size=(128, 128),
    #                                                               patching_stride=(64, 64))
    #     # Store the patches into dictionary for further matches
    #     patches[image_name] = np.reshape(patches_temp, (row_s * column_s, 128, 128))
    #
    #
    # # Working for second layer, comment when trying to patch first layer patches
    # # Match padded image with feature images
    # for image_feature, images in image_feature_path.items():
    #
    #     # 'images' stores all the feature image path in list
    #     # Extract the image name need to be matched
    #     image_req_patched = image_feature.rsplit('_', 1)[0]
    #     image_req_patched = image_req_patched.rsplit('\\', 1)[-1]
    #
    #     # Extract the number of patch, and then match with padded image dictionary
    #     for img_patch in images:
    #
    #         print('processing {}'.format(img_patch))
    #
    #         patch_path = img_patch.rsplit('.', 1)[0]
    #
    #         try:
    #             patch_num = int(patch_path.rsplit('_', 1)[-1])
    #             if patch_num == 1:
    #                 print('First layer patch! Skipping ...')
    #                 continue
    #             if patch_num == 2:
    #                 print('Second layer patch! Skipping ...')
    #                 continue
    #         except ValueError:
    #             print('Cannot extract the patch number, check the image patch {}'.format(img_patch))
    #             continue
    #
    #         img_sec_layer = patches[image_req_patched][patch_num - 1]
    #
    #         cv2.imwrite(patch_path + '_2.jpg', img_sec_layer)


    ###################################################################################################################
    #                                            End of code                                                          #
    ###################################################################################################################






    ###################################################################################################################
    #        Patch third layer of feature image, padding rectangular image(padded image) to square image              #
    #     Padding method is constant color padding, the mean value of padded image is used to padding color code      #
    ###################################################################################################################

    # # Image padding using cv2 reflection border type
    # image_clahe_path = 'F:\Monitoring_Image\Image_scale\image_padded'
    # # image_clahe_path = ['F:\Monitoring_Image\Image_scale\yothing_photo210_clahe.jpg']
    # img_path = glob.glob(image_clahe_path + '/*.jpg')
    #
    # for path in img_path:
    #
    #     image = cv2.imread(path)
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     img_size_h, img_size_w = image.shape
    #     image_mean = image.mean()
    #
    #     if img_size_w > img_size_h:
    #
    #         padding_size = int((img_size_w - img_size_h) / 2)
    #
    #         image_reflect = cv2.copyMakeBorder(image, padding_size, padding_size, 0, 0, cv2.BORDER_CONSTANT, value=image_mean)
    #
    #     else:
    #         padding_size = int((img_size_h - img_size_w) / 2)
    #
    #         image_reflect = cv2.copyMakeBorder(image, 0, 0, padding_size, padding_size, cv2.BORDER_CONSTANT, value=image_mean)
    #
    #     temp_path = path.rsplit('.', 1)[0] + '_3.jpg'
    #     print('saving {}'.format(path))
    #     cv2.imwrite(temp_path, image_reflect)

    ###################################################################################################################
    #                                            End of code                                                          #
    ###################################################################################################################

    # path_1 = 'F:\Monitoring_Image\Image_scale\yothing_photo100_clahe_edge\yothing_photo100_clahe_32_first layer.jpg'
    # path_2 = 'F:\Monitoring_Image\Image_scale\yothing_photo100_clahe_edge\yothing_photo100_clahe_32_second layer.jpg'
    # image_1 = cv2.imread(path_1)
    # image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    # image_2 = cv2.imread(path_2)
    # image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    # # cv2.imwrite('32_fst_layer_test.jpg', image_1)
    # image_3 = cv2.imread('32_fst_layer_test.jpg')


















