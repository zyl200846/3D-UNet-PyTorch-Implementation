# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 21/08/2019 22:46
import numpy as np


def standardize(data):
    avgs = list()
    stds = list()
    for i in range(len(data)):
        avgs.append(np.mean(data[i], axis=(1, 2, 3)))
        stds.append(np.std(data[i], axis=(1, 2, 3)))
    avg = np.mean(np.asarray(avgs), axis=0)
    std = np.mean(np.asarray(stds), axis=0)
    data = (data - avg) / std
    return data


def normalize(data):
    min_values = list()
    max_values = list()
    for i in range(len(data)):
        min_values.append(np.min(data[i], axis=(1, 2, 3)))
        max_values.append(np.max(data[i], axis=(1, 2, 3)))
    data_min = np.mean(np.asarray(min_values), axis=0)
    data_max = np.mean(np.asarray(max_values), axis=0)
    data = (data - data_min) / data_max
    return data


def label_converter(mask_data):
    new_masks = np.zeros(shape=mask_data.shape, dtype=np.uint8)
    for i in range(len(mask_data)):
        new_masks[i] = np.where(mask_data[i] > 0, 1, 0)

    return new_masks


if __name__ == "__main__":
    import SimpleITK as sitk
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    img = sitk.GetArrayFromImage(sitk.ReadImage("data/STS_001/STS_001_PT_COR_16.tiff"))
    print(img.shape)
    img = img / 255.
    print(img[64])
