# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 21/08/2019 21:46
import os
import glob
import natsort
import json
import numpy as np
import SimpleITK as sitk


def get_results_path(data_dir):
    complete_paths = list()
    paths_list = glob.glob(os.path.join(os.path.dirname(__file__), data_dir, "*"))
    for path in paths_list:
        root_path = os.path.join(path, "MASK")
        mask_paths = [os.path.join(root_path, subject_path) for subject_path in os.listdir(root_path)]
        mask_paths = natsort.natsorted(mask_paths)
        complete_paths.append(mask_paths)
    return complete_paths


def rotate(image_path):
    """
    This function should be added to reconstruction function if flip augmentation is used while training unet3d_model
    :param image_path: the patch prediction path
    :return: rotated image
    """
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    d, h, w = image.shape
    new_image = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            new_image[:, h, w - 1 - j] = new_image[:, h - 1 - i, j]
    return new_image


def get_patches_num_and_patch_idxs(patch_dict_json):
    """
    Load patch dict to get corresponding information, e.g. patches number, patch indices, mask path
    :param patch_dict_json: json file that stores information mentioned above
    :return: loaded dict
    """
    with open(patch_dict_json, "r") as f:
        loaded_dict = json.load(f)
    return loaded_dict


def recombine_results(patches_dict, results_path, output_path):
    """
    This function is used to reconstruct the cropped mask image patches back to the original size of image
    :param patches_dict: the patch dic that stores the number of patches, patch indices and mask path
    :param results_path: the patch predictions path
    :param output_path: where to store reconstructed image
    :return: None
    """
    # output path = ./predictions/STS_001/STS_001_MASS_PET_COR_16_prediction.tiff
    for i, (k, cropped_result_paths) in enumerate(zip(patches_dict.keys(), results_path)):
        mask_name, mask_ext = os.path.splitext(os.path.basename(patches_dict.get(k)[1]))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(patches_dict.get(k)[1]))
        patch_indices = patches_dict.get(k)[2]
        print(patch_indices)
        _d, _h, _w = mask.shape
        predicted_mask = np.zeros((_d, _h, _w), np.uint8)
        for patch_idx, path in zip(patch_indices, cropped_result_paths):
            d, h, w = patch_idx
            pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
            depth, height, width = pred_arr.shape
            roi = pred_arr[:, :, :]
            predicted_mask[d: d + depth, h: h + height, w: w + width] = roi
        if not os.path.exists(os.path.join(output_path, k)):
            os.makedirs(os.path.join(output_path, k))
        # print(os.path.join(output_path, k, mask_name + "predictions" + mask_ext))
        sitk.WriteImage(sitk.GetImageFromArray(predicted_mask), os.path.join(output_path,
                                                                             k,
                                                                             mask_name + "_predictions" + mask_ext))


if __name__ == "__main__":
    complete_mask_paths = get_results_path("processed")
    pd = get_patches_num_and_patch_idxs("patches_dict.json")
    recombine_results(pd, complete_mask_paths, output_path="predictions")
