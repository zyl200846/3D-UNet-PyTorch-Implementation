# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 16/08/2019 14:29
import os
import glob
import json
import SimpleITK as sitk
from utils import label_converter


def fetch_data_path(data_dir):
    """
    Fetch all data path
    :param data_dir: the root folder where data stored
    :return: data path (pet_path, ct_path, mask_path), dtype: tuple
    """
    data_paths = list()
    paths_list = glob.glob(os.path.join(os.path.dirname(__file__), data_dir, "*"))
    for i, subject_dir in enumerate(paths_list):
        if i < 10:
            pet_path = os.path.join(subject_dir, "STS_00" + str(i + 1) + "_PT_COR_16.tiff")
            ct_path = os.path.join(subject_dir, "STS_00" + str(i + 1) + "_CT_COR_16.tiff")
            mask_path = os.path.join(subject_dir, "STS_00" + str(i + 1) + "_MASS_PET_COR_16.tiff")
            data_paths.append((pet_path, ct_path, mask_path))
        else:
            pet_path = os.path.join(subject_dir, "STS_0" + str(i + 1) + "_PT_COR_16.tiff")
            ct_path = os.path.join(subject_dir, "STS_0" + str(i + 1) + "_CT_COR_16.tiff")
            mask_path = os.path.join(subject_dir, "STS_0" + str(i + 1) + "_MASS_PET_COR_16.tiff")
            data_paths.append((pet_path, ct_path, mask_path))
    return data_paths


def get_data(data_path):
    """
    use (pet_path, ct_path, mask_path) to get corresponding image array
    :param data_path: path consists of (pet_path, ct_path, mask_path)
    :return: a list of corresponding image path
    """
    pet_path, ct_path, mask_path = data_path
    pet_img = sitk.GetArrayFromImage(sitk.ReadImage(pet_path))
    ct_img = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    mask = label_converter(mask)
    return [pet_img, ct_img, mask]


def get_img_patch_idxs(img, overlap_stepsize):
    """
    This function is used to get patch indices of a single image
    The patch indices generated here are used to crop one image into patches
    :param img: the single image
    :param overlap_stepsize: the overlap step size to generate patches
    :return: patch indices
    """
    patch_idxs = []
    depth, height, width = img.shape
    patch_depth, patch_height, patch_width = 128, 128, 128

    depth_range = list(range(0, depth - patch_depth + 1, overlap_stepsize))
    height_range = list(range(0, height - patch_height + 1, overlap_stepsize))
    width_range = list(range(0, width - patch_width + 1, overlap_stepsize))

    if (depth - patch_depth) % overlap_stepsize != 0:
        depth_range.append(depth - patch_depth)
    if (height - patch_height) % overlap_stepsize != 0:
        height_range.append(height - patch_height)
    if (width - patch_width) % overlap_stepsize != 0:
        width_range.append(width - patch_width)

    for d in depth_range:
        for h in height_range:
            for w in width_range:
                patch_idxs.append((d, h, w))

    return patch_idxs


def crop_image(data_path, output_path, overlap_stepsize):
    """
    Cropping volumetric images into various patches with fixed size, e.g. (96, 96, 96)
    :param data_path: the complete data path for original data
    :param output_path: the output root folder for cropped images
    :param overlap_stepsize: the overlap step size used for cropping images
    :return: data paths and patches dict that
            stores number of patches for a single volumetric image and indices for that image
    """
    patch_dict = dict()
    patch_depth, patch_height, patch_width = 128, 128, 128
    no_sample = len(data_path)

    for i in range(no_sample):
        pet_path, ct_path, mask_path = data_path[i]
        subject = os.path.basename(os.path.dirname(pet_path))
        print("Start Processing subject {}".format(subject))
        pet, ct, mask = get_data(data_path=data_path[i])
        patch_idxs = get_img_patch_idxs(img=pet, overlap_stepsize=overlap_stepsize)

        total_no_of_patch_idxs = len(patch_idxs)

        patch_dict[subject] = (total_no_of_patch_idxs, mask_path, patch_idxs)
        for j in range(len(patch_idxs)):
            d, h, w = patch_idxs[j]
            cropped_pet = sitk.GetImageFromArray(pet[d: d + patch_depth, h: h + patch_height, w: w + patch_width])
            cropped_ct = sitk.GetImageFromArray(ct[d: d + patch_depth, h: h + patch_height, w: w + patch_width])
            cropped_mask = sitk.GetImageFromArray(mask[d: d + patch_depth, h: h + patch_height, w: w + patch_width])
            subject_dir = os.path.basename(os.path.dirname(pet_path))

            if not os.path.exists(os.path.join(output_path, subject_dir, "PET")) or \
                    not os.path.exists(os.path.join(output_path, subject_dir, "CT")) or \
                    not os.path.exists(os.path.join(output_path, subject_dir, "MASK")):
                os.makedirs(os.path.join(output_path, subject_dir, "PET"))
                os.makedirs(os.path.join(output_path, subject_dir, "CT"))
                os.makedirs(os.path.join(output_path, subject_dir, "MASK"))

            sitk.WriteImage(cropped_pet, os.path.join(os.path.join(output_path,
                                                                   os.path.basename(os.path.dirname(pet_path)),
                                                                   "PET"),
                                                      os.path.splitext(os.path.basename(pet_path))[0]
                                                      + "_" + str(j + 1)
                                                      + os.path.splitext(os.path.basename(pet_path))[1]))
            sitk.WriteImage(cropped_ct, os.path.join(os.path.join(output_path,
                                                                  os.path.basename(os.path.dirname(ct_path)),
                                                                  "CT"),
                                                     os.path.splitext(os.path.basename(ct_path))[0]
                                                     + "_" + str(j + 1)
                                                     + os.path.splitext(os.path.basename(ct_path))[1]))
            sitk.WriteImage(cropped_mask, os.path.join(os.path.join(output_path,
                                                                    os.path.basename(os.path.dirname(mask_path)),
                                                                    "MASK"),
                                                       os.path.splitext(os.path.basename(mask_path))[0]
                                                       + "_" + str(j + 1)
                                                       + os.path.splitext(os.path.basename(mask_path))[1]))
    with open("patches_dict.json", "w") as f:
        json.dump(patch_dict, f)

    return data_path, patch_dict


if __name__ == "__main__":
    ps = fetch_data_path("data")
    dp, pd = crop_image(ps, "./processed", overlap_stepsize=64)
