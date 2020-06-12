# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import numpy as np
import torch

# Needed to prevent pytest from raising a warning when importing torchvision via dream
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import imp

from dream import image_proc

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_shrink_resolution():

    # Testing that (640 x 480) shrinks to (533 x 400) using a (400, 400) reference
    input_resolution = (640, 480)
    ref_resolution = (400, 400)
    shrink_res = image_proc.shrink_resolution(input_resolution, ref_resolution)
    shrink_res_gt = (533, 400)
    assert shrink_res == shrink_res_gt

    # Testing that shrink is a no-op if the resolution is the same
    ref_resolution_same = input_resolution
    shrink_res_same = image_proc.shrink_resolution(
        input_resolution, ref_resolution_same
    )
    assert shrink_res_same == input_resolution


def test_shrink_and_crop_resolution():

    # Testing with (640 x 480) input with (400, 400) reference
    input_resolution = (640, 480)
    ref_resolution = (400, 400)
    input_cropped_res, cropped_coords = image_proc.shrink_and_crop_resolution(
        input_resolution, ref_resolution
    )

    input_cropped_res_gt = (480, 480)
    cropped_coords_gt = (80, 0)

    assert input_cropped_res == input_cropped_res_gt
    assert cropped_coords == cropped_coords_gt

    # Testing that shrink-and-crop is a no-op if the resolution is the same
    ref_resolution_same = input_resolution
    input_cropped_res_same, cropped_coords_same = image_proc.shrink_and_crop_resolution(
        input_resolution, ref_resolution_same
    )
    cropped_coords_same_gt = (0, 0)

    assert input_cropped_res_same == input_resolution
    assert cropped_coords_same == cropped_coords_same_gt


def test_resolution_after_preprocessing():

    # Testing with (640 x 480) input with (400, 400) reference
    input_resolution = (640, 480)
    ref_resolution = (400, 400)

    preproc_res_none = image_proc.resolution_after_preprocessing(
        input_resolution, ref_resolution, "none"
    )
    preproc_res_none_gt = input_resolution
    assert preproc_res_none == preproc_res_none_gt

    preproc_res_resize = image_proc.resolution_after_preprocessing(
        input_resolution, ref_resolution, "resize"
    )
    preproc_res_resize_gt = ref_resolution
    assert preproc_res_resize == preproc_res_resize_gt

    preproc_res_shrink = image_proc.resolution_after_preprocessing(
        input_resolution, ref_resolution, "shrink"
    )
    preproc_res_shrink_gt = (533, 400)
    assert preproc_res_shrink == preproc_res_shrink_gt

    preproc_res_snc = image_proc.resolution_after_preprocessing(
        input_resolution, ref_resolution, "shrink-and-crop"
    )
    preproc_res_snc_gt = ref_resolution
    assert preproc_res_snc == preproc_res_snc_gt


def test_belief_maps():

    # Generate belief maps
    belief_map_resolution = (80, 60)
    kp_proj = np.array([65.0, 20.0])
    kp_proj_out_of_frame = np.array(
        [belief_map_resolution[0] + 20.0, belief_map_resolution[1] + 20.0]
    )
    belief_maps = image_proc.create_belief_map(
        belief_map_resolution, [kp_proj, kp_proj_out_of_frame]
    )

    belief_maps_as_tensor = torch.tensor(belief_maps).float()
    # Uncomment below for showing visuals
    # image_proc.image_from_belief_map(belief_maps_as_tensor[0]).show()
    # image_proc.image_from_belief_map(belief_maps_as_tensor[1]).show()

    # Detect keypoints in belief maps
    all_peaks = image_proc.peaks_from_belief_maps(belief_maps_as_tensor, 0.0)

    # First belief map should only contain one detected keypoint, the ground truth
    assert len(all_peaks[0]) == 1
    kp_proj_detected = np.array(all_peaks[0][0][:2])
    assert np.linalg.norm(kp_proj - kp_proj_detected) < 1.0e-3

    # Second belief map, for out-of-frame keypoint, should return no peaks
    assert len(all_peaks[1]) == 0
