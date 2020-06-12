# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import os

import numpy as np

# Needed to prevent pytest from raising a warning when importing torchvision via dream
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import imp

from dream import utilities as dream_utils

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_load_camera_intrinsics():

    # Provide the camera setting json that exists in this directory
    this_script_dir = os.path.dirname(os.path.realpath(__file__))
    camera_data_path = os.path.join(this_script_dir, "_camera_settings.json")
    camera_K = dream_utils.load_camera_intrinsics(camera_data_path)

    # Construct the ground truth camera matrix
    cam_fx = 160.0
    cam_fy = 160.0
    cam_cx = 160.0
    cam_cy = 120.0

    camera_K_gt = np.array(
        [[cam_fx, 0.0, cam_cx], [0.0, cam_fy, cam_cy], [0.0, 0.0, 1.0]]
    )

    # Test assertion
    assert np.all(camera_K == camera_K_gt)


def test_load_image_resolution():

    # Provide the camera setting json that exists in this directory
    this_script_dir = os.path.dirname(os.path.realpath(__file__))
    camera_data_path = os.path.join(this_script_dir, "_camera_settings.json")
    image_resolution = dream_utils.load_image_resolution(camera_data_path)

    # Construct the ground truth image resolution
    image_resolution_gt = (320, 240)

    # Test assertion
    assert image_resolution == image_resolution_gt
