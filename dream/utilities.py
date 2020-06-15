# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import os
import random

import numpy as np
from ruamel.yaml import YAML
import torch

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def set_random_seed(seed):
    assert isinstance(
        seed, int
    ), 'Expected "seed" to be an integer, but it is "{}".'.format(type(seed))
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def makedirs(directory, exist_ok=False):
    """A method that replicates the functionality of os.makedirs that works for both Python 2 and Python 3."""
    if os.path.exists(directory):
        assert exist_ok, 'Specified directory "{}" already exists.'.format(directory)
    else:
        os.makedirs(directory)
    return


def is_ndds_dataset(input_dir, data_extension="json"):

    # Input argument handling
    # Expand user shortcut if it exists
    input_dir = os.path.expanduser(input_dir)
    assert os.path.exists(
        input_dir
    ), 'Expected path "{}" to exist, but it does not.'.format(input_dir)
    assert isinstance(
        data_extension, str
    ), 'Expected "data_extension" to be a string, but it is "{}".'.format(
        type(data_extension)
    )

    data_full_ext = "." + data_extension

    dirlist = os.listdir(input_dir)

    # Find json files
    data_filenames = [f for f in dirlist if f.endswith(data_full_ext)]

    # Extract name from json file
    data_names = [os.path.splitext(f)[0] for f in data_filenames if f[0].isdigit()]

    is_ndds_dataset = True if data_names else False

    return is_ndds_dataset


def find_ndds_data_in_dir(
    input_dir, data_extension="json", image_extension=None, requested_image_types="all",
):

    # Input argument handling
    # Expand user shortcut if it exists
    input_dir = os.path.expanduser(input_dir)
    assert os.path.exists(
        input_dir
    ), 'Expected path "{}" to exist, but it does not.'.format(input_dir)
    dirlist = os.listdir(input_dir)

    assert isinstance(
        data_extension, str
    ), 'Expected "data_extension" to be a string, but it is "{}".'.format(
        type(data_extension)
    )
    data_full_ext = "." + data_extension

    if image_extension is None:
        # Auto detect based on list of image extensions to try
        # In case there is a tie, prefer the extensions that are closer to the front
        image_exts_to_try = ["png", "jpg"]
        num_image_exts = []
        for image_ext in image_exts_to_try:
            num_image_exts.append(len([f for f in dirlist if f.endswith(image_ext)]))
        max_num_image_exts = np.max(num_image_exts)
        idx_max = np.where(num_image_exts == max_num_image_exts)[0]
        # If there are multiple indices due to ties, this uses the one closest to the front
        image_extension = image_exts_to_try[idx_max[0]]
        # Mention to user if there are multiple cases to ensure they are aware of the selection
        if len(idx_max) > 1 and max_num_image_exts > 0:
            print(
                'Multiple sets of images detected in NDDS dataset with different extensions. Using extension "{}".'.format(
                    image_extension
                )
            )
    else:
        assert isinstance(
            image_extension, str
        ), 'If specified, expected "image_extension" to be a string, but it is "{}".'.format(
            type(image_extension)
        )
    image_full_ext = "." + image_extension

    assert (
        requested_image_types is None
        or requested_image_types == "all"
        or isinstance(requested_image_types, list)
    ), "Expected \"requested_image_types\" to be None, 'all', or a list of requested_image_types."

    # Read in json files
    data_filenames = [f for f in dirlist if f.endswith(data_full_ext)]

    # Sort candidate data files by name
    data_filenames.sort()

    data_names = [os.path.splitext(f)[0] for f in data_filenames if f[0].isdigit()]

    # If there are no matching json files -- this is not an NDDS dataset -- return None
    if not data_names:
        return None, None

    data_paths = [os.path.join(input_dir, f) for f in data_filenames if f[0].isdigit()]

    if requested_image_types == "all":
        # Detect based on first entry
        first_entry_name = data_names[0]
        matching_image_names = [
            f
            for f in dirlist
            if f.startswith(first_entry_name) and f.endswith(image_full_ext)
        ]
        find_rgb = (
            True
            if first_entry_name + ".rgb" + image_full_ext in matching_image_names
            else False
        )
        find_depth = (
            True
            if first_entry_name + ".depth" + image_full_ext in matching_image_names
            else False
        )
        find_cs = (
            True
            if first_entry_name + ".cs" + image_full_ext in matching_image_names
            else False
        )
        if len(matching_image_names) > 3:
            print("Image types detected that are not yet implemented in this function.")

    elif requested_image_types:
        # Check based on known data types
        known_image_types = ["rgb", "depth", "cs"]
        for this_image_type in requested_image_types:
            assert (
                this_image_type in known_image_types
            ), 'Image type "{}" not recognized.'.format(this_image_type)

        find_rgb = True if "rgb" in requested_image_types else False
        find_depth = True if "depth" in requested_image_types else False
        find_cs = True if "cs" in requested_image_types else False

    else:
        find_rgb = False
        find_depth = False
        find_cs = False

    dict_of_lists_images = {}
    n_samples = len(data_names)

    if find_rgb:
        rgb_paths = [
            os.path.join(input_dir, f + ".rgb" + image_full_ext) for f in data_names
        ]
        for n in range(n_samples):
            assert os.path.exists(
                rgb_paths[n]
            ), 'Expected image "{}" to exist, but it does not.'.format(rgb_paths[n])
        dict_of_lists_images["rgb"] = rgb_paths

    if find_depth:
        depth_paths = [
            os.path.join(input_dir, f + ".depth" + image_full_ext) for f in data_names
        ]
        for n in range(n_samples):
            assert os.path.exists(
                depth_paths[n]
            ), 'Expected image "{}" to exist, but it does not.'.format(depth_paths[n])
        dict_of_lists_images["depth"] = depth_paths

    if find_cs:
        cs_paths = [
            os.path.join(input_dir, f + ".cs" + image_full_ext) for f in data_names
        ]
        for n in range(n_samples):
            assert os.path.exists(
                cs_paths[n]
            ), 'Expected image "{}" to exist, but it does not.'.format(cs_paths[n])
        dict_of_lists_images["class_segmentation"] = cs_paths

    found_images = [
        dict(zip(dict_of_lists_images, t)) for t in zip(*dict_of_lists_images.values())
    ]

    # Create output dictionaries
    dict_of_lists = {"name": data_names, "data_path": data_paths}

    if find_rgb or find_depth or find_cs:
        dict_of_lists["image_paths"] = found_images

    found_data = [dict(zip(dict_of_lists, t)) for t in zip(*dict_of_lists.values())]

    # Process config files, which are data files that don't have an associated image
    found_configs = {"camera": None, "object": None, "unsorted": []}
    data_filenames_without_images = [f for f in data_filenames if not f[0].isdigit()]

    for data_filename in data_filenames_without_images:
        if data_filename == "_camera_settings" + data_full_ext:
            found_configs["camera"] = os.path.join(input_dir, data_filename)
        elif data_filename == "_object_settings" + data_full_ext:
            found_configs["object"] = os.path.join(input_dir, data_filename)
        else:
            found_configs["unsorted"].append(os.path.join(input_dir, data_filename))

    return found_data, found_configs


def load_camera_intrinsics(camera_data_path):

    # Input argument handling
    assert os.path.exists(
        camera_data_path
    ), 'Expected path "{}" to exist, but it does not.'.format(camera_data_path)

    # Create YAML/json parser
    data_parser = YAML(typ="safe")

    with open(camera_data_path, "r") as f:
        cam_settings_data = data_parser.load(f)

    camera_fx = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["fx"]
    camera_fy = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["fy"]
    camera_cx = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["cx"]
    camera_cy = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["cy"]
    camera_K = np.array(
        [[camera_fx, 0.0, camera_cx], [0.0, camera_fy, camera_cy], [0.0, 0.0, 1.0]]
    )

    return camera_K


def load_image_resolution(camera_data_path):

    # Input argument handling
    assert os.path.exists(
        camera_data_path
    ), 'Expected path "{}" to exist, but it does not.'.format(camera_data_path)

    # Create YAML/json parser
    data_parser = YAML(typ="safe")

    with open(camera_data_path, "r") as f:
        cam_settings_data = data_parser.load(f)

    image_width = cam_settings_data["camera_settings"][0]["captured_image_size"][
        "width"
    ]
    image_height = cam_settings_data["camera_settings"][0]["captured_image_size"][
        "height"
    ]
    image_resolution = (image_width, image_height)

    return image_resolution


def load_keypoints(data_path, object_name, keypoint_names):
    assert os.path.exists(
        data_path
    ), 'Expected data_path "{}" to exist, but it does not.'.format(data_path)

    # Set up output structure
    keypoint_data = {"positions_wrt_cam": [], "projections": []}

    # Load keypoints for a particular object for now
    parser = YAML(typ="safe")
    with open(data_path, "r") as f:
        data = parser.load(f)

    assert (
        "objects" in data.keys()
    ), 'Expected "objects" key to exist in data file, but it does not.'

    object_names = [o["class"] for o in data["objects"]]
    assert (
        object_name in object_names
    ), 'Requested object_name "{}" does not exist in the data file objects.'.format(
        object_name
    )

    idx_object = object_names.index(object_name)

    object_data = data["objects"][idx_object]
    object_keypoints = object_data["keypoints"]

    object_keypoint_names = [kp["name"] for kp in object_keypoints]

    # Process in same order as keypoint_names to retain same order
    for kp_name in keypoint_names:
        assert (
            kp_name in object_keypoint_names
        ), "Expected keypoint '{}' to exist in the data file '{}', but it does not.  Rather, the keypoints are '{}'".format(
            kp_name, data_path, object_keypoint_names
        )

        idx_kp = object_keypoint_names.index(kp_name)
        kp_data = object_keypoints[idx_kp]

        kp_position_wrt_cam = kp_data["location"]
        kp_projection = kp_data["projected_location"]

        keypoint_data["positions_wrt_cam"].append(kp_position_wrt_cam)
        keypoint_data["projections"].append(kp_projection)

    return keypoint_data
