# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from scipy.ndimage.filters import gaussian_filter
import torch
import torchvision.transforms.functional as TVTransformsFunc
import webcolors

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


KNOWN_IMAGE_PREPROC_TYPES = [
    "none",  # No preprocessing. Output is the same as the input image.
    "resize",  # Resizes (without fixing aspect ratio). Output has same resolution as reference, but may have different aspect ratio as input.
    "shrink",  # Resizes (with fixed aspect ratio) to reference resolution height. Output has same aspect ratio as input.
    "shrink-and-crop",
]  # Crops and resizes (with fixed aspect ratio) to reference resolution. Output has same resolution as reference, same aspect ratio as input, but may be missing part of input frame due to crop.


def preprocess_image(input_image, image_ref_resolution, image_preprocessing):

    # Input argument handling
    assert isinstance(
        input_image, PILImage.Image
    ), 'Expected "input_image" to be a PIL Image, but it is "{}".'.format(
        type(input_image)
    )
    assert (
        image_preprocessing in KNOWN_IMAGE_PREPROC_TYPES
    ), 'Image preprocessing type "{}" is not recognized.'.format(image_preprocessing)

    if image_preprocessing == "none":
        preprocessed_image = input_image
    elif image_preprocessing == "resize":
        preprocessed_image = input_image.resize(
            image_ref_resolution, resample=PILImage.BILINEAR
        )
    elif image_preprocessing == "shrink":
        preprocessed_image = scale_image(
            input_image, new_height=image_ref_resolution[1]
        )
    elif image_preprocessing == "shrink-and-crop":
        preprocessed_image = shrink_and_crop_image(input_image, image_ref_resolution)

    return preprocessed_image


def inverse_preprocess_image(
    preprocessed_image, image_input_resolution, image_preprocessing
):

    # Input argument handling
    assert isinstance(
        preprocessed_image, PILImage.Image
    ), 'Expected "preprocessed_image" to be a PIL Image, but it is "{}".'.format(
        type(preprocessed_image)
    )
    assert (
        image_preprocessing in KNOWN_IMAGE_PREPROC_TYPES
    ), 'Image preprocessing type "{}" is not recognized.'.format(image_preprocessing)

    if image_preprocessing == "none":
        inv_preproc_image = preprocessed_image
    elif image_preprocessing == "resize":
        inv_preproc_image = preprocessed_image.resize(
            image_input_resolution, resample=PILImage.BILINEAR
        )
    elif image_preprocessing == "shrink":
        # shrink preserves aspect ratio, so a simple resize will work
        inv_preproc_image = preprocessed_image.resize(
            image_input_resolution, resample=PILImage.BILINEAR
        )
    elif image_preprocessing == "shrink-and-crop":
        cropped_res, cropped_coords = shrink_and_crop_resolution(
            image_input_resolution, preprocessed_image.size
        )
        # We cannot recover the pixels that were cropped, so we will just use black pixels
        inv_preproc_image = PILImage.new("RGB", image_input_resolution)
        inv_resize_preproc_image = preprocessed_image.resize(
            cropped_res, resample=PILImage.BILINEAR
        )
        inv_preproc_image.paste(inv_resize_preproc_image, box=cropped_coords)

    return inv_preproc_image


def resolution_after_preprocessing(
    image_input_resolution, image_ref_resolution, image_preprocessing
):

    # Input argument handling
    assert (
        len(image_input_resolution) == 2
    ), 'Expected "image_input_resolution" to have length 2, but it has length {}.'.format(
        len(image_input_resolution)
    )
    assert (
        len(image_ref_resolution) == 2
    ), 'Expected "image_ref_resolution" to have length 2, but it has length {}.'.format(
        len(image_ref_resolution)
    )

    assert (
        image_preprocessing in KNOWN_IMAGE_PREPROC_TYPES
    ), 'Image preprocessing type "{}" is not recognized.'.format(image_preprocessing)

    if image_preprocessing == "none":
        image_res_after_preproc = image_input_resolution
    elif image_preprocessing == "resize":
        image_res_after_preproc = image_ref_resolution
    elif image_preprocessing == "shrink":
        image_res_after_preproc = shrink_resolution(
            image_input_resolution, image_ref_resolution
        )
    elif image_preprocessing == "shrink-and-crop":
        image_res_after_preproc = image_ref_resolution

    return image_res_after_preproc


def shrink_resolution(image_input_resolution, image_ref_resolution):
    # Casting to float for Py2
    factor = float(image_ref_resolution[1]) / float(image_input_resolution[1])
    new_width = int(image_input_resolution[0] * factor)
    shrink_image_res = (new_width, image_ref_resolution[1])
    return shrink_image_res


def convert_keypoints_to_netin_from_netout(
    keypoints_netout, net_output_resolution, net_input_resolution
):
    # Convert keypoints from net-output to net-input
    temp = []
    for kps in keypoints_netout:
        proj = [
            kps[0] / net_output_resolution[0] * net_input_resolution[0],
            kps[1] / net_output_resolution[1] * net_input_resolution[1],
        ]
        temp.append(proj)
    keypoints_netin = np.array(temp)
    return keypoints_netin


def convert_keypoints_to_netout_from_netin(
    keypoints_netin, net_input_resolution, net_output_resolution
):
    # Convert keypoints from net-input to net-output
    temp = []
    for kps in keypoints_netin:
        proj = [
            kps[0] / net_input_resolution[0] * net_output_resolution[0],
            kps[1] / net_input_resolution[1] * net_output_resolution[1],
        ]
        temp.append(proj)
    keypoints_netout = np.array(temp)
    return keypoints_netout


def convert_keypoints_to_netin_from_raw(
    keypoints_raw, image_raw_resolution, net_input_resolution, image_preprocessing
):

    assert (
        image_preprocessing in KNOWN_IMAGE_PREPROC_TYPES
    ), 'Image preprocessing type "{}" is not recognized.'.format(image_preprocessing)

    if image_preprocessing == "none":
        keypoints_netin = keypoints_raw

    elif image_preprocessing == "resize":
        keypoints_netin = []
        for proj in keypoints_raw:
            kp_netin = [
                proj[0] / image_raw_resolution[0] * net_input_resolution[0],
                proj[1] / image_raw_resolution[1] * net_input_resolution[1],
            ]
            keypoints_netin.append(kp_netin)

    elif image_preprocessing == "shrink":
        shrink_image_res = shrink_resolution(image_raw_resolution, net_input_resolution)
        keypoints_netin = []
        for proj in keypoints_raw:
            kp_netin = [
                proj[0] / image_raw_resolution[0] * shrink_image_res[0],
                proj[1] / image_raw_resolution[1] * shrink_image_res[1],
            ]
            keypoints_netin.append(kp_netin)

    elif image_preprocessing == "shrink-and-crop":
        image_raw_cropped_res, image_raw_cropped_coords = shrink_and_crop_resolution(
            image_raw_resolution, net_input_resolution
        )
        keypoints_netin = []
        for proj in keypoints_raw:
            kp_netin = [
                (proj[0] - image_raw_cropped_coords[0])
                / image_raw_cropped_res[0]
                * net_input_resolution[0],
                (proj[1] - image_raw_cropped_coords[1])
                / image_raw_cropped_res[1]
                * net_input_resolution[1],
            ]
            keypoints_netin.append(kp_netin)

    keypoints_netin = np.array(keypoints_netin)
    return keypoints_netin


def convert_keypoints_to_raw_from_netin(
    keypoints_netin, net_input_resolution, image_raw_resolution, image_preprocessing
):

    assert (
        image_preprocessing in KNOWN_IMAGE_PREPROC_TYPES
    ), 'Image preprocessing type "{}" is not recognized.'.format(image_preprocessing)

    # Reverse preprocessing
    if image_preprocessing == "none":
        keypoints_raw = keypoints_netin

    elif image_preprocessing == "resize":
        keypoints_raw = []
        for proj in keypoints_netin:
            kp_raw = [
                proj[0] / net_input_resolution[0] * image_raw_resolution[0],
                proj[1] / net_input_resolution[1] * image_raw_resolution[1],
            ]
            keypoints_raw.append(kp_raw)

    elif image_preprocessing == "shrink":
        keypoints_raw = []
        for proj in keypoints_netin:
            kp_raw = [
                proj[0] / net_input_resolution[0] * image_raw_resolution[0],
                proj[1] / net_input_resolution[1] * image_raw_resolution[1],
            ]
            keypoints_raw.append(kp_raw)

    elif image_preprocessing == "shrink-and-crop":
        keypoints_raw = []
        image_raw_cropped_res, image_raw_cropped_coords = shrink_and_crop_resolution(
            image_raw_resolution, net_input_resolution
        )
        for proj in keypoints_netin:
            kp_raw = [
                proj[0] / net_input_resolution[0] * image_raw_cropped_res[0]
                + image_raw_cropped_coords[0],
                proj[1] / net_input_resolution[1] * image_raw_cropped_res[1]
                + image_raw_cropped_coords[1],
            ]
            keypoints_raw.append(kp_raw)

    keypoints_raw = np.array(keypoints_raw)
    return keypoints_raw


def convert_image_to_netin_from_netout(image_netout, net_input_resolution):

    # Input argument handling
    assert isinstance(
        image_netout, PILImage.Image
    ), 'Expected "image_netout" to be a PIL Image, but it is "{}".'.format(
        type(image_netout)
    )

    # This is just resizing
    image_netin = image_netout.resize(net_input_resolution, resample=PILImage.BILINEAR)
    return image_netin


def convert_image_to_netout_from_netin(image_netin, net_output_resolution):

    # Input argument handling
    assert isinstance(
        image_netin, PILImage.Image
    ), 'Expected "image_netin" to be a PIL Image, but it is "{}".'.format(
        type(image_netin)
    )

    # This is just resizing
    image_netout = image_netin.resize(net_output_resolution, resample=PILImage.BILINEAR)
    return image_netout


def shrink_and_crop_image(input_image, image_ref_resolution):

    # Input argument handling
    assert isinstance(
        input_image, PILImage.Image
    ), 'Expected "input_image" to be a PIL Image, but it is "{}".'.format(
        type(input_image)
    )

    (
        image_input_cropped_resolution,
        image_input_cropped_coords,
    ) = shrink_and_crop_resolution(input_image.size, image_ref_resolution)
    image_rgb_cropped, crop_image_coords = centered_crop_image(
        input_image,
        image_input_cropped_resolution[0],
        image_input_cropped_resolution[1],
    )
    # get rid of thses
    assert image_rgb_cropped.size == image_input_cropped_resolution
    assert image_input_cropped_coords == crop_image_coords
    processed_image = image_rgb_cropped.resize(
        image_ref_resolution, resample=PILImage.BILINEAR
    )
    return processed_image


def shrink_and_crop_resolution(image_input_resolution, image_ref_resolution):

    image_input_width, image_input_height = image_input_resolution
    image_ref_width, image_ref_height = image_ref_resolution

    # Casting to float for Py2
    scale_factor_based_on_width = float(image_input_width) / float(image_ref_width)
    image_ref_height_based_on_width = int(
        scale_factor_based_on_width * image_ref_height
    )

    # Casting to float for Py2
    scale_factor_based_on_height = float(image_input_height) / float(image_ref_height)
    image_ref_width_based_on_height = int(
        scale_factor_based_on_height * image_ref_width
    )

    if image_input_width >= image_ref_width_based_on_height:
        image_input_cropped_resolution = (
            image_ref_width_based_on_height,
            image_input_height,
        )
    else:
        assert image_input_height >= image_ref_height_based_on_width
        image_input_cropped_resolution = (
            image_input_width,
            image_ref_height_based_on_width,
        )

    image_input_cropped_coords = (
        (image_input_width - image_input_cropped_resolution[0]) // 2,
        (image_input_height - image_input_cropped_resolution[1]) // 2,
    )
    return image_input_cropped_resolution, image_input_cropped_coords


def crop_image(image, u, v, cropped_width, cropped_height):
    """Crop the given PIL.Image.
    Args:
        image (PIL.Image): Image to be cropped.
        u: Left/horizontal pixel coordinate.
        v: Upper/vertical pixel coordinate.
        cropped_width: Width of the cropped image.
        cropped_height: Height of the cropped image.
    Returns:
        PIL.Image: Cropped image.
    """
    # Input argument handling
    assert isinstance(
        image, PILImage.Image
    ), 'Expected "image" to be a PIL Image, but it is "{}".'.format(type(image))
    return image.crop((u, v, u + cropped_width, v + cropped_height))


def centered_crop_image(image, cropped_width, cropped_height):

    # Input argument handling
    assert isinstance(
        image, PILImage.Image
    ), 'Expected "image" to be a PIL Image, but it is "{}".'.format(type(image))
    image_width, image_height = image.size

    assert isinstance(
        cropped_width, int
    ), 'Expected cropped_width to be an integer, but it is "{}".'.format(
        type(cropped_width)
    )
    assert isinstance(
        cropped_height, int
    ), 'Expected cropped_width to be an integer, but it is "{}".'.format(
        type(cropped_height)
    )

    assert (
        0 < cropped_width and cropped_width <= image_width
    ), "Expected cropped_width to be greater than zero and less than or equal to image_width ({}), but it is {}.".format(
        image_width, cropped_width
    )
    assert (
        0 < cropped_height and cropped_height <= image_height
    ), "Expected cropped_height to be greater than zero and less than or equal to image_height ({}), but it is {}.".format(
        image_height, cropped_height
    )

    # Determine u & v coordinates for image cropping
    width_diff = image_width - cropped_width
    height_diff = image_height - cropped_height

    crop_u = width_diff // 2
    crop_v = height_diff // 2
    crop_image_coords = (crop_u, crop_v)

    return (
        crop_image(image, crop_u, crop_v, cropped_width, cropped_height),
        crop_image_coords,
    )


def scale_image(image, factor=-1, new_width=-1, new_height=-1):
    """ Scale an image, while preserving the aspect ratio.
        Specify either 'factor', or 'new_width', or 'new_height'.  If more are specified, then one will be used arbitrarily."""

    # Input argument handling
    assert isinstance(
        image, PILImage.Image
    ), 'Expected "image" to be a PIL Image, but it is "{}".'.format(type(image))
    image_width, image_height = image.size

    # Calculate new dimensions
    if factor > 0:
        new_width = int(image_width * factor)
        new_height = int(image_height * factor)
    elif new_width > 0:
        factor = new_width / image_width
        new_height = int(image_height * factor)
    elif new_height > 0:
        factor = new_height / image_height
        new_width = int(image_width * factor)
    else:
        assert (
            False
        ), "scale_image:  Must specify either 'factor', or 'new_width', or 'new_height'."

    assert isinstance(
        new_width, int
    ), 'Expected new_width to be an integer, but it is "{}".'.format(type(new_width))
    assert isinstance(
        new_height, int
    ), 'Expected new_width to be an integer, but it is "{}".'.format(type(new_height))

    assert (
        0 < new_width and new_width <= image_width
    ), "Expected new_width to be greater than zero and less than or equal to image_width ({}), but it is {}.".format(
        image_width, new_width
    )
    assert (
        0 < new_height and new_height <= image_height
    ), "Expected new_width to be greater than zero and less than or equal to image_height ({}), but it is {}.".format(
        image_height, new_height
    )

    return image.resize((new_width, new_height), resample=PILImage.BILINEAR)


def overlay_points_on_image(
    image_input,
    image_points,
    image_point_names=None,
    annotation_color_dot="red",
    annotation_color_text="red",
    point_diameter=6.0,
    point_thickness=-1,
):  # any negative value means a filled point will be drawn

    # Input argument handling
    if isinstance(image_input, str):
        image = PILImage.open(image_input).convert("RGB")
    else:
        assert isinstance(
            image_input, PILImage.Image
        ), 'Expected "image_input" to be either a PIL Image or an image path, but it is "{}".'.format(
            type(image_input)
        )
        image = image_input

    if image_points is None or len(image_points) == 0:
        return image_input  # return input if no points to overlay

    n_image_points = len(image_points)
    if image_point_names:
        assert n_image_points == len(
            image_point_names
        ), "Expected the number of image point names to be the same as the number of image points."

    if isinstance(annotation_color_dot, str):
        # Replicate to all points we're annotating
        annotation_color_dot = n_image_points * [annotation_color_dot]
    else:
        # Assume we've been provided an array
        assert (
            len(annotation_color_dot) == n_image_points
        ), "Expected length of annotation_color to equal the number of image points when annotation_color is an array (dot)."

    if isinstance(annotation_color_text, str):
        # Replicate to all points we're annotating
        annotation_color_text = n_image_points * [annotation_color_text]
    else:
        # Assume we've been provided an array
        assert (
            len(annotation_color_text) == n_image_points
        ), "Expected length of annotation_color to equal the number of image points when annotation_color is an array (text)."

    if isinstance(point_diameter, float) or isinstance(point_diameter, int):
        # Replicate to all points we're annotating
        point_diameters = n_image_points * [point_diameter]
    else:
        # Assume we've been provided an array
        assert (
            len(point_diameter) == n_image_points
        ), "Expected length of point_diameter to equal the number of image points when point_diameter is an array."
        point_diameters = point_diameter

    if isinstance(point_thickness, float):
        point_thickness_to_use = int(point_thickness)
    else:
        assert isinstance(
            point_thickness, int
        ), 'Expected "point_thickness" to be either an int, but it is "{}".'.format(
            type(point_thickness)
        )
        point_thickness_to_use = point_thickness

    # Copy image
    drawn_image = np.array(image).copy()

    # Support for sub-pixel circles using cv2!
    shift = 4
    factor = 1 << shift

    # Annotate the image
    for idx_point in range(n_image_points):
        point = image_points[idx_point]
        # Skip points that don't have values defined (e.g. passed as None or empty arrays)
        if point is None or len(point) == 0:
            continue

        point_fixedpt = (int(point[0] * factor), int(point[1] * factor))

        point_radius = point_diameters[idx_point] / 2.0
        radius_fixedpt = int(point_radius * factor)

        # Convert color to rgb tuple if it was passed as a name
        col = annotation_color_dot[idx_point]
        annot_color_dot = webcolors.name_to_rgb(col) if isinstance(col, str) else col
        col = annotation_color_text[idx_point]
        annot_color_text = webcolors.name_to_rgb(col) if isinstance(col, str) else col

        drawn_image = cv2.circle(
            drawn_image,
            point_fixedpt,
            radius_fixedpt,
            annot_color_dot,
            thickness=point_thickness_to_use,
            shift=shift,
        )

        if image_point_names:
            text_position = (int(point[0]) + 10, int(point[1]))
            # Manual adjustments for Baxter, frame 500
            # Also:  Baxter uses the following parameters:
            #        cv2.putText(drawn_image, image_point_names[idx_point], text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, annot_color_text, 3)
            # if idx_point == 2:  # L-S1
            #     text_position = (text_position[0], text_position[1] + 20)
            # elif idx_point == 4:  # L-E1
            #     text_position = (text_position[0], text_position[1] + 25)
            # elif idx_point == 8:  # L-Hand
            #     text_position = (text_position[0], text_position[1] + 25)
            # elif idx_point == 10:  # R-S1
            #     text_position = (text_position[0], text_position[1] + 25)
            # elif idx_point == 13:  # R-W0
            #     text_position = (text_position[0], text_position[1] + 10)
            # elif idx_point == 16:  # R-Hand
            #     text_position = (text_position[0], text_position[1] + 20)
            cv2.putText(
                drawn_image,
                image_point_names[idx_point],
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                annot_color_text,
                2,
            )

    image_as_pil = PILImage.fromarray(drawn_image)

    return image_as_pil


def image_from_tensor(image_tensor):

    # Input argument handling
    assert isinstance(
        image_tensor, torch.Tensor
    ), "Expected image_tensor to be a torch.Tensor, but it is {}.".format(
        type(image_tensor)
    )

    image = TVTransformsFunc.to_pil_image(image_tensor)

    return image


def images_from_tensor(images_tensor_batch, *args, **kwargs):

    # Input argument handling
    assert isinstance(
        images_tensor_batch, torch.Tensor
    ), "Expected images_tensor_batch to be a torch.Tensor, but it is {}.".format(
        type(images_tensor_batch)
    )

    # Expectation is that input is [num_images x num_channels x height x width]
    assert (
        len(images_tensor_batch.shape) == 4
    ), "Expected images_tensor_batch to have shape [num_images x num_channels x height x width], but it is {}.".format(
        images_tensor_batch.shape
    )

    images = []
    for image_tensor in images_tensor_batch:
        this_image = image_from_tensor(image_tensor, *args, **kwargs)
        images.append(this_image)

    return images


def image_from_belief_map(
    belief_map_tensor, normalize=True, colormap="inferno", normalization_method=6
):

    # Input argument handling
    assert isinstance(
        belief_map_tensor, torch.Tensor
    ), "Expected belief_map_tensor to be a torch.Tensor, but it is {}.".format(
        type(belief_map_tensor)
    )

    # Expectation is either [height X width] or [1 x height x width]
    if len(belief_map_tensor.shape) == 2:
        belief_map_tensor_copy = belief_map_tensor.unsqueeze(0).cpu()  # COPY TO CPU
    elif len(belief_map_tensor.shape) == 3:
        assert (
            belief_map_tensor.shape[0] == 1
        ), "Expected belief_map_tensor to have shape [height x width] or [1 x height x width], but it is {}.".format(
            belief_map_tensor.shape
        )
        belief_map_tensor_copy = belief_map_tensor.cpu()  # COPY TO CPU
    else:
        assert (
            False
        ), "Expected belief_map_tensor to have shape [height x width] or [1 x height x width], but it is {}.".format(
            belief_map_tensor.shape
        )

    belief_map_to_proc = belief_map_tensor_copy[0]

    if normalize:

        if normalization_method == 0:
            # Shift belief map so minimum element is at zero, then scale by the max
            # This is the original normalization algorithm
            belief_map_to_proc -= belief_map_to_proc.min()
            belief_map_to_proc /= belief_map_to_proc.max()
        elif normalization_method == 1:
            # Clamp values below 0 to 0, then scale by the max
            # We'll call it "zero clamp and scale"
            tensor_max = belief_map_to_proc.max()
            belief_map_to_proc = belief_map_to_proc.clamp(0.0, tensor_max)
            belief_map_to_proc /= tensor_max
        elif normalization_method == 2:
            # Clamp values below median, then scale by the max
            belief_map_to_proc -= belief_map_to_proc.median()
            tensor_max = belief_map_to_proc.max()
            belief_map_to_proc = belief_map_to_proc.clamp(0.0, tensor_max)
            belief_map_to_proc /= tensor_max
        elif normalization_method == 3:
            # Clamp values below the 25th percentile IQR, shift up, then scale by the max
            # We'll call it "IQR25 clamp and scale"
            q25 = np.percentile(belief_map_to_proc, 25)
            belief_map_to_proc -= q25
            tensor_max = belief_map_to_proc.max()
            belief_map_to_proc = belief_map_to_proc.clamp(0.0, tensor_max)
            belief_map_to_proc /= tensor_max
        elif normalization_method == 4:
            # Clamp values below the 75th percentile IQR, shift up, then scale by the max
            # We'll call it "IQR25 clamp and scale"
            q75 = np.percentile(belief_map_to_proc, 75)
            belief_map_to_proc -= q75
            tensor_max = belief_map_to_proc.max()
            belief_map_to_proc = belief_map_to_proc.clamp(0.0, tensor_max)
            belief_map_to_proc /= tensor_max
        elif normalization_method == 5:
            # Clamp values below 0 to 0, then don't do anything with the max
            # This helps if we know that generally the belief map should be within 0 to 1
            # Sometimes the belief maps are slightly negative
            # We'll call it "zero clamp"
            tensor_max = belief_map_to_proc.max()
            belief_map_to_proc = belief_map_to_proc.clamp(0.0, tensor_max)
        elif normalization_method == 6:
            # Clamp values below 0 to 0, then clamp values above 1 to 1
            # This helps if we know that generally the belief map should be within 0 to 1
            # Sometimes the belief maps are slightly negative or slightly above 1
            # We'll call it "zero-one clamp"
            belief_map_to_proc = belief_map_to_proc.clamp(0.0, 1.0)
        else:
            assert False, "Normalization method not defined."

    belief_map_image = TVTransformsFunc.to_pil_image(belief_map_to_proc)

    if colormap:
        cmap = plt.get_cmap(colormap)
        rgba_image = cmap(np.array(belief_map_image))
        heatmap_image = np.delete(rgba_image, 3, 2)
        belief_map_image = PILImage.fromarray(np.uint8(255 * heatmap_image))

    return belief_map_image


def images_from_belief_maps(belief_maps_tensor, *args, **kwargs):

    # Input argument handling
    assert isinstance(
        belief_maps_tensor, torch.Tensor
    ), "Expected belief_maps_tensor to be a torch.Tensor, but it is {}.".format(
        type(belief_maps_tensor)
    )

    # Expectation is that input is [num_maps x height x width]
    assert (
        len(belief_maps_tensor.shape) == 3
    ), "Expected belief_maps_tensor to have shape [num_maps x height x width], but it is {}.".format(
        belief_maps_tensor.shape
    )

    belief_map_images = []
    for n in range(belief_maps_tensor.shape[0]):
        this_belief_map_image = image_from_belief_map(
            belief_maps_tensor[n], *args, **kwargs
        )
        belief_map_images.append(this_belief_map_image)

    return belief_map_images


def mosaic_images(
    image_array_input,
    rows=None,
    cols=None,
    outer_padding_px=0,
    inner_padding_px=0,
    fill_color_rgb=(255, 255, 255),
):

    # Input argument handling
    assert (
        image_array_input
        and len(image_array_input) > 0
        and not isinstance(image_array_input, str)
    ), "Expected image_array_input to be an array of image inputs, but it is {}.".format(
        type(image_array_input)
    )

    # Check whether we were provided PIL images or paths to images
    if isinstance(image_array_input[0], str):
        # Assume we're given image paths
        image_array = [
            PILImage.open(img_path).convert("RGB") for img_path in image_array_input
        ]
    else:
        # Assume we're given PIL images
        assert isinstance(
            image_array_input[0], PILImage.Image
        ), 'Expected "image_array_input" to contain either image paths or PIL Images, but it is "{}".'.format(
            type(image_array_input[0])
        )
        image_array = image_array_input

    # Verify that all images have the same resolution. Necessary for the mosaic image math to work correctly.
    n_images = len(image_array)
    image_width, image_height = image_array[0].size
    for image in image_array:
        this_image_width, this_image_height = image.size
        assert (
            this_image_width == image_width and this_image_height == image_height
        ), "All images must have the same resolution."

    # Handle rows and cols inputs
    assert (
        rows or cols
    ), "Expected either rows or cols (or both) to be specified, but neither are."

    if rows:
        assert (
            isinstance(rows, int) and rows > 0
        ), "If specified, expected rows to be a positive integer, but it is {} with value {}.".format(
            type(rows), rows
        )
    else:
        # Calculate rows from cols
        rows = int(math.ceil(float(n_images) / float(cols)))

    if cols:
        assert (
            isinstance(cols, int) and cols > 0
        ), "If specified, expected cols to be a positive integer, but it is {} with value {}.".format(
            type(cols), cols
        )
    else:
        # Calculate cols from rows
        cols = int(math.ceil(float(n_images) / float(rows)))

    assert (
        rows * cols >= n_images
    ), "The number of mosaic rows and columns is too small for the number of input images."

    assert (
        isinstance(outer_padding_px, int) and outer_padding_px >= 0
    ), "Expected outer_padding_px to be an integer with value greater than or equal to zero, but it is {} with value {}".format(
        type(outer_padding_px), outer_padding_px
    )

    assert (
        isinstance(inner_padding_px, int) and inner_padding_px >= 0
    ), "Expected inner_padding_px to be an integer with value greater than or equal to zero, but it is {} with value {}".format(
        type(inner_padding_px), inner_padding_px
    )

    assert (
        len(fill_color_rgb) == 3
    ), "Expected fill_color_rgb to be a RGB array of length 3, but it has length {}.".format(
        len(fill_color_rgb)
    )

    # Construct mosaic
    mosaic = PILImage.new(
        "RGB",
        (
            cols * image_width + 2 * outer_padding_px + (cols - 1) * inner_padding_px,
            rows * image_height + 2 * outer_padding_px + (rows - 1) * inner_padding_px,
        ),
        fill_color_rgb,
    )

    # Paste images into mosaic
    img_idx = 0
    for r in range(rows):
        for c in range(cols):
            if img_idx < n_images:
                img_loc = (
                    c * image_width + outer_padding_px + c * inner_padding_px,
                    r * image_height + outer_padding_px + r * inner_padding_px,
                )
                mosaic.paste(image_array[img_idx], img_loc)
                img_idx += 1

    return mosaic


def create_belief_map(
    image_resolution,
    # image size (width x height)
    pointsBelief,
    # list of points to draw in a 7x2 tensor
    sigma=2
    # the size of the point
    # returns a tensor of n_points x h x w with the belief maps
):

    # Input argument handling
    assert (
        len(image_resolution) == 2
    ), 'Expected "image_resolution" to have length 2, but it has length {}.'.format(
        len(image_resolution)
    )
    image_width, image_height = image_resolution
    image_transpose_resolution = (image_height, image_width)
    out = np.zeros((len(pointsBelief), image_height, image_width))

    w = int(sigma * 2)

    for i_point, point in enumerate(pointsBelief):
        pixel_u = int(point[0])
        pixel_v = int(point[1])
        array = np.zeros(image_transpose_resolution)

        # TODO makes this dynamics so that 0,0 would generate a belief map.
        if (
            pixel_u - w >= 0
            and pixel_u + w + 1 < image_width
            and pixel_v - w >= 0
            and pixel_v + w + 1 < image_height
        ):
            for i in range(pixel_u - w, pixel_u + w + 1):
                for j in range(pixel_v - w, pixel_v + w + 1):
                    array[j, i] = np.exp(
                        -(
                            ((i - pixel_u) ** 2 + (j - pixel_v) ** 2)
                            / (2 * (sigma ** 2))
                        )
                    )
        out[i_point] = array

    return out


# Code adapted from code originally written by Jon Tremblay
def peaks_from_belief_maps(belief_map_tensor, offset_due_to_upsampling):
    # print("pfbm**************************************")

    assert (
        len(belief_map_tensor.shape) == 3
    ), "Expected belief_map_tensor to have shape [N x height x width], but it is {}.".format(
        belief_map_tensor.shape
    )

    # thresh_map_after_gaussian_filter specifies the minimum intensity in the belief map AFTER the gaussian filter.
    # with sigma = 3, a perfect heat map will have a max intensity of about 0.3. -- both in a 100x100 frame and a 400x400 frame
    thresh_map_after_gaussian_filter = 0.01
    sigma = 3

    all_peaks = []
    peak_counter = 0

    for j in range(belief_map_tensor.size()[0]):
        belief_map = belief_map_tensor[j].clone()
        map_ori = belief_map.cpu().data.numpy()

        map = gaussian_filter(map_ori, sigma=sigma)
        p = 1
        map_left = np.zeros(map.shape)
        map_left[p:, :] = map[:-p, :]
        map_right = np.zeros(map.shape)
        map_right[:-p, :] = map[p:, :]
        map_up = np.zeros(map.shape)
        map_up[:, p:] = map[:, :-p]
        map_down = np.zeros(map.shape)
        map_down[:, :-p] = map[:, p:]

        peaks_binary = np.logical_and.reduce(
            (
                map >= map_left,
                map >= map_right,
                map >= map_up,
                map >= map_down,
                map > thresh_map_after_gaussian_filter,
            )
        )
        peaks = zip(
            np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]  # x values
        )  # y values

        # Computing the weigthed average for localizing the peaks
        peaks = list(peaks)
        win = 5
        ran = win // 2
        peaks_avg = []
        for p_value in range(len(peaks)):
            p = peaks[p_value]
            weights = np.zeros((win, win))
            i_values = np.zeros((win, win))
            j_values = np.zeros((win, win))
            for i in range(-ran, ran + 1):
                for j in range(-ran, ran + 1):
                    if (
                        p[1] + i < 0
                        or p[1] + i >= map_ori.shape[0]
                        or p[0] + j < 0
                        or p[0] + j >= map_ori.shape[1]
                    ):
                        continue

                    i_values[j + ran, i + ran] = p[1] + i
                    j_values[j + ran, i + ran] = p[0] + j

                    weights[j + ran, i + ran] = map_ori[p[1] + i, p[0] + j]

            # if the weights are all zeros
            # then add the none continuous points
            try:
                peaks_avg.append(
                    (
                        np.average(j_values, weights=weights)
                        + offset_due_to_upsampling,
                        np.average(i_values, weights=weights)
                        + offset_due_to_upsampling,
                    )
                )
            except:
                peaks_avg.append(
                    (p[0] + offset_due_to_upsampling, p[1] + offset_due_to_upsampling)
                )
        # Note: Python3 doesn't support len for zip object
        peaks_len = min(
            len(np.nonzero(peaks_binary)[1]), len(np.nonzero(peaks_binary)[0])
        )

        peaks_with_score = [
            peaks_avg[x_] + (map_ori[peaks[x_][1], peaks[x_][0]],)
            for x_ in range(len(peaks))
        ]

        id = range(peak_counter, peak_counter + peaks_len)

        peaks_with_score_and_id = [
            peaks_with_score[i] + (id[i],) for i in range(len(id))
        ]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += peaks_len

    return all_peaks
