# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import argparse
import math
import os
from PIL import Image as PILImage

import numpy as np
from ruamel.yaml import YAML
import torch
import torchvision.transforms as TVTransforms

import dream

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def generate_belief_map_visualizations(
    belief_maps, keypoint_projs_detected, keypoint_projs_gt=None
):

    belief_map_images = dream.image_proc.images_from_belief_maps(
        belief_maps, normalization_method=6
    )

    belief_map_images_kp = []
    for kp in range(len(keypoint_projs_detected)):
        if keypoint_projs_gt:
            keypoint_projs = [keypoint_projs_gt[kp], keypoint_projs_detected[kp]]
            color = ["green", "red"]
        else:
            keypoint_projs = [keypoint_projs_detected[kp]]
            color = "red"
        belief_map_image_kp = dream.image_proc.overlay_points_on_image(
            belief_map_images[kp],
            keypoint_projs,
            annotation_color_dot=color,
            annotation_color_text=color,
            point_diameter=4,
        )
        belief_map_images_kp.append(belief_map_image_kp)
    n_cols = int(math.ceil(len(keypoint_projs_detected) / 2.0))
    belief_maps_kp_mosaic = dream.image_proc.mosaic_images(
        belief_map_images_kp,
        rows=2,
        cols=n_cols,
        inner_padding_px=10,
        fill_color_rgb=(0, 0, 0),
    )
    return belief_maps_kp_mosaic


def network_inference(args):

    # Input argument handling
    assert os.path.exists(
        args.input_params_path
    ), 'Expected input_params_path "{}" to exist, but it does not.'.format(
        args.input_params_path
    )

    if args.input_config_path:
        input_config_path = args.input_config_path
    else:
        # Use params filepath to infer the config filepath
        input_config_path = os.path.splitext(args.input_params_path)[0] + ".yaml"

    assert os.path.exists(
        input_config_path
    ), 'Expected input_config_path "{}" to exist, but it does not.'.format(
        input_config_path
    )

    assert os.path.exists(
        args.image_path
    ), 'Expected image_path "{}" to exist, but it does not.'.format(args.image_path)

    # Create parser
    print("# Opening config file:  {} ...".format(input_config_path))
    data_parser = YAML(typ="safe")

    with open(input_config_path, "r") as f:
        network_config = data_parser.load(f)

    # Overwrite GPU
    # If nothing is specified at the command line, None is the default, which uses all GPUs
    # TBD - think about a better way of doing this
    network_config["training"]["platform"]["gpu_ids"] = args.gpu_ids

    # Load network
    print("# Creating network...")
    dream_network = dream.create_network_from_config_data(network_config)

    print("Loading network with weights from:  {} ...".format(args.input_params_path))
    dream_network.model.load_state_dict(torch.load(args.input_params_path))
    dream_network.enable_evaluation()

    # Load in image
    print("# Loading image:  {} ...".format(args.image_path))
    image_rgb_OrigInput_asPilImage = PILImage.open(args.image_path).convert("RGB")
    orig_image_dim = tuple(image_rgb_OrigInput_asPilImage.size)

    # Use image preprocessing specified by config by default, unless user specifies otherwise
    if args.image_preproc_override:
        image_preprocessing = args.image_preproc_override
        print(
            "  Image preprocessing: '{}' --- as specified by user".format(
                image_preprocessing
            )
        )
    else:
        image_preprocessing = dream_network.image_preprocessing()
        print(
            "  Image preprocessing: '{}' --- default as specified by network config".format(
                image_preprocessing
            )
        )

    print("Detecting keypoints...")
    detection_result = dream_network.keypoints_from_image(
        image_rgb_OrigInput_asPilImage,
        image_preprocessing_override=image_preprocessing,
        debug=True,
    )
    kp_coords_wrtOrigInput_asArray = detection_result["detected_keypoints"]
    print(
        "Detected keypoints in input image:\n{}".format(kp_coords_wrtOrigInput_asArray)
    )

    kp_coords_wrtNetOutput_asArray = detection_result["detected_keypoints_net_output"]
    image_rgb_NetInput_asPilImage = detection_result["image_rgb_net_input"]
    input_image_dim = image_rgb_NetInput_asPilImage.size
    belief_maps_wrtNetOutput_asTensor = detection_result["belief_maps"]
    kp_coords_wrtNetInput_asArray = detection_result["detected_keypoints_net_input"]

    # Read in keypoints if provided
    if args.keypoints_path:
        print(
            "# Loading ground truth keypoints from {} ...".format(args.keypoints_path)
        )
        keypoints_gt = dream.utilities.load_keypoints(
            args.keypoints_path,
            dream_network.manipulator_name,
            dream_network.keypoint_names,
        )
        kp_coords_gt_wrtOrig = keypoints_gt["projections"]
        print(
            "Ground truth keypoints in input image:\n{}".format(
                np.array(kp_coords_gt_wrtOrig)
            )
        )

        kp_coords_gt_wrtNetInput_asArray = dream.image_proc.convert_keypoints_to_netin_from_raw(
            kp_coords_gt_wrtOrig,
            orig_image_dim,
            dream_network.trained_net_input_resolution(),
            image_preprocessing,
        )
        kp_coords_gt_wrtNetOutput_asArray = dream.image_proc.convert_keypoints_to_netout_from_netin(
            kp_coords_gt_wrtNetInput_asArray,
            dream_network.trained_net_input_resolution(),
            dream_network.trained_net_output_resolution(),
        )
        kp_coords_gt_wrtNetInput_asList = kp_coords_gt_wrtNetInput_asArray.tolist()
        kp_coords_gt_wrtNetOutput_asList = kp_coords_gt_wrtNetOutput_asArray.tolist()
    else:
        print("# Not loading ground truth keypoints (not provided)")
        kp_coords_gt_wrtNetInput_asList = None
        kp_coords_gt_wrtNetOutput_asList = None

    # Generate visualization output:  keypoints, with ground truth if requested) overlaid on image used for network input
    keypoints_wrtNetInput_overlay = dream.image_proc.overlay_points_on_image(
        image_rgb_NetInput_asPilImage,
        kp_coords_gt_wrtNetInput_asList,
        dream_network.friendly_keypoint_names,
        annotation_color_dot="green",
        annotation_color_text="white",
    )
    keypoints_wrtNetInput_overlay = dream.image_proc.overlay_points_on_image(
        keypoints_wrtNetInput_overlay,
        kp_coords_wrtNetInput_asArray,
        dream_network.friendly_keypoint_names,
        annotation_color_dot="red",
        annotation_color_text="white",
    )
    keypoints_wrtNetInput_overlay.show(
        title="Keypoints (possibly with ground truth) on net input image"
    )

    # Generate visualization output:  mosaic of raw belief maps from network
    belief_maps_overlay = generate_belief_map_visualizations(
        belief_maps_wrtNetOutput_asTensor,
        kp_coords_wrtNetOutput_asArray,
        kp_coords_gt_wrtNetOutput_asList,
    )
    belief_maps_overlay.show(title="Belief map output mosaic")

    # Generate visualization output:  mosaic of belief maps, with keypoints, overlaid on image used for network input
    belief_maps_wrtNetOutput_asListOfPilImages = dream.image_proc.images_from_belief_maps(
        belief_maps_wrtNetOutput_asTensor, normalization_method=6
    )
    blended_array = []

    for n in range(len(kp_coords_wrtNetOutput_asArray)):

        bm_wrtNetOutput_asPilImage = belief_maps_wrtNetOutput_asListOfPilImages[n]
        kp = kp_coords_wrtNetInput_asArray[n]
        fname = dream_network.friendly_keypoint_names[n]

        bm_wrtNetInput_asPilImage = bm_wrtNetOutput_asPilImage.resize(
            input_image_dim, resample=PILImage.BILINEAR
        )
        blended = PILImage.blend(
            image_rgb_NetInput_asPilImage, bm_wrtNetInput_asPilImage, alpha=0.5
        )
        blended = dream.image_proc.overlay_points_on_image(
            blended,
            [kp],
            [fname],
            annotation_color_dot="red",
            annotation_color_text="white",
        )
        blended_array.append(blended)

    n_cols = int(math.ceil(len(kp_coords_wrtNetOutput_asArray) / 2.0))
    belief_maps_with_kp_overlaid_mosaic = dream.image_proc.mosaic_images(
        blended_array, rows=2, cols=n_cols, fill_color_rgb=(0, 0, 0)
    )
    belief_maps_with_kp_overlaid_mosaic.show(
        title="Mosaic of belief maps, with keypoints, on original"
    )

    # Squash belief maps into one combined image
    belief_map_combined_wrtNetOutput_asTensor = belief_maps_wrtNetOutput_asTensor.sum(
        dim=0
    )
    belief_map_combined_wrtNetOutput_asPilImage = dream.image_proc.image_from_belief_map(
        belief_map_combined_wrtNetOutput_asTensor, normalization_method=6
    )  # clamps to b/w 0 and 1
    belief_map_combined_wrtNetInput_asPilImage = dream.image_proc.convert_image_to_netin_from_netout(
        belief_map_combined_wrtNetOutput_asPilImage, input_image_dim
    )

    # Generate visualization output:  belief maps, with keypoints, overlaid on image used for network input
    belief_map_combined_wrtNetInput_overlay = PILImage.blend(
        image_rgb_NetInput_asPilImage,
        belief_map_combined_wrtNetInput_asPilImage,
        alpha=0.5,
    )
    belief_map_combined_wrtNetInput_overlay = dream.image_proc.overlay_points_on_image(
        belief_map_combined_wrtNetInput_overlay,
        kp_coords_wrtNetInput_asArray,
        dream_network.friendly_keypoint_names,
        annotation_color_dot="red",
        annotation_color_text="white",
    )
    belief_map_combined_wrtNetInput_overlay.show(
        title="Belief maps, with keypoints, on net input image"
    )

    # Generate visualization output:  belief maps, with keypoints, overlaid on original image
    belief_map_combined_wrtOrigInput_asPilImage = dream.image_proc.inverse_preprocess_image(
        belief_map_combined_wrtNetInput_asPilImage, orig_image_dim, image_preprocessing
    )
    belief_map_combined_wrtOrigInput_overlay = PILImage.blend(
        image_rgb_OrigInput_asPilImage,
        belief_map_combined_wrtOrigInput_asPilImage,
        alpha=0.5,
    )
    belief_map_combined_wrtOrigInput_overlay = dream.image_proc.overlay_points_on_image(
        belief_map_combined_wrtOrigInput_overlay,
        kp_coords_wrtOrigInput_asArray,
        dream_network.friendly_keypoint_names,
        annotation_color_dot="red",
        annotation_color_text="white",
    )
    belief_map_combined_wrtOrigInput_overlay.show(
        title="Belief maps, with keypoints, on original image"
    )

    print("Done.")


if __name__ == "__main__":

    print(
        "---------- Running 'network_inference.py' -------------------------------------------------"
    )

    # Parse input arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input-params-path",
        required=True,
        help="Path to network parameters file.",
    )
    parser.add_argument(
        "-c",
        "--input-config-path",
        default=None,
        help="Path to network configuration file. If nothing is specified, the script will search for a config file by the same name as the network parameters file.",
    )
    parser.add_argument(
        "-m", "--image_path", required=True, help="Path to image used for inference."
    )
    parser.add_argument(
        "-k",
        "--keypoints_path",
        default=None,
        help="Path to NDDS dataset with ground truth keypoints information.",
    )
    parser.add_argument(
        "-g",
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="The GPU IDs on which to conduct network inference. Nothing specified means all GPUs will be utilized. Does not affect results, only how quickly the results are obtained.",
    )
    parser.add_argument(
        "-p",
        "--image-preproc-override",
        default=None,
        help="Overrides the image preprocessing specified by the network. (Debug argument.)",
    )
    args = parser.parse_args()

    # Run network inference
    network_inference(args)
