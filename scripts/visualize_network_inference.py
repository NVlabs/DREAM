# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import argparse
import os
from PIL import Image as PILImage
import shutil
import subprocess
import sys

import numpy as np
from ruamel.yaml import YAML
import torch
from torch.utils.data import DataLoader as TorchDataLoader
import torchvision.transforms as TVTransforms
from tqdm import tqdm

import dream

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def video_from_frames(frames_dir, video_output_path, video_framerate):
    force_str = "-y"
    loglevel_str = "-loglevel 24"
    framerate_str = "-framerate {}".format(video_framerate)
    input_data_str = '-pattern_type glob -i "{}"'.format(
        os.path.join(frames_dir, "*.png")
    )
    output_vid_str = '"{}"'.format(video_output_path)
    encoding_str = "-vcodec libx264 -pix_fmt yuv420p"
    ffmpeg_vid_cmd = (
        "ffmpeg "
        + force_str
        + " "
        + loglevel_str
        + " "
        + framerate_str
        + " "
        + input_data_str
        + " "
        + encoding_str
        + " "
        + output_vid_str
    )

    print("Running command: {}".format(ffmpeg_vid_cmd))
    subprocess.call(ffmpeg_vid_cmd, shell=True)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

KP_OVERLAY_RAW = "kp_raw"
KP_OVERLAY_NET_INPUT = "kp_net_input"
KP_BELIEF_OVERLAY_RAW = "kp_belief_raw"
BELIEF_OVERLAY_RAW = "belief_raw"


def visualize_network_inference(args):

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
        args.dataset_path
    ), 'Expected dataset_path "{}" to exist, but it does not.'.format(args.dataset_path)

    # Determine what types of visualizations to do
    print("visualization types: {}".format(args.visualization_types))
    do_kp_overlay_raw = True if KP_OVERLAY_RAW in args.visualization_types else False
    do_kp_overlay_net_input = (
        True if KP_OVERLAY_NET_INPUT in args.visualization_types else False
    )
    do_kp_belief_overlay_raw = (
        True if KP_BELIEF_OVERLAY_RAW in args.visualization_types else False
    )
    do_belief_overlay_raw = (
        True if BELIEF_OVERLAY_RAW in args.visualization_types else False
    )

    videos_to_make = []
    needs_belief_maps = False
    if do_kp_overlay_raw:
        idx_kp_overlay_raw = len(videos_to_make)
        videos_to_make.append(
            {
                "frames_dir": os.path.join(args.output_dir, "frames_kp_overlay_raw"),
                "output_path": os.path.join(args.output_dir, "kp_overlay_raw.mp4"),
                "frame": [],
            }
        )
    if do_kp_overlay_net_input:
        idx_kp_overlay_net_input = len(videos_to_make)
        videos_to_make.append(
            {
                "frames_dir": os.path.join(
                    args.output_dir, "frames_kp_overlay_net_input"
                ),
                "output_path": os.path.join(
                    args.output_dir, "kp_overlay_net_input.mp4"
                ),
                "frame": [],
            }
        )
    if do_kp_belief_overlay_raw:
        idx_kp_belief_overlay_raw = len(videos_to_make)
        needs_belief_maps = True
        videos_to_make.append(
            {
                "frames_dir": os.path.join(
                    args.output_dir, "frames_kp_belief_overlay_raw"
                ),
                "output_path": os.path.join(
                    args.output_dir, "kp_belief_overlay_raw.mp4"
                ),
                "frame": [],
            }
        )
    if do_belief_overlay_raw:
        idx_belief_overlay_raw = len(videos_to_make)
        needs_belief_maps = True
        videos_to_make.append(
            {
                "frames_dir": os.path.join(
                    args.output_dir, "frames_belief_overlay_raw"
                ),
                "output_path": os.path.join(args.output_dir, "belief_overlay_raw.mp4"),
                "frame": [],
            }
        )

    if len(videos_to_make) == 0:
        print("No visualizations have been selected.")
        sys.exit(0)

    dream.utilities.makedirs(args.output_dir, exist_ok=args.force_overwrite)
    for video in videos_to_make:
        if os.path.exists(video["frames_dir"]):
            assert args.force_overwrite, 'Frames directory "{}" already exists.'.format(
                video["frames_dir"]
            )
            shutil.rmtree(video["frames_dir"])
        dream.utilities.makedirs(video["frames_dir"], exist_ok=args.force_overwrite)

    # Create parser
    data_parser = YAML(typ="safe")

    with open(input_config_path, "r") as f:
        network_config = data_parser.load(f)

    # Overwrite GPU
    # If nothing is specified at the command line, None is the default, which uses all GPUs
    # TBD - think about a better way of doing this
    network_config["training"]["platform"]["gpu_ids"] = args.gpu_ids

    # Load network
    dream_network = dream.create_network_from_config_data(network_config)
    dream_network.model.load_state_dict(torch.load(args.input_params_path))
    dream_network.enable_evaluation()

    # Use image preprocessing specified by config by default, unless user specifies otherwise
    if args.image_preproc_override:
        image_preprocessing = args.image_preproc_override
    else:
        image_preprocessing = dream_network.image_preprocessing()

    if args.keypoint_ids is None or len(args.keypoint_ids) == 0:
        idx_keypoints = list(range(dream_network.n_keypoints))
    else:
        idx_keypoints = args.keypoint_ids
    n_idx_keypoints = len(idx_keypoints)

    sample_results = []

    dataset_to_viz = dream.utilities.find_ndds_data_in_dir(args.dataset_path)
    dataset_file_dict_list = dataset_to_viz[
        0
    ]  # list of data file dictionaries; each dictionary indicates the files names for rgb, depth, seg, ...
    dataset_meta_dict = dataset_to_viz[1]  # dictionary of camera, object files, etc.

    if dataset_file_dict_list:

        # Downselect based on frame name
        if args.start_frame or args.end_frame:
            sample_names = [x["name"] for x in dataset_file_dict_list]
            start_idx = sample_names.index(args.start_frame) if args.start_frame else 0
            end_idx = (
                sample_names.index(args.end_frame) + 1
                if args.end_frame
                else len(dataset_file_dict_list)
            )

            dataset_to_viz = (
                dataset_file_dict_list[start_idx:end_idx],
                dataset_meta_dict,
            )

        image_raw_resolution = dream.utilities.load_image_resolution(
            dataset_meta_dict["camera"]
        )
        (
            network_input_res_inf,
            network_output_res_inf,
        ) = dream_network.net_resolutions_from_image_raw_resolution(
            image_raw_resolution, image_preprocessing_override=image_preprocessing
        )

        manip_dataset_debug_mode = dream.datasets.ManipulatorNDDSDatasetDebugLevels[
            "LIGHT"
        ]
        manip_dataset = dream.datasets.ManipulatorNDDSDataset(
            dataset_to_viz,
            dream_network.manipulator_name,
            dream_network.keypoint_names,
            network_input_res_inf,
            network_output_res_inf,
            dream_network.image_normalization,
            image_preprocessing,
            augment_data=False,
            include_ground_truth=not args.no_ground_truth,
            debug_mode=manip_dataset_debug_mode,
        )

        # TODO: set batch size and num_workers at command line
        batch_size = 8
        num_workers = 4
        training_data = TorchDataLoader(
            manip_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )

        # Network inference on dataset
        with torch.no_grad():

            for batch_idx, sample in enumerate(tqdm(training_data)):

                this_batch_size = len(sample["config"]["name"])

                # Conduct inference
                network_image_input = sample["image_rgb_input"].cuda()
                (
                    belief_maps_batch,
                    detected_kp_projs_netout_batch,
                ) = dream_network.inference(network_image_input)

                for b in range(this_batch_size):

                    input_image_path = sample["config"]["image_paths"]["rgb"][b]

                    if needs_belief_maps:
                        belief_maps = belief_maps_batch[b]
                        selected_belief_maps_copy = (
                            belief_maps[idx_keypoints, :, :].detach().clone()
                        )
                    else:
                        selected_belief_maps_copy = []

                    detected_kp_projs_netout = np.array(
                        detected_kp_projs_netout_batch[b], dtype=float
                    )
                    selected_detected_kp_projs_netout = detected_kp_projs_netout[
                        idx_keypoints, :
                    ]
                    selected_detected_kp_projs_netin = dream.image_proc.convert_keypoints_to_netin_from_netout(
                        selected_detected_kp_projs_netout,
                        network_output_res_inf,
                        network_input_res_inf,
                    )
                    selected_detected_kp_projs_raw = dream.image_proc.convert_keypoints_to_raw_from_netin(
                        selected_detected_kp_projs_netin,
                        network_input_res_inf,
                        image_raw_resolution,
                        image_preprocessing,
                    )

                    if args.no_ground_truth:
                        selected_gt_kp_projs_raw = []
                        selected_gt_kp_projs_netin = []
                    else:
                        selected_gt_kp_projs_raw = np.array(
                            sample["keypoint_projections_raw"][b][idx_keypoints, :],
                            dtype=float,
                        )
                        selected_gt_kp_projs_netin = np.array(
                            sample["keypoint_projections_input"][b][idx_keypoints, :],
                            dtype=float,
                        )

                    input_image_raw = PILImage.open(input_image_path).convert("RGB")
                    image_net_input = dream.image_proc.image_from_tensor(
                        sample["image_rgb_input_viz"][b]
                    )

                    sample_results.append(
                        (
                            input_image_raw,
                            image_net_input,
                            selected_belief_maps_copy,
                            selected_detected_kp_projs_raw,
                            selected_detected_kp_projs_netin,
                            selected_gt_kp_projs_raw,
                            selected_gt_kp_projs_netin,
                        )
                    )

    else:
        # Probably a directory of images - fix this later to avoid code duplication
        dirlist = os.listdir(args.dataset_path)
        dirlist.sort()
        png_image_names = [f for f in dirlist if f.endswith(".png")]
        jpg_image_names = [f for f in dirlist if f.endswith(".jpg")]
        image_names = (
            png_image_names
            if len(png_image_names) > len(jpg_image_names)
            else jpg_image_names
        )

        if args.start_frame or args.end_frame:
            sample_names = [os.path.splitext(i)[0] for i in image_names]
            start_idx = sample_names.index(args.start_frame) if args.start_frame else 0
            end_idx = (
                sample_names.index(args.end_frame) + 1
                if args.end_frame
                else len(sample_names)
            )

            image_names = image_names[start_idx:end_idx]

        # Just use a heuristic to determine the image extension
        image_paths = [os.path.join(args.dataset_path, i) for i in image_names]

        for input_image_path in tqdm(image_paths):

            input_image_raw = PILImage.open(input_image_path).convert("RGB")
            detection_result = dream_network.keypoints_from_image(
                input_image_raw,
                image_preprocessing_override=image_preprocessing,
                debug=True,
            )

            selected_detected_kps_raw = detection_result["detected_keypoints"][
                idx_keypoints, :
            ]
            selected_detected_kps_netin = detection_result[
                "detected_keypoints_net_input"
            ][idx_keypoints, :]
            image_net_input = detection_result["image_rgb_net_input"]
            selected_belief_maps = (
                detection_result["belief_maps"][idx_keypoints, :, :]
                if needs_belief_maps
                else []
            )
            selected_gt_kps_raw = []
            selected_gt_kps_netin = []

            sample_results.append(
                (
                    input_image_raw,
                    image_net_input,
                    selected_belief_maps,
                    selected_detected_kps_raw,
                    selected_detected_kps_netin,
                    selected_gt_kps_raw,
                    selected_gt_kps_netin,
                )
            )

    # Iterate through inferred results
    idx_this_frame = 1
    print("Creating visualizations...")
    for (
        image_raw,
        input_image,
        belief_maps,
        detected_kp_projs_raw,
        detected_kp_projs_net_input,
        gt_kp_projs_raw,
        gt_kp_projs_net_input,
    ) in tqdm(sample_results):

        show_gt_keypoints = (not args.no_ground_truth) and len(gt_kp_projs_raw) > 0

        image_raw_resolution = image_raw.size
        net_input_resolution = input_image.size

        if do_kp_overlay_net_input:
            videos_to_make[idx_kp_overlay_net_input]["frame"] = input_image

        if do_kp_overlay_raw:
            videos_to_make[idx_kp_overlay_raw]["frame"] = image_raw

        if do_kp_belief_overlay_raw:
            flattened_belief_tensor = belief_maps.sum(dim=0)
            flattened_belief_image = dream.image_proc.image_from_belief_map(
                flattened_belief_tensor, colormap="hot", normalization_method=6
            )
            flattened_belief_image_netin = dream.image_proc.convert_image_to_netin_from_netout(
                flattened_belief_image, net_input_resolution
            )
            flattened_belief_image_raw = dream.image_proc.inverse_preprocess_image(
                flattened_belief_image_netin, image_raw_resolution, image_preprocessing
            )
            videos_to_make[idx_kp_belief_overlay_raw]["frame"] = PILImage.blend(
                image_raw, flattened_belief_image_raw, alpha=0.5
            )

            # Previous code here, but the overlays don't look as nice
            # Note - this seems pretty slow
            # I = np.asarray(flattened_belief_image_raw.convert('L'))
            # I_black = I < 20
            # mask = PILImage.fromarray(np.uint8(255*I_black))
            # temp = PILImage.composite(image_raw, flattened_belief_image_raw, mask)
            # videos_to_make[idx_kp_belief_overlay_raw]['frame'] = PILImage.blend(image_raw, temp, alpha=0.75)
            # #PILImage.alpha_composite(flattened_belief_image_raw.convert('RGBA'), image_raw.convert('RGBA'))

        if do_belief_overlay_raw:
            flattened_belief_tensor = belief_maps.sum(dim=0)
            flattened_belief_image = dream.image_proc.image_from_belief_map(
                flattened_belief_tensor, colormap="hot", normalization_method=6
            )
            flattened_belief_image_netin = dream.image_proc.convert_image_to_netin_from_netout(
                flattened_belief_image, net_input_resolution
            )
            flattened_belief_image_raw = dream.image_proc.inverse_preprocess_image(
                flattened_belief_image_netin, image_raw_resolution, image_preprocessing
            )
            videos_to_make[idx_belief_overlay_raw]["frame"] = PILImage.blend(
                image_raw, flattened_belief_image_raw, alpha=0.5
            )

        for n in range(n_idx_keypoints):
            detected_kp_proj_raw = detected_kp_projs_raw[n, :]
            detected_kp_proj_net_input = detected_kp_projs_net_input[n, :]

            if show_gt_keypoints:
                gt_kp_proj_raw = gt_kp_projs_raw[n, :]
                gt_kp_proj_net_input = gt_kp_projs_net_input[n, :]

            # Overlay
            if do_kp_overlay_net_input:
                # Heuristic to make point diameter look good for larger raw resolutions
                pt_diameter = (
                    12.0
                    if image_raw_resolution[0] * image_raw_resolution[1] > 500000
                    else 6.0
                )
                if show_gt_keypoints:
                    videos_to_make[idx_kp_overlay_net_input][
                        "frame"
                    ] = dream.image_proc.overlay_points_on_image(
                        videos_to_make[idx_kp_overlay_net_input]["frame"],
                        [gt_kp_proj_net_input],
                        annotation_color_dot="green",
                        annotation_color_text="white",
                        point_thickness=2,
                        point_diameter=pt_diameter,
                    )

                videos_to_make[idx_kp_overlay_net_input][
                    "frame"
                ] = dream.image_proc.overlay_points_on_image(
                    videos_to_make[idx_kp_overlay_net_input]["frame"],
                    [detected_kp_proj_net_input],
                    annotation_color_dot="red",
                    annotation_color_text="white",
                    point_diameter=pt_diameter,
                )

            if do_kp_overlay_raw:
                # Heuristic to make point diameter look good for larger raw resolutions
                pt_diameter = (
                    12.0
                    if image_raw_resolution[0] * image_raw_resolution[1] > 500000
                    else 6.0
                )

                if show_gt_keypoints:
                    videos_to_make[idx_kp_overlay_raw][
                        "frame"
                    ] = dream.image_proc.overlay_points_on_image(
                        videos_to_make[idx_kp_overlay_raw]["frame"],
                        [gt_kp_proj_raw],
                        annotation_color_dot="green",
                        annotation_color_text="white",
                        point_thickness=2,
                        point_diameter=pt_diameter + 2,
                    )

                videos_to_make[idx_kp_overlay_raw][
                    "frame"
                ] = dream.image_proc.overlay_points_on_image(
                    videos_to_make[idx_kp_overlay_raw]["frame"],
                    [detected_kp_proj_raw],
                    annotation_color_dot="red",
                    annotation_color_text="white",
                    point_diameter=pt_diameter,
                )

            if do_kp_belief_overlay_raw:
                # Heuristic to make point diameter look good for larger raw resolutions
                pt_diameter = (
                    12.0
                    if image_raw_resolution[0] * image_raw_resolution[1] > 500000
                    else 6.0
                )

                if show_gt_keypoints:
                    videos_to_make[idx_kp_belief_overlay_raw][
                        "frame"
                    ] = dream.image_proc.overlay_points_on_image(
                        videos_to_make[idx_kp_belief_overlay_raw]["frame"],
                        [gt_kp_proj_raw],
                        annotation_color_dot="green",
                        annotation_color_text="white",
                        point_thickness=2,
                        point_diameter=pt_diameter + 2,
                    )

                videos_to_make[idx_kp_belief_overlay_raw][
                    "frame"
                ] = dream.image_proc.overlay_points_on_image(
                    videos_to_make[idx_kp_belief_overlay_raw]["frame"],
                    [detected_kp_proj_raw],
                    annotation_color_dot="red",
                    annotation_color_text="white",
                    point_diameter=pt_diameter,
                )

        frame_output_filename = str(idx_this_frame).zfill(6) + ".png"
        for video in videos_to_make:
            video["frame"].save(
                os.path.join(video["frames_dir"], frame_output_filename)
            )

        idx_this_frame += 1

    # Call to ffmpeg
    for video in videos_to_make:
        video_from_frames(video["frames_dir"], video["output_path"], args.framerate)
        shutil.rmtree(video["frames_dir"])


if __name__ == "__main__":
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
        "-d", "--dataset-path", required=True, help="Path to dataset to visualize."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Path to output directory to save visualization.",
    )
    parser.add_argument(
        "-f",
        "--force-overwrite",
        action="store_true",
        default=False,
        help="Forces overwriting of analysis results in the provided directory.",
    )
    parser.add_argument(
        "-k",
        "--keypoint-ids",
        nargs="+",
        type=int,
        default=None,
        help="List of keypoint indices to use in the visualization. No input (None) means all keypoints will be used.",
    )
    parser.add_argument(
        "-not-gt",
        "--no-ground-truth",
        action="store_true",
        default=False,
        help="Do not overlay ground truth keypoints when the dataset has ground truth.",
    )
    parser.add_argument(
        "-v",
        "--visualization-types",
        nargs="+",
        choices=[
            KP_OVERLAY_RAW,
            KP_OVERLAY_NET_INPUT,
            KP_BELIEF_OVERLAY_RAW,
            BELIEF_OVERLAY_RAW,
        ],
        default=[
            KP_OVERLAY_RAW,
            KP_OVERLAY_NET_INPUT,
            KP_BELIEF_OVERLAY_RAW,
            BELIEF_OVERLAY_RAW,
        ],
        help="Specify the visulizations to generate: 'kp_raw' overlays keypoints onto the original input image. 'kp_net_input' overlays keypoints onto the input image to the network. Multiple analyses can be specified.",
    )
    parser.add_argument(
        "-fps",
        "--framerate",
        type=float,
        default=30.0,
        help="Framerate (frames/sec) for the video that is created from the dataset.",
    )
    parser.add_argument(
        "-s",
        "--start-frame",
        default=None,
        help='Start frame of the dataset to visualize. Use the name, not the file extension (e.g., "-s 001022", not "-s 001022.png").',
    )
    parser.add_argument(
        "-e",
        "--end-frame",
        default=None,
        help='End frame of the dataset to visualize. Use the name, not the file extension (e.g., "-e 001022", not "-e 001022.png").',
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
    visualize_network_inference(args)
