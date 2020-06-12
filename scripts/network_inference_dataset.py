# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import argparse
import os

import dream.analysis as dream_analysis

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def network_inference_dataset(args):

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
        args.dataset_dir
    ), 'Expected dataset_dir "{}" to exist, but it does not.'.format(args.dataset_dir)

    assert (
        isinstance(args.batch_size, int) and args.batch_size > 0
    ), 'If specified, "batch_size" must be a positive integer.'

    visualize_belief_maps = not args.not_visualize_belief_maps

    dream_analysis.analyze_ndds_dataset(
        args.input_params_path,
        input_config_path,
        args.dataset_dir,
        args.output_dir,
        visualize_belief_maps=visualize_belief_maps,
        pnp_analysis=True,
        force_overwrite=args.force_overwrite,
        image_preprocessing_override=args.image_preproc_override,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gpu_ids=args.gpu_ids,
    )


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
        "-d",
        "--dataset-dir",
        required=True,
        help="Path to NDDS dataset on which to conduct network inference.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Path to output directory to save analysis results.",
    )
    parser.add_argument(
        "-not-v",
        "--not-visualize-belief-maps",
        action="store_true",
        default=False,
        help="Disable belief map visualization. Without this flag, belief map visualization is enabled by default.",
    )
    parser.add_argument(
        "-f",
        "--force-overwrite",
        action="store_true",
        default=False,
        help="Forces overwriting of analysis results in the provided directory.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=16,
        help="The batch size used to process the data. Does not affect results, only how quickly the results are obtained.",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=8,
        help='The number of subprocesses ("workers") used for loading the data. 0 means that no subprocesses are used. '
        + "Does not affect results, only how quickly the results are obtained.",
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
    network_inference_dataset(args)
