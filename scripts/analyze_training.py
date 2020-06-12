# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import argparse
import os
import pickle

from ruamel.yaml import YAML

import dream.analysis as dream_analysis

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LOSS_TEXT = "loss"
VIZ_TEXT = "viz"


def analyze_training(args):

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

    do_training_plots = True if LOSS_TEXT in args.analyses else False
    do_visualizations = True if VIZ_TEXT in args.analyses else False

    dream.utilities.makedirs(args.output_dir, exist_ok=args.force_overwrite)

    if do_training_plots:

        # Use params filepath to infer the training log pickle
        training_log_path = os.path.join(
            os.path.dirname(args.input_params_path), "training_log.pkl"
        )
        with open(training_log_path, "rb") as f:
            training_log = pickle.load(f)

        epochs = training_log["epochs"]
        batch_training_losses = training_log["batch_training_losses"]
        batch_validation_losses = training_log["batch_validation_losses"]

        save_plot_path = os.path.join(args.output_dir, "train_valid_loss.png")
        dream_analysis.plot_train_valid_loss(
            epochs,
            batch_training_losses,
            batch_validation_losses,
            save_plot_path=save_plot_path,
        )

    if do_visualizations:

        # Create parser
        data_parser = YAML(typ="safe")

        with open(input_config_path, "r") as f:
            network_config = data_parser.load(f)

        dataset_dir = os.path.expanduser(network_config["data_path"])

        dream_analysis.analyze_ndds_dataset(
            args.input_params_path,
            input_config_path,
            dataset_dir,
            args.output_dir,
            batch_size=args.batch_size,
            force_overwrite=args.force_overwrite,
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
        help="Path to network configuration file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Path to output directory to save training analysis results.",
    )
    parser.add_argument(
        "-f",
        "--force-overwrite",
        action="store_true",
        default=False,
        help="Forces overwriting of analysis results in the provided directory.",
    )
    parser.add_argument(
        "-a",
        "--analyses",
        nargs="+",
        choices=[LOSS_TEXT, VIZ_TEXT],
        default=[LOSS_TEXT, VIZ_TEXT],
        help="Specify the analyses to run: 'loss' is loss plots; 'viz' is visualizations. Multiple analyses can be specified.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="The batch size used to process the data. Does not affect results, only how quickly the results are obtained.",
    )
    parser.add_argument(
        "-g",
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="The GPU IDs on which to conduct network inference. Nothing specified means all GPUs will be utilized. Does not affect results, only how quickly the results are obtained.",
    )
    args = parser.parse_args()
    analyze_training(args)
