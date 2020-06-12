# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import argparse
from collections import OrderedDict as odict
import pickle
import os
import random
import socket
import time

import numpy as np
from ruamel.yaml import YAML
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

import dream

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


def train_network(args):

    # Input argument handling
    assert (
        args.epochs > 0
    ), "The number of training epochs must be greater than 0, but it is {}.".format(
        args.epochs
    )
    assert (
        args.batch_size > 0
    ), "The training batch size must be greater than 0, but it is {}.".format(
        args.batch_size
    )
    assert (
        args.num_workers >= 0
    ), "The number of subprocesses used for training data loading must be greater than or equal to 0, but it is {}.".format(
        args.num_workers
    )

    # Parse training fraction
    assert (
        0.0 < args.training_data_fraction and args.training_data_fraction < 1.0
    ), "Expected training_data_fraction to be within 0. and 1., but it is {}.".format(
        args.training_data_fraction
    )
    validation_data_fraction = 1.0 - args.training_data_fraction

    if args.output_dir:
        save_results = True
        if not args.resume_training:
            dream.utilities.makedirs(args.output_dir, exist_ok=args.force_overwrite)
    else:
        assert (
            not args.resume_training
        ), "Cannot resume training; output directory not provided."
        save_results = False

    training_start_time = time.time()

    if args.resume_training:

        # Find the latest network we have
        dirlist = os.listdir(args.output_dir)
        epoch_weight_paths_unsorted = [
            x for x in dirlist if x.startswith("epoch") and x.endswith(".pth")
        ]
        epoch_numbers_unsorted = []
        for net_path in epoch_weight_paths_unsorted:
            epoch_number = int(net_path.split("_")[1].split(".")[0])
            epoch_numbers_unsorted.append(epoch_number)

        temp = sorted(
            zip(epoch_weight_paths_unsorted, epoch_numbers_unsorted),
            key=lambda pair: pair[1],
            reverse=True,
        )
        epoch_weight_paths = [x[0] for x in temp]
        epoch_numbers = [x[1] for x in temp]

        # Most recent network
        most_recent_epoch_weight_path = epoch_weight_paths[0]
        start_epoch = epoch_numbers[0]

        assert (
            start_epoch < args.epochs
        ), "Network is already trained for the number of requested epochs."

        # Find the best network to determine its validation loss
        best_valid_network_config_path = os.path.join(
            args.output_dir, "best_network.yaml"
        )
        assert os.path.exists(
            best_valid_network_config_path
        ), "Could not determine the best validation loss."

        valid_parser = YAML(typ="safe")
        with open(best_valid_network_config_path, "r") as f:
            best_valid_network_config = valid_parser.load(f)
        best_valid_loss = best_valid_network_config["training"]["results"][
            "validation_loss"
        ]["mean"]

        # Load in the old training log
        if os.path.exists(os.path.join(args.output_dir, "training_log.pkl")):
            train_log_path = os.path.join(args.output_dir, "training_log.pkl")
            with open(train_log_path, "rb") as f:
                train_log = pickle.load(f)
            # Move this to make this consistent as if we're in the middle of training
            os.rename(
                train_log_path,
                os.path.join(
                    args.output_dir, "training_log_e{}.pkl".format(start_epoch)
                ),
            )

        elif os.path.exists(
            os.path.join(args.output_dir, "training_log_e{}.pkl".format(start_epoch))
        ):
            train_log_path = os.path.join(
                args.output_dir, "training_log_e{}.pkl".format(start_epoch)
            )
            with open(train_log_path, "rb") as f:
                train_log = pickle.load(f)
        else:
            assert False, "Could not determine training log file to resume."

        # Get the random seed that was used here - we need to to ensure test/valid splits are right
        random_seed = train_log["random_seed"]

        # Set the random seed here because it's different
        if not isinstance(train_log["start_time"], list):
            # Convert to a list
            train_log["start_time"] = [train_log["start_time"]]

        train_log["start_time"].append(training_start_time)

        # Also log the fact that we resumed
        if "epochs_resumed" in train_log:
            train_log["epochs_resumed"].append(start_epoch + 1)
        else:
            train_log["epochs_resumed"] = [start_epoch + 1]

    else:
        # Determine the random seed
        random_seed = (
            args.random_seed if args.random_seed else random.randint(0, 999999)
        )

        train_log = {
            "epochs": [],
            "losses": [],
            "validation_losses": [],
            "batch_training_losses": [],
            "batch_validation_losses": [],
            "batch_training_sample_names": [],
            "batch_validation_sample_names": [],
            "start_time": training_start_time,
            "timestamps": [],
            "random_seed": random_seed,
        }
        best_valid_loss = float("Inf")

    dream.utilities.set_random_seed(random_seed)

    enable_augment_data = not args.not_augment_data

    gpu_ids = args.gpu_ids if args.gpu_ids else []

    try:
        user = os.getlogin()
    except:
        user = "not found"

    # Parse input data
    input_data_path = args.input_data_path
    # Attempt path contraction to make path portable between different platforms
    input_data_abs_path = os.path.abspath(input_data_path)
    input_data_abs_path_split = input_data_abs_path.split("/")
    if (
        len(input_data_abs_path_split) >= 3
        and input_data_abs_path_split[0] == ""
        and input_data_abs_path_split[1] == "home"
        and input_data_abs_path_split[2] == user
    ):
        # Change the path to use the tilde shortcut
        input_data_path = os.path.join("~", *input_data_abs_path_split[3:])

    # Find data in provided directory
    found_data = dream.utilities.find_ndds_data_in_dir(input_data_path)
    found_data_config = found_data[1]
    image_raw_resolution = dream.utilities.load_image_resolution(
        found_data_config["camera"]
    )

    # Parse manipulation configuration file
    yaml_parser = YAML(typ="safe")
    assert os.path.exists(
        args.manipulator_config_path
    ), 'Expected manipulator_config_path "{}" to exist, but it does not.'.format(
        args.manipulator_config_path
    )
    with open(args.manipulator_config_path, "r") as f:
        manipulator_config_file = yaml_parser.load(f)
    assert (
        "manipulator" in manipulator_config_file
    ), 'Expected key "manipulator" to exist in the manipulator config file, but it does not.'
    manipulator_config = manipulator_config_file["manipulator"]

    # Parse architecture
    assert os.path.exists(
        args.architecture_config
    ), 'Expected architecture_config file "{}" to exist, but it does not.'.format(
        args.architecture_config
    )
    with open(args.architecture_config, "r") as f:
        architecture_config_file = yaml_parser.load(f)
    assert (
        "architecture" in architecture_config_file
    ), 'Expected key "architecture" to exist in the architecture config file, but it does not.'
    architecture_config = architecture_config_file["architecture"]

    assert (
        "training" in architecture_config_file
    ), 'Expected key "training" to exist in the architecture config file, but it does not.'
    assert (
        "config" in architecture_config_file["training"]
    ), 'Expected key "config" to exist in training dictionary in the architecture config file, but it does not.'
    training_config = architecture_config_file["training"]["config"]
    assert (
        "image_preprocessing" in training_config
    ), 'Expected key "image_preprocessing" to exist in the training config in the architecture config file, but it does not.'
    training_image_preprocessing = training_config["image_preprocessing"]
    assert (
        "net_input_resolution" in training_config
    ), 'Expected key "net_input_resolution" to exist in the training config in the architecture config file, but it does not.'
    training_net_input_resolution = training_config["net_input_resolution"]
    # TODO: possibly read in other arguments here, such as optimizer, instead of using command line defaults

    if "image_preprocessing" in architecture_config:
        # This could happen if we're trying to resume training.
        assert (
            architecture_config["image_preprocessing"] == training_image_preprocessing
        ), 'If defined, "image_preprocessing" in the architecture and training record must be consistent for this script to work properly.'
    else:
        architecture_config["image_preprocessing"] = training_image_preprocessing

    if enable_augment_data:
        # TODO: specify the types of image augmentation
        data_augment_config = odict([("image_rgb", True)])
    else:
        data_augment_config = False

    network_config = odict(
        [
            ("data_path", input_data_path),
            ("manipulator", manipulator_config),
            ("architecture", architecture_config),
            (
                "training",
                odict(
                    [
                        (
                            "config",
                            odict(
                                [
                                    ("epochs", args.epochs),
                                    (
                                        "training_data_fraction",
                                        args.training_data_fraction,
                                    ),
                                    (
                                        "validation_data_fraction",
                                        validation_data_fraction,
                                    ),
                                    ("batch_size", args.batch_size),
                                    ("data_augmentation", data_augment_config),
                                    ("worker_size", args.num_workers),
                                    (
                                        "optimizer",
                                        odict(
                                            [
                                                ("type", args.optimizer),
                                                ("learning_rate", args.learning_rate),
                                            ]
                                        ),
                                    ),
                                    (
                                        "image_preprocessing",
                                        training_image_preprocessing,
                                    ),
                                    (
                                        "image_raw_resolution",
                                        list(image_raw_resolution),
                                    ),
                                    (
                                        "net_input_resolution",
                                        training_net_input_resolution,
                                    ),
                                ]
                            ),
                        ),  # net_output_resolution is set below
                        (
                            "platform",
                            odict(
                                [
                                    ("user", user),
                                    ("hostname", socket.gethostname()),
                                    ("gpu_ids", gpu_ids),
                                ]
                            ),
                        ),
                        ("results", odict([("epochs_trained", 0)])),
                    ]
                ),
            ),
        ]
    )

    # Now check against existing network configuration if we are resuming training
    if args.resume_training:

        # Load corresponding config file to ensure we're consistent
        most_recent_config_path = most_recent_epoch_weight_path.replace("pth", "yaml")
        config_parser = YAML(typ="safe")

        with open(os.path.join(args.output_dir, most_recent_config_path), "r") as f:
            most_recent_network_config_file = config_parser.load(f)

        # Do a bunch of network consistency checks
        assert (
            most_recent_network_config_file["data_path"] == network_config["data_path"]
        )
        assert (
            most_recent_network_config_file["manipulator"]
            == network_config["manipulator"]
        )
        assert (
            most_recent_network_config_file["architecture"]
            == network_config["architecture"]
        )
        assert (
            most_recent_network_config_file["training"]["config"][
                "training_data_fraction"
            ]
            == network_config["training"]["config"]["training_data_fraction"]
        )
        assert (
            most_recent_network_config_file["training"]["config"][
                "validation_data_fraction"
            ]
            == network_config["training"]["config"]["validation_data_fraction"]
        )
        assert (
            most_recent_network_config_file["training"]["config"]["batch_size"]
            == network_config["training"]["config"]["batch_size"]
        )
        assert (
            most_recent_network_config_file["training"]["config"]["data_augmentation"]
            == network_config["training"]["config"]["data_augmentation"]
        )
        assert (
            most_recent_network_config_file["training"]["config"]["worker_size"]
            == network_config["training"]["config"]["worker_size"]
        )
        assert (
            most_recent_network_config_file["training"]["config"]["optimizer"]
            == network_config["training"]["config"]["optimizer"]
        )
        assert (
            most_recent_network_config_file["training"]["config"]["image_preprocessing"]
            == network_config["training"]["config"]["image_preprocessing"]
        )
        assert (
            most_recent_network_config_file["training"]["config"][
                "image_raw_resolution"
            ]
            == network_config["training"]["config"]["image_raw_resolution"]
        )
        assert (
            most_recent_network_config_file["training"]["config"][
                "net_input_resolution"
            ]
            == network_config["training"]["config"]["net_input_resolution"]
        )

        # Use this one instead!
        network_config = most_recent_network_config_file

        print("~~ RESUMING TRAINING FROM {} ~~".format(most_recent_epoch_weight_path))
        print("")

    else:
        start_epoch = 0

    # Print to screen
    print("Network configuration: {}".format(network_config))
    dream_network = dream.create_network_from_config_data(network_config)
    if args.resume_training:
        dream_network.model.load_state_dict(
            torch.load(os.path.join(args.output_dir, most_recent_epoch_weight_path))
        )
    dream_network.enable_training()

    # The following ensures the config is consistent with the dataloader
    (
        trained_net_input_res,
        trained_net_output_res,
    ) = dream_network.net_resolutions_from_image_raw_resolution(image_raw_resolution)
    assert dream_network.trained_net_input_resolution() == trained_net_input_res
    assert dream_network.trained_net_output_resolution() == trained_net_output_res
    dream_network.network_config["training"]["config"][
        "net_output_resolution"
    ] = trained_net_output_res

    # Create NDDS dataset and loader
    training_debug_mode = dream.datasets.ManipulatorNDDSDatasetDebugLevels["NONE"]
    network_requires_belief_maps = (
        dream_network.network_config["architecture"]["target"] == "belief_maps"
    )
    found_dataset = dream.datasets.ManipulatorNDDSDataset(
        found_data,
        manipulator_config["name"],
        dream_network.keypoint_names,
        trained_net_input_res,
        trained_net_output_res,
        dream_network.image_normalization,
        dream_network.image_preprocessing(),
        augment_data=enable_augment_data,
        include_ground_truth=True,
        include_belief_maps=network_requires_belief_maps,
        debug_mode=training_debug_mode,
    )

    # Split into train and validation subsets
    n_data = len(found_dataset)
    n_train_data = int(round(n_data * args.training_data_fraction))
    n_valid_data = n_data - n_train_data
    train_dataset, valid_dataset = torch.utils.data.random_split(
        found_dataset, [n_train_data, n_valid_data]
    )

    train_data_loader = TorchDataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    valid_data_loader = TorchDataLoader(
        valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Train the network
    print("")
    print("TRAINING NETWORK ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")

    last_epoch_timestamp = 0.0

    for e in tqdm(range(start_epoch, args.epochs)):
        this_epoch = e + 1
        print("Epoch {} ------------".format(this_epoch))

        # Training Phase ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if args.verbose:
            print("")
            print("~~ Training Phase ~~")

        dream_network.enable_training()

        training_batch_losses = []
        training_batch_sample_names = []

        for batch_idx, sample in enumerate(tqdm(train_data_loader)):

            this_batch_sample_names = sample["config"]["name"]
            this_batch_size = sample["image_rgb_input"].shape[0]

            if args.verbose:
                print("Processing batch index {} for training...".format(batch_idx))
                print(
                    "Sample names in this training batch: {}".format(
                        this_batch_sample_names
                    )
                )
                print("This training batch size: {}".format(this_batch_size))

            # New unified training
            network_input_heads = []
            network_input_heads.append(sample["image_rgb_input"].cuda())

            if dream_network.network_config["architecture"]["target"] == "belief_maps":
                training_labels = sample["belief_maps"].cuda()
            elif dream_network.network_config["architecture"]["target"] == "keypoints":
                training_labels = sample["keypoint_projections_output"].cuda()
            else:
                assert (
                    False
                ), "Could not determine how to provide training labels to network."

            loss = dream_network.train(network_input_heads, training_labels)

            training_loss_this_batch = loss.item()
            training_batch_losses.append(training_loss_this_batch)
            if args.verbose:
                print(
                    "Training loss for this batch: {}".format(training_loss_this_batch)
                )
                print("")
            training_batch_sample_names.append(this_batch_sample_names)

        mean_training_loss_per_batch = np.mean(training_batch_losses)
        std_training_loss_per_batch = np.std(training_batch_losses)

        # Evaluation Phase ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if args.verbose:
            print("")
            print("~~ Validation Phase ~~")

        dream_network.enable_evaluation()

        with torch.no_grad():

            valid_batch_losses = []
            valid_batch_sample_names = []

            for valid_batch_idx, valid_sample in enumerate(tqdm(valid_data_loader)):

                this_valid_batch_sample_names = valid_sample["config"]["name"]
                this_valid_batch_size = valid_sample["image_rgb_input"].shape[0]

                if args.verbose:
                    print(
                        "Processing batch index {} for validation...".format(
                            valid_batch_idx
                        )
                    )
                    print(
                        "Sample names in this validation batch: {}".format(
                            this_valid_batch_sample_names
                        )
                    )
                    print(
                        "This validation batch size: {}".format(this_valid_batch_size)
                    )

                # New unified validation
                valid_network_input_heads = []
                valid_network_input_heads.append(valid_sample["image_rgb_input"].cuda())

                if (
                    dream_network.network_config["architecture"]["target"]
                    == "belief_maps"
                ):
                    valid_labels = valid_sample["belief_maps"].cuda()
                elif (
                    dream_network.network_config["architecture"]["target"]
                    == "keypoints"
                ):
                    valid_labels = valid_sample["keypoint_projections_output"].cuda()
                else:
                    assert (
                        False
                    ), "Could not determine how to provide validation labels to network."

                valid_loss = dream_network.loss(valid_network_input_heads, valid_labels)

                valid_loss_this_batch = valid_loss.item()
                valid_batch_losses.append(valid_loss_this_batch)
                if args.verbose:
                    print(
                        "Validation loss for this batch: {}".format(
                            valid_loss_this_batch
                        )
                    )
                    print("")
                valid_batch_sample_names.append(this_valid_batch_sample_names)

            mean_valid_loss_per_batch = np.mean(valid_batch_losses)
            std_valid_loss_per_batch = np.std(valid_batch_losses)

        # Bookkeeping and print info
        dream_network.network_config["training"]["results"]["epochs_trained"] += 1
        dream_network.network_config["training"]["results"]["training_loss"] = odict(
            [
                ("mean", float(mean_training_loss_per_batch)),
                ("stdev", float(std_training_loss_per_batch)),
            ]
        )
        dream_network.network_config["training"]["results"]["validation_loss"] = odict(
            [
                ("mean", float(mean_valid_loss_per_batch)),
                ("stdev", float(std_valid_loss_per_batch)),
            ]
        )
        print(
            "Training Loss (batch-wise mean +- 1 stdev): {} +- {}".format(
                mean_training_loss_per_batch, std_training_loss_per_batch
            )
        )
        print(
            "Validation Loss (batch-wise mean +- 1 stdev): {} +- {}".format(
                mean_valid_loss_per_batch, std_valid_loss_per_batch
            )
        )

        # Save network if it's better than anything trained so far
        if mean_valid_loss_per_batch < best_valid_loss:

            print("Best network result so far.")
            best_valid_loss = mean_valid_loss_per_batch

            if save_results:
                dream_network.save_network(
                    args.output_dir, "best_network", overwrite=True
                )

        this_epoch_timestamp = time.time() - training_start_time
        print(
            "This epoch took {} seconds.".format(
                this_epoch_timestamp - last_epoch_timestamp
            )
        )
        last_epoch_timestamp = this_epoch_timestamp
        print("")

        # Append to history
        train_log["epochs"].append(this_epoch)
        train_log["losses"].append(mean_training_loss_per_batch)
        train_log["validation_losses"].append(mean_valid_loss_per_batch)
        train_log["batch_training_losses"].append(training_batch_losses)
        train_log["batch_validation_losses"].append(valid_batch_losses)
        train_log["batch_training_sample_names"].append(training_batch_sample_names)
        train_log["batch_validation_sample_names"].append(valid_batch_sample_names)
        train_log["timestamps"].append(this_epoch_timestamp)

        if save_results:
            # Write training log so far
            epoch_training_log_path = os.path.join(
                args.output_dir, "training_log_e{}.pkl".format(this_epoch)
            )
            with open(epoch_training_log_path, "wb") as f:
                pickle.dump(train_log, f)

            # Remove old training log
            last_epoch_training_log_path = os.path.join(
                args.output_dir, "training_log_e{}.pkl".format(e)
            )
            if os.path.exists(last_epoch_training_log_path):
                os.remove(last_epoch_training_log_path)

            # Save this epoch
            dream_network.save_network(
                args.output_dir, "epoch_{}".format(this_epoch), overwrite=True
            )

    # Save results
    if save_results:
        # Rename the final training log instead of re-writing it
        training_log_path = os.path.join(args.output_dir, "training_log.pkl")
        os.rename(epoch_training_log_path, training_log_path)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")
    print("Done.")
    print("")
    print("Total training time: {} seconds.".format(time.time() - training_start_time))
    print("")


if __name__ == "__main__":

    # Parse input arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-data-path", required=True, help="Path to training data."
    )
    parser.add_argument(
        "-t",
        "--training-data-fraction",
        type=float,
        default=0.8,
        help="Fraction of training data to use for training. 1 - this quantity will be used for validation during training.",
    )
    parser.add_argument(
        "-m",
        "--manipulator-config-path",
        type=str,
        required=True,
        help="Path to a configuration file that specifies the manipulator and keypoint configuration.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Path to output directory for training results. Nothing specified means training results will NOT be saved.",
    )
    parser.add_argument(
        "-f",
        "--force-overwrite",
        action="store_true",
        default=False,
        help="Forces overwriting of analysis results in the provided directory.",
    )
    parser.add_argument(
        "-ar",
        "--architecture-config",
        type=str,
        required=True,
        help="Path to a configuration file that describes the neural network architecture configuration.",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, required=True, help="Number of epochs to train."
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        required=True,
        help="The number of samples per batch used for training.",
    )
    parser.add_argument(
        "-z",
        "--optimizer",
        choices=dream.KNOWN_OPTIMIZERS,
        default="adam",
        help="The optimizer type to use.",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.0001,
        help="The learning rate used for the optimizer.",
    )
    parser.add_argument(
        "-not-a",
        "--not-augment-data",
        action="store_true",
        default=False,
        help="Disable data augmentation. Without this flag, data augmentation is enabled by default.",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=8,
        help='The number of subprocesses ("workers") used for loading the training data. 0 means that no subprocesses are used.',
    )
    parser.add_argument(
        "-g",
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="The GPU IDs on which to train the network. Nothing specified means all GPUs will be utilized.",
    )
    parser.add_argument(
        "-s", "--random-seed", type=int, help="Manually specify the random seed."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Outputs all diagnostic information to the screen.",
    )
    parser.add_argument(
        "-r",
        "--resume-training",
        action="store_true",
        default=False,
        help="Resumes training. The epoch argument provided now is the new training duration. All arguments must match the previously trained networks.",
    )

    args = parser.parse_args()

    # Train the network
    train_network(args)
