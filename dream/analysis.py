# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import csv
import math
import os
from PIL import Image as PILImage

import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import YAML
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

import dream

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def plot_train_valid_loss(
    epochs, training_loss, validation_loss, dataset_name=None, save_plot_path=None
):

    # Input argument handling
    assert len(epochs) == len(
        training_loss
    ), "Expected the number of elements for epochs ({}) and training_loss ({}) to be the same.".format(
        len(epochs), len(training_loss)
    )
    assert len(epochs) == len(
        validation_loss
    ), "Expected the number of elements for epochs ({}) and validation_loss ({}) to be the same.".format(
        len(epochs), len(validation_loss)
    )

    plot_title = "Training vs. validation loss"

    # Create plot
    fig, ax = plt.subplots()

    # Determine whether we've been given estimates for each epoch, or batch-wise totals
    if isinstance(training_loss[0], float):
        training_loss_mean = training_loss
        validation_loss_mean = validation_loss

        ax.plot(epochs, training_loss_mean, ".-", label="Training")
        ax.plot(epochs, validation_loss_mean, ".-", label="Validation")
    else:
        plot_title += " (batch-wise mean +- 1 stdev)"

        training_loss_mean = [np.mean(x) for x in training_loss]
        validation_loss_mean = [np.mean(x) for x in validation_loss]

        training_loss_std = [np.std(x) for x in training_loss]
        validation_loss_std = [np.std(x) for x in validation_loss]

        ax.errorbar(
            epochs,
            training_loss_mean,
            yerr=training_loss_std,
            marker=".",
            linestyle="-",
            label="Training",
        )
        ax.errorbar(
            epochs,
            validation_loss_mean,
            yerr=validation_loss_std,
            marker=".",
            linestyle="-",
            label="Validation",
        )

    ax.grid()
    plt.xlabel("Training epoch")
    plt.ylabel("Loss")
    plt.xlim((epochs[0], epochs[-1]))

    if dataset_name:
        plot_title += ": {}".format(dataset_name)

    plt.title(plot_title)
    ax.legend(loc="best")

    if save_plot_path:
        plt.savefig(save_plot_path)

    return fig, ax


def analyze_ndds_dataset(
    network_params_path,
    network_config_path,
    dataset_dir,
    output_dir,
    visualize_belief_maps=True,
    pnp_analysis=True,
    force_overwrite=False,
    image_preprocessing_override=None,
    batch_size=16,
    num_workers=8,
    gpu_ids=None,
):

    # Input argument handling
    assert os.path.exists(
        network_params_path
    ), 'Expected network_params_path "{}" to exist, but it does not.'.format(
        network_params_path
    )
    assert os.path.exists(
        network_config_path
    ), 'Expected network_config_path "{}" to exist, but it does not.'.format(
        network_config_path
    )
    assert os.path.exists(
        dataset_dir
    ), 'Expected dataset_dir "{}" to exist, but it does not.'.format(dataset_dir)
    assert dream.utilities.is_ndds_dataset(
        dataset_dir
    ), 'Expected dataset_dir "{}" to be an NDDS Dataset, but it is not.'.format(
        dataset_dir
    )
    assert (
        isinstance(batch_size, int) and batch_size > 0
    ), 'If specified, "batch_size" must be a positive integer.'
    assert (
        isinstance(num_workers, int) and num_workers >= 0
    ), 'If specified, "num_workers" must be an integer greater than or equal to zero.'

    dream.utilities.makedirs(output_dir, exist_ok=force_overwrite)

    # Create parser
    data_parser = YAML(typ="safe")

    with open(network_config_path, "r") as f:
        network_config = data_parser.load(f)

    # Overwrite GPU
    # If nothing is specified, None is the default, which uses all GPUs
    # TBD - think about a better way of doing this
    network_config["training"]["platform"]["gpu_ids"] = gpu_ids

    # Load network
    dream_network = dream.create_network_from_config_data(network_config)
    dream_network.model.load_state_dict(torch.load(network_params_path))
    dream_network.enable_evaluation()

    # Use image preprocessing specified by config by default, unless user specifies otherwise
    image_preprocessing = (
        image_preprocessing_override
        if image_preprocessing_override
        else dream_network.image_preprocessing()
    )

    # Create NDDS dataset to process
    manip_dataset_debug_mode = dream.datasets.ManipulatorNDDSDatasetDebugLevels["LIGHT"]
    (
        found_ndds_dataset_data,
        found_ndds_dataset_config,
    ) = dream.utilities.find_ndds_data_in_dir(dataset_dir)
    image_raw_resolution = dream.utilities.load_image_resolution(
        found_ndds_dataset_config["camera"]
    )
    (
        network_input_res_inf,
        network_output_res_inf,
    ) = dream_network.net_resolutions_from_image_raw_resolution(
        image_raw_resolution, image_preprocessing_override=image_preprocessing
    )

    found_manip_dataset = dream.datasets.ManipulatorNDDSDataset(
        (found_ndds_dataset_data, found_ndds_dataset_config),
        dream_network.manipulator_name,
        dream_network.keypoint_names,
        network_input_res_inf,
        network_output_res_inf,
        dream_network.image_normalization,
        image_preprocessing,
        augment_data=False,
        debug_mode=manip_dataset_debug_mode,
    )

    data_loader = TorchDataLoader(
        found_manip_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    all_kp_projs_gt_raw = []
    all_kp_projs_detected_raw = []
    all_gt_kp_positions = []

    sample_results = []
    sample_idx = 0

    with torch.no_grad():

        # Step through data
        print("Conducting inference...")
        for batch_idx, sample in enumerate(tqdm(data_loader)):
            sample_info = sample["config"]
            this_batch_size = len(sample_info["name"])

            # Conduct inference - disregard belief maps in this part of the analysis
            network_image_input = sample["image_rgb_input"].cuda()
            _, detected_keypoints_netout_batch = dream_network.inference(
                network_image_input
            )

            for b in range(this_batch_size):
                # We are using the data loader, which does the preprocessing
                # So we need to take the output from inference and then convert keypoints back to the raw frame
                # We do this one element of the batch at a time because we don't have on-batch keypoint convertion right now
                # Alternatively, we could have used keypoints_from_image, which internalizes all preprocessing, but that would have been slower
                this_detected_kps_netout = np.array(
                    detected_keypoints_netout_batch[b], dtype=float
                )
                this_detected_kps_netin = dream.image_proc.convert_keypoints_to_netin_from_netout(
                    this_detected_kps_netout,
                    network_output_res_inf,
                    network_input_res_inf,
                )
                this_detected_kps_raw = dream.image_proc.convert_keypoints_to_raw_from_netin(
                    this_detected_kps_netin,
                    network_input_res_inf,
                    image_raw_resolution,
                    image_preprocessing,
                )

                all_kp_projs_detected_raw.append(this_detected_kps_raw.tolist())

                gt_kps_raw = np.array(
                    sample["keypoint_projections_raw"][b], dtype=float
                )
                all_kp_projs_gt_raw.append(gt_kps_raw.tolist())

                # Metric is just L2 error at the raw image frame for in-frame keypoints if network detects one (original, before network input)
                kp_l2_err = []
                for this_kp_detect_raw, this_kp_gt_raw in zip(
                    this_detected_kps_raw, gt_kps_raw
                ):
                    if (
                        (
                            this_kp_detect_raw[0] < -999.0
                            and this_kp_detect_raw[1] < -999.0
                        )
                        or this_kp_gt_raw[0] < 0.0
                        or this_kp_gt_raw[0] > image_raw_resolution[0]
                        or this_kp_gt_raw[1] < 0.0
                        or this_kp_gt_raw[1] > image_raw_resolution[1]
                    ):
                        continue

                    kp_l2_err.append(
                        np.linalg.norm(this_kp_detect_raw - this_kp_gt_raw)
                    )

                if kp_l2_err:
                    this_metric = np.mean(kp_l2_err)
                else:
                    this_metric = 999.999

                if pnp_analysis:
                    gt_kp_pos = sample["keypoint_positions"][b]
                    all_gt_kp_positions.append(gt_kp_pos.tolist())

                this_sample_info = {
                    "name": sample_info["name"][b],
                    "image_paths": {"rgb": sample_info["image_paths"]["rgb"][b]},
                }
                sample_results.append((sample_idx, this_sample_info, this_metric))

                sample_idx += 1

    all_kp_projs_detected_raw = np.array(all_kp_projs_detected_raw)
    all_kp_projs_gt_raw = np.array(all_kp_projs_gt_raw)

    # Write keypoint file
    n_samples = len(sample_results)
    kp_metrics = keypoint_metrics(
        all_kp_projs_detected_raw.reshape(n_samples * dream_network.n_keypoints, 2),
        all_kp_projs_gt_raw.reshape(n_samples * dream_network.n_keypoints, 2),
        image_raw_resolution,
    )
    keypoint_path = os.path.join(output_dir, "keypoints.csv")
    sample_names = [x[1]["name"] for x in sample_results]

    write_keypoint_csv(
        keypoint_path, sample_names, all_kp_projs_detected_raw, all_kp_projs_gt_raw
    )

    # PNP analysis
    pnp_attempts_successful = []
    poses_xyzxyzw = []
    all_n_inframe_projs_gt = []
    pnp_add = []

    if pnp_analysis:
        all_gt_kp_positions = np.array(all_gt_kp_positions)
        camera_K = dream.utilities.load_camera_intrinsics(
            found_ndds_dataset_config["camera"]
        )
        for kp_projs_est, kp_projs_gt, kp_pos_gt in zip(
            all_kp_projs_detected_raw, all_kp_projs_gt_raw, all_gt_kp_positions
        ):

            n_inframe_projs_gt = 0
            for kp_proj_gt in kp_projs_gt:
                if (
                    0.0 < kp_proj_gt[0]
                    and kp_proj_gt[0] < image_raw_resolution[0]
                    and 0.0 < kp_proj_gt[1]
                    and kp_proj_gt[1] < image_raw_resolution[1]
                ):
                    n_inframe_projs_gt += 1

            idx_good_detections = np.where(kp_projs_est > -999.0)
            idx_good_detections_rows = np.unique(idx_good_detections[0])
            kp_projs_est_pnp = kp_projs_est[idx_good_detections_rows, :]
            kp_pos_gt_pnp = kp_pos_gt[idx_good_detections_rows, :]

            pnp_retval, translation, quaternion = dream.geometric_vision.solve_pnp(
                kp_pos_gt_pnp, kp_projs_est_pnp, camera_K
            )
            # pnp_retval, translation, quaternion, inliers = dream.geometric_vision.solve_pnp_ransac(kp_pos_gt_pnp, kp_projs_est_pnp, camera_K)

            pnp_attempts_successful.append(pnp_retval)

            all_n_inframe_projs_gt.append(n_inframe_projs_gt)

            if pnp_retval:
                poses_xyzxyzw.append(translation.tolist() + quaternion.tolist())
                add = dream.geometric_vision.add_from_pose(
                    translation, quaternion, kp_pos_gt_pnp, camera_K
                )
            else:
                poses_xyzxyzw.append([-999.99] * 7)
                add = -999.99

            pnp_add.append(add)

        pnp_path = os.path.join(output_dir, "pnp_results.csv")
        write_pnp_csv(
            pnp_path,
            sample_names,
            pnp_attempts_successful,
            poses_xyzxyzw,
            pnp_add,
            all_n_inframe_projs_gt,
        )
        pnp_results = pnp_metrics(pnp_add, all_n_inframe_projs_gt)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def print_to_screen_and_file(file, text):
        print(text)
        file.write(text + "\n")

    results_log_path = os.path.join(output_dir, "analysis_results.txt")
    with open(results_log_path, "w") as f:

        # Write results header
        print_to_screen_and_file(
            f, "Analysis results for dataset: {}".format(dataset_dir)
        )
        print_to_screen_and_file(
            f, "Number of frames in this dataset: {}".format(n_samples)
        )
        print_to_screen_and_file(
            f, "Using network config defined from: {}".format(network_config_path)
        )
        print_to_screen_and_file(f, "")

        # Write keypoint metric summary to file
        if kp_metrics["num_gt_outframe"] > 0:
            print_to_screen_and_file(
                f,
                "Percentage out-of-frame gt keypoints not found (correct): {:.3f}% ({}/{})".format(
                    float(kp_metrics["num_missing_gt_outframe"])
                    / float(kp_metrics["num_gt_outframe"])
                    * 100.0,
                    kp_metrics["num_missing_gt_outframe"],
                    kp_metrics["num_gt_outframe"],
                ),
            )
            print_to_screen_and_file(
                f,
                "Percentage out-of-frame gt keypoints found (incorrect): {:.3f}% ({}/{})".format(
                    float(kp_metrics["num_found_gt_outframe"])
                    / float(kp_metrics["num_gt_outframe"])
                    * 100.0,
                    kp_metrics["num_found_gt_outframe"],
                    kp_metrics["num_gt_outframe"],
                ),
            )
        else:
            print_to_screen_and_file(f, "No out-of-frame gt keypoints.")

        if kp_metrics["num_gt_inframe"] > 0:
            print_to_screen_and_file(
                f,
                "Percentage in-frame gt keypoints not found (incorrect): {:.3f}% ({}/{})".format(
                    float(kp_metrics["num_missing_gt_inframe"])
                    / float(kp_metrics["num_gt_inframe"])
                    * 100.0,
                    kp_metrics["num_missing_gt_inframe"],
                    kp_metrics["num_gt_inframe"],
                ),
            )
            print_to_screen_and_file(
                f,
                "Percentage in-frame gt keypoints found (correct): {:.3f}% ({}/{})".format(
                    float(kp_metrics["num_found_gt_inframe"])
                    / float(kp_metrics["num_gt_inframe"])
                    * 100.0,
                    kp_metrics["num_found_gt_inframe"],
                    kp_metrics["num_gt_inframe"],
                ),
            )
            if kp_metrics["num_found_gt_inframe"] > 0:
                print_to_screen_and_file(
                    f,
                    "L2 error (px) for in-frame keypoints (n = {}):".format(
                        kp_metrics["num_found_gt_inframe"]
                    ),
                )
                print_to_screen_and_file(
                    f, "   AUC: {:.5f}".format(kp_metrics["l2_error_auc"])
                )
                print_to_screen_and_file(
                    f,
                    "      AUC threshold: {:.5f}".format(
                        kp_metrics["l2_error_auc_thresh_px"]
                    ),
                )
                print_to_screen_and_file(
                    f, "   Mean: {:.5f}".format(kp_metrics["l2_error_mean_px"])
                )
                print_to_screen_and_file(
                    f, "   Median: {:.5f}".format(kp_metrics["l2_error_median_px"])
                )
                print_to_screen_and_file(
                    f, "   Std Dev: {:.5f}".format(kp_metrics["l2_error_std_px"])
                )
            else:
                print_to_screen_and_file(f, "No in-frame gt keypoints were detected.")
        else:
            print_to_screen_and_file(f, "No in-frame gt keypoints.")

        print_to_screen_and_file(f, "")

        if pnp_analysis:
            n_pnp_possible = pnp_results["num_pnp_possible"]
            if n_pnp_possible > 0:
                n_pnp_successful = pnp_results["num_pnp_found"]
                n_pnp_fails = pnp_results["num_pnp_not_found"]
                print_to_screen_and_file(
                    f,
                    "Percentage of frames where PNP failed when viable (incorrect): {:.3f}% ({}/{})".format(
                        float(n_pnp_fails) / float(n_pnp_possible) * 100.0,
                        n_pnp_fails,
                        n_pnp_possible,
                    ),
                )
                print_to_screen_and_file(
                    f,
                    "Percentage of frames where PNP was successful when viable (correct): {:.3f}% ({}/{})".format(
                        float(n_pnp_successful) / float(n_pnp_possible) * 100.0,
                        n_pnp_successful,
                        n_pnp_possible,
                    ),
                )
                print_to_screen_and_file(
                    f,
                    "ADD (m) for frames where PNP was successful when viable (n = {}):".format(
                        n_pnp_successful
                    ),
                )
                print_to_screen_and_file(
                    f, "   AUC: {:.5f}".format(pnp_results["add_auc"])
                )
                print_to_screen_and_file(
                    f,
                    "      AUC threshold: {:.5f}".format(pnp_results["add_auc_thresh"]),
                )
                print_to_screen_and_file(
                    f, "   Mean: {:.5f}".format(pnp_results["add_mean"])
                )
                print_to_screen_and_file(
                    f, "   Median: {:.5f}".format(pnp_results["add_median"])
                )
                print_to_screen_and_file(
                    f, "   Std Dev: {:.5f}".format(pnp_results["add_std"])
                )
            else:
                print_to_screen_and_file(f, "No frames where PNP is possible.")

            print_to_screen_and_file(f, "")

        if visualize_belief_maps:

            # Sort the sample results to determine the ranges to provide further analysis
            sample_results_sorted = sorted(sample_results, key=lambda x: x[2])

            n_outliers = min([5, n_samples // 10]) if n_samples >= 10 else 1

            best_samples_range = range(n_outliers)
            worst_samples_range = range(n_samples - n_outliers, n_samples)

            n_med_start = int(np.floor(n_samples / 2.0 - n_outliers / 2.0))
            median_samples_range = range(n_med_start, n_med_start + n_outliers)

            # Best Samples ---------------------------------------------------------------------------------------------
            best_sample_names = [
                sample_results_sorted[i][1]["name"] for i in best_samples_range
            ]
            best_sample_image_paths = [
                sample_results_sorted[i][1]["image_paths"]["rgb"]
                for i in best_samples_range
            ]
            best_sample_ranks = [i + 1 for i in best_samples_range]
            best_sample_metrics = [
                sample_results_sorted[i][2] for i in best_samples_range
            ]

            best_samples_ndds_data = [
                found_ndds_dataset_data[sample_results_sorted[i][0]]
                for i in best_samples_range
            ]
            best_samples_manip_dataset = dream.datasets.ManipulatorNDDSDataset(
                (best_samples_ndds_data, found_ndds_dataset_config),
                dream_network.manipulator_name,
                dream_network.keypoint_names,
                network_input_res_inf,
                network_output_res_inf,
                dream_network.image_normalization,
                image_preprocessing,
                augment_data=False,
                debug_mode=manip_dataset_debug_mode,
            )

            best_samples_data_loader = TorchDataLoader(
                best_samples_manip_dataset,
                batch_size=n_outliers,
                num_workers=num_workers,
                shuffle=False,
            )

            with torch.no_grad():
                for batch_idx, sample in enumerate(best_samples_data_loader):
                    network_image_input = sample["image_rgb_input"].cuda()

                    network_output = dream_network.inference(network_image_input)
                    best_samples_belief_maps = network_output[0]
                    best_detected_keypoints_netout_batch = network_output[1]
                    best_gt_keypoints_netout_batch = sample[
                        "keypoint_projections_output"
                    ]
                    best_net_input_images_viz_batch = sample["image_rgb_input_viz"]

            best_detected_keypoints_netout = np.array(
                best_detected_keypoints_netout_batch, dtype=float
            )
            best_gt_keypoints_netout = np.array(
                best_gt_keypoints_netout_batch, dtype=float
            )

            print_to_screen_and_file(f, "{} best samples:".format(n_outliers))
            sample_range_analysis(
                best_sample_image_paths,
                best_detected_keypoints_netout,
                best_gt_keypoints_netout,
                best_samples_belief_maps,
                best_sample_names,
                best_sample_ranks,
                "best_samples",
                output_dir,
                dream_network.keypoint_names,
                best_net_input_images_viz_batch,
            )
            for this_sample_name, this_sample_rank, this_sample_metric in zip(
                best_sample_names, best_sample_ranks, best_sample_metrics
            ):
                print_to_screen_and_file(
                    f,
                    "Sample: {}, Rank: {}, Metric: {}".format(
                        this_sample_name, this_sample_rank, this_sample_metric
                    ),
                )
            print_to_screen_and_file(f, "")

            # Median Samples -------------------------------------------------------------------------------------------
            median_sample_names = [
                sample_results_sorted[i][1]["name"] for i in median_samples_range
            ]
            median_sample_image_paths = [
                sample_results_sorted[i][1]["image_paths"]["rgb"]
                for i in median_samples_range
            ]
            median_sample_ranks = [i + 1 for i in median_samples_range]
            median_sample_metrics = [
                sample_results_sorted[i][2] for i in median_samples_range
            ]

            median_samples_ndds_data = [
                found_ndds_dataset_data[sample_results_sorted[i][0]]
                for i in median_samples_range
            ]
            median_samples_manip_dataset = dream.datasets.ManipulatorNDDSDataset(
                (median_samples_ndds_data, found_ndds_dataset_config),
                dream_network.manipulator_name,
                dream_network.keypoint_names,
                network_input_res_inf,
                network_output_res_inf,
                dream_network.image_normalization,
                image_preprocessing,
                augment_data=False,
                debug_mode=manip_dataset_debug_mode,
            )

            median_samples_data_loader = TorchDataLoader(
                median_samples_manip_dataset,
                batch_size=n_outliers,
                num_workers=num_workers,
                shuffle=False,
            )

            with torch.no_grad():
                for batch_idx, sample in enumerate(median_samples_data_loader):
                    network_image_input = sample["image_rgb_input"].cuda()

                    network_output = dream_network.inference(network_image_input)
                    median_samples_belief_maps = network_output[0]
                    median_detected_keypoints_netout_batch = network_output[1]
                    median_gt_keypoints_netout_batch = sample[
                        "keypoint_projections_output"
                    ]
                    median_net_input_images_viz_batch = sample["image_rgb_input_viz"]

            median_detected_keypoints_netout = np.array(
                median_detected_keypoints_netout_batch, dtype=float
            )
            median_gt_keypoints_netout = np.array(
                median_gt_keypoints_netout_batch, dtype=float
            )

            print_to_screen_and_file(f, "{} median samples:".format(n_outliers))
            sample_range_analysis(
                median_sample_image_paths,
                median_detected_keypoints_netout,
                median_gt_keypoints_netout,
                median_samples_belief_maps,
                median_sample_names,
                median_sample_ranks,
                "median_samples",
                output_dir,
                dream_network.keypoint_names,
                median_net_input_images_viz_batch,
            )
            for this_sample_name, this_sample_rank, this_sample_metric in zip(
                median_sample_names, median_sample_ranks, median_sample_metrics
            ):
                print_to_screen_and_file(
                    f,
                    "Sample: {}, Rank: {}, Metric: {}".format(
                        this_sample_name, this_sample_rank, this_sample_metric
                    ),
                )
            print_to_screen_and_file(f, "")

            # Worst Samples --------------------------------------------------------------------------------------------
            worst_sample_names = [
                sample_results_sorted[i][1]["name"] for i in worst_samples_range
            ]
            worst_sample_image_paths = [
                sample_results_sorted[i][1]["image_paths"]["rgb"]
                for i in worst_samples_range
            ]
            worst_sample_ranks = [i + 1 for i in worst_samples_range]
            worst_sample_metrics = [
                sample_results_sorted[i][2] for i in worst_samples_range
            ]

            worst_samples_ndds_data = [
                found_ndds_dataset_data[sample_results_sorted[i][0]]
                for i in worst_samples_range
            ]
            worst_samples_manip_dataset = dream.datasets.ManipulatorNDDSDataset(
                (worst_samples_ndds_data, found_ndds_dataset_config),
                dream_network.manipulator_name,
                dream_network.keypoint_names,
                network_input_res_inf,
                network_output_res_inf,
                dream_network.image_normalization,
                image_preprocessing,
                augment_data=False,
                debug_mode=manip_dataset_debug_mode,
            )

            worst_samples_data_loader = TorchDataLoader(
                worst_samples_manip_dataset,
                batch_size=n_outliers,
                num_workers=num_workers,
                shuffle=False,
            )

            with torch.no_grad():
                for batch_idx, sample in enumerate(worst_samples_data_loader):
                    network_image_input = sample["image_rgb_input"].cuda()

                    network_output = dream_network.inference(network_image_input)
                    worst_samples_belief_maps = network_output[0]
                    worst_detected_keypoints_netout_batch = network_output[1]
                    worst_gt_keypoints_netout_batch = sample[
                        "keypoint_projections_output"
                    ]
                    worst_net_input_images_viz_batch = sample["image_rgb_input_viz"]

            worst_detected_keypoints_netout = np.array(
                worst_detected_keypoints_netout_batch, dtype=float
            )
            worst_gt_keypoints_netout = np.array(
                worst_gt_keypoints_netout_batch, dtype=float
            )

            print_to_screen_and_file(f, "{} worst samples:".format(n_outliers))
            sample_range_analysis(
                worst_sample_image_paths,
                worst_detected_keypoints_netout,
                worst_gt_keypoints_netout,
                worst_samples_belief_maps,
                worst_sample_names,
                worst_sample_ranks,
                "worst_samples",
                output_dir,
                dream_network.keypoint_names,
                worst_net_input_images_viz_batch,
            )
            for this_sample_name, this_sample_rank, this_sample_metric in zip(
                worst_sample_names, worst_sample_ranks, worst_sample_metrics
            ):
                print_to_screen_and_file(
                    f,
                    "Sample: {}, Rank: {}, Metric: {}".format(
                        this_sample_name, this_sample_rank, this_sample_metric
                    ),
                )

    if pnp_analysis:
        return (
            sample_names,
            all_kp_projs_detected_raw,
            all_kp_projs_gt_raw,
            pnp_attempts_successful,
            poses_xyzxyzw,
            pnp_add,
            all_n_inframe_projs_gt,
        )
    else:
        return sample_names, all_kp_projs_detected_raw, all_kp_projs_gt_raw


def write_keypoint_csv(keypoint_path, sample_names, keypoints_detected, keypoints_gt):

    assert (
        keypoints_detected.shape == keypoints_gt.shape
    ), 'Expected "keypoints_detected" and "keypoints_gt" to have the same shape.'

    n_samples = len(sample_names)

    assert (
        n_samples == keypoints_detected.shape[0]
    ), "Expected number of sample names to equal the number of keypoint entries."

    n_keypoints = keypoints_detected.shape[1]
    n_keypoint_dims = keypoints_detected.shape[2]

    assert n_keypoint_dims == 2, "Expected the number of keypoint dimensions to be 2."

    n_keypoint_elements = n_keypoints * n_keypoint_dims

    with open(keypoint_path, "w") as csvfile:
        csv_writer = csv.writer(csvfile)

        kp_detected_colnames = []
        kp_gt_colnames = []
        for kp_idx in range(n_keypoints):
            kp_detected_colnames.append("kp{}x".format(kp_idx))
            kp_detected_colnames.append("kp{}y".format(kp_idx))
            kp_gt_colnames.append("kp{}x_gt".format(kp_idx))
            kp_gt_colnames.append("kp{}y_gt".format(kp_idx))
        header = ["name"] + kp_detected_colnames + kp_gt_colnames
        csv_writer.writerow(header)

        for name, kp_detected, kp_gt in zip(
            sample_names, keypoints_detected, keypoints_gt
        ):
            entry = (
                [name]
                + kp_detected.reshape(n_keypoint_elements).tolist()
                + kp_gt.reshape(n_keypoint_elements).tolist()
            )
            csv_writer.writerow(entry)


# write_pnp_csv: poses is expected to be array of [x y z x y z w]
def write_pnp_csv(
    pnp_path,
    sample_names,
    pnp_attempts_successful,
    poses,
    pnp_add,
    num_inframe_projs_gt,
):

    n_samples = len(sample_names)

    assert n_samples == len(pnp_attempts_successful)
    assert n_samples == len(poses)
    assert n_samples == len(num_inframe_projs_gt)
    assert n_samples == len(pnp_add)

    with open(pnp_path, "w") as csvfile:
        csv_writer = csv.writer(csvfile)

        header = [
            "name",
            "pnp_success",
            "pose_x",
            "pose_y",
            "pose_z",
            "pose_qx",
            "pose_qy",
            "pose_qz",
            "pose_qw",
            "add",
            "n_inframe_gt_projs",
        ]
        csv_writer.writerow(header)

        for name, pnp_successful, pose, this_pnp_add, this_num_inframe_projs_gt in zip(
            sample_names, pnp_attempts_successful, poses, pnp_add, num_inframe_projs_gt
        ):
            entry = (
                [name]
                + [pnp_successful]
                + pose
                + [this_pnp_add]
                + [this_num_inframe_projs_gt]
            )
            csv_writer.writerow(entry)


def keypoint_metrics(
    keypoints_detected, keypoints_gt, image_resolution, auc_pixel_threshold=20.0
):

    # TBD: input argument handling
    num_gt_outframe = 0
    num_gt_inframe = 0
    num_missing_gt_outframe = 0
    num_found_gt_outframe = 0
    num_found_gt_inframe = 0
    num_missing_gt_inframe = 0

    kp_errors = []
    for kp_proj_detect, kp_proj_gt in zip(keypoints_detected, keypoints_gt):

        if (
            kp_proj_gt[0] < 0.0
            or kp_proj_gt[0] > image_resolution[0]
            or kp_proj_gt[1] < 0.0
            or kp_proj_gt[1] > image_resolution[1]
        ):
            # GT keypoint is out of frame
            num_gt_outframe += 1

            if kp_proj_detect[0] < -999.0 and kp_proj_detect[1] < -999.0:
                # Did not find a keypoint (correct)
                num_missing_gt_outframe += 1
            else:
                # Found a keypoint (wrong)
                num_found_gt_outframe += 1

        else:
            # GT keypoint is in frame
            num_gt_inframe += 1

            if kp_proj_detect[0] < -999.0 and kp_proj_detect[1] < -999.0:
                # Did not find a keypoint (wrong)
                num_missing_gt_inframe += 1
            else:
                # Found a keypoint (correct)
                num_found_gt_inframe += 1

                kp_errors.append((kp_proj_detect - kp_proj_gt).tolist())

    kp_errors = np.array(kp_errors)

    if len(kp_errors) > 0:
        kp_l2_errors = np.linalg.norm(kp_errors, axis=1)
        kp_l2_error_mean = np.mean(kp_l2_errors)
        kp_l2_error_median = np.median(kp_l2_errors)
        kp_l2_error_std = np.std(kp_l2_errors)

        # compute the auc
        delta_pixel = 0.01
        pck_values = np.arange(0, auc_pixel_threshold, delta_pixel)
        y_values = []

        for value in pck_values:
            valids = len(np.where(kp_l2_errors < value)[0])
            y_values.append(valids)

        kp_auc = (
            np.trapz(y_values, dx=delta_pixel)
            / float(auc_pixel_threshold)
            / float(num_gt_inframe)
        )

    else:
        kp_l2_error_mean = None
        kp_l2_error_median = None
        kp_l2_error_std = None
        kp_auc = None

    metrics = {
        "num_gt_outframe": num_gt_outframe,
        "num_missing_gt_outframe": num_missing_gt_outframe,
        "num_found_gt_outframe": num_found_gt_outframe,
        "num_gt_inframe": num_gt_inframe,
        "num_found_gt_inframe": num_found_gt_inframe,
        "num_missing_gt_inframe": num_missing_gt_inframe,
        "l2_error_mean_px": kp_l2_error_mean,
        "l2_error_median_px": kp_l2_error_median,
        "l2_error_std_px": kp_l2_error_std,
        "l2_error_auc": kp_auc,
        "l2_error_auc_thresh_px": auc_pixel_threshold,
    }
    return metrics


def pnp_metrics(
    pnp_add,
    num_inframe_projs_gt,
    num_min_inframe_projs_gt_for_pnp=4,
    add_auc_threshold=0.1,
    pnp_magic_number=-999.0,
):
    pnp_add = np.array(pnp_add)
    num_inframe_projs_gt = np.array(num_inframe_projs_gt)

    idx_pnp_found = np.where(pnp_add > pnp_magic_number)[0]
    add_pnp_found = pnp_add[idx_pnp_found]
    num_pnp_found = len(idx_pnp_found)

    mean_add = np.mean(add_pnp_found)
    median_add = np.median(add_pnp_found)
    std_add = np.std(add_pnp_found)

    num_pnp_possible = len(
        np.where(num_inframe_projs_gt >= num_min_inframe_projs_gt_for_pnp)[0]
    )
    num_pnp_not_found = num_pnp_possible - num_pnp_found

    delta_threshold = 0.00001
    add_threshold_values = np.arange(0.0, add_auc_threshold, delta_threshold)

    counts = []
    for value in add_threshold_values:
        under_threshold = len(np.where(add_pnp_found <= value)[0]) / float(
            num_pnp_possible
        )
        counts.append(under_threshold)

    auc = np.trapz(counts, dx=delta_threshold) / float(add_auc_threshold)

    metrics = {
        "num_pnp_found": num_pnp_found,
        "num_pnp_not_found": num_pnp_not_found,
        "num_pnp_possible": num_pnp_possible,
        "num_min_inframe_projs_gt_for_pnp": num_min_inframe_projs_gt_for_pnp,
        "pnp_magic_number": pnp_magic_number,
        "add_mean": mean_add,
        "add_median": median_add,
        "add_std": std_add,
        "add_auc": auc,
        "add_auc_thresh": add_auc_threshold,
    }
    return metrics


def sample_range_analysis(
    raw_images_or_image_paths,
    sample_kp_proj_detected_netout,
    sample_kp_proj_gt_netout,
    sample_belief_maps,
    sample_names,
    sample_ranks,
    image_prefix,
    output_dir,
    keypoint_names,
    images_net_input_tensor_batch,
):
    n_keypoints = len(keypoint_names)
    n_cols = int(math.ceil(n_keypoints / 2.0))

    n_sample_range = len(raw_images_or_image_paths)

    images_net_input = dream.image_proc.images_from_tensor(
        images_net_input_tensor_batch
    )
    images_net_input_overlay = []

    # Assume the belief maps are in the net output frame
    net_output_res_inf = (
        sample_belief_maps[0].shape[2],
        sample_belief_maps[0].shape[1],
    )

    for (
        keypoint_projs_detected,
        keypoint_projs_gt,
        belief_maps,
        sample_name,
        sample_rank,
        image_rgb_net_input,
    ) in zip(
        sample_kp_proj_detected_netout,
        sample_kp_proj_gt_netout,
        sample_belief_maps,
        sample_names,
        sample_ranks,
        images_net_input,
    ):

        # Create belief map mosaics, with and without keypoint overlay
        # This is in the "net output" frame belief maps
        belief_maps_mosaic_path = os.path.join(
            output_dir,
            image_prefix
            + "_belief_maps_rank_{}_id_{}.png".format(sample_rank, sample_name),
        )
        belief_maps_kp_mosaic_path = os.path.join(
            output_dir,
            image_prefix
            + "_belief_maps_kp_rank_{}_id_{}.png".format(sample_rank, sample_name),
        )

        belief_map_images = dream.image_proc.images_from_belief_maps(
            belief_maps, normalization_method=6
        )
        belief_maps_mosaic = dream.image_proc.mosaic_images(
            belief_map_images, rows=2, cols=n_cols, inner_padding_px=10
        )
        belief_maps_mosaic.save(belief_maps_mosaic_path)

        # This is in the "net output" frame belief maps with keypoint overlays
        belief_map_images_kp = []
        for n_kp in range(n_keypoints):
            belief_map_image_kp = dream.image_proc.overlay_points_on_image(
                belief_map_images[n_kp],
                [keypoint_projs_gt[n_kp, :], keypoint_projs_detected[n_kp, :]],
                annotation_color_dot=["green", "red"],
                point_diameter=4,
            )
            belief_map_images_kp.append(belief_map_image_kp)
        belief_maps_kp_mosaic = dream.image_proc.mosaic_images(
            belief_map_images_kp, rows=2, cols=n_cols, inner_padding_px=10
        )
        belief_maps_kp_mosaic.save(belief_maps_kp_mosaic_path)

        # Create overlay of keypoints (detected and gt) on network input image
        net_input_res_inf = image_rgb_net_input.size
        scale_factor_netin_from_netout = (
            float(net_input_res_inf[0]) / float(net_output_res_inf[0]),
            float(net_input_res_inf[1]) / float(net_output_res_inf[1]),
        )

        kp_projs_detected_net_input = []
        kp_projs_gt_net_input = []
        for n_kp in range(n_keypoints):
            kp_projs_detected_net_input.append(
                [
                    keypoint_projs_detected[n_kp][0]
                    * scale_factor_netin_from_netout[0],
                    keypoint_projs_detected[n_kp][1]
                    * scale_factor_netin_from_netout[1],
                ]
            )
            kp_projs_gt_net_input.append(
                [
                    keypoint_projs_gt[n_kp][0] * scale_factor_netin_from_netout[0],
                    keypoint_projs_gt[n_kp][1] * scale_factor_netin_from_netout[1],
                ]
            )

        image_rgb_net_input_overlay = dream.image_proc.overlay_points_on_image(
            image_rgb_net_input,
            kp_projs_gt_net_input,
            keypoint_names,
            annotation_color_dot="green",
            annotation_color_text="green",
        )
        image_rgb_net_input_overlay = dream.image_proc.overlay_points_on_image(
            image_rgb_net_input_overlay,
            kp_projs_detected_net_input,
            keypoint_names,
            annotation_color_dot="red",
            annotation_color_text="red",
        )
        images_net_input_overlay.append(image_rgb_net_input_overlay)

        # Generate blended (net input + belief map) images
        blend_input_belief_map_images = []
        blend_input_belief_map_kp_images = []

        for n in range(len(belief_map_images)):
            # Upscale belief map to net input resolution
            belief_map_image_upscaled = belief_map_images[n].resize(
                net_input_res_inf, resample=PILImage.BILINEAR
            )

            # Increase image brightness to account for the belief map overlay
            # TBD - maybe use a mask instead
            blend_input_belief_map_image = PILImage.blend(
                belief_map_image_upscaled, image_rgb_net_input, alpha=0.5
            )
            blend_input_belief_map_images.append(blend_input_belief_map_image)

            # Overlay on the blended one directly so the annotation isn't blurred
            blend_input_belief_map_kp_image = dream.image_proc.overlay_points_on_image(
                blend_input_belief_map_image,
                [kp_projs_gt_net_input[n], kp_projs_detected_net_input[n]],
                [keypoint_names[n]] * 2,
                annotation_color_dot=["green", "red"],
                annotation_color_text=["green", "red"],
                point_diameter=4,
            )
            blend_input_belief_map_kp_images.append(blend_input_belief_map_kp_image)

        mosaic_blend_input_belief_map_images = dream.image_proc.mosaic_images(
            blend_input_belief_map_images, rows=2, cols=n_cols, inner_padding_px=10
        )
        mosaic_blend_input_belief_map_images_path = os.path.join(
            output_dir,
            image_prefix + "_blend_rank_{}_id_{}.png".format(sample_rank, sample_name),
        )
        mosaic_blend_input_belief_map_images.save(
            mosaic_blend_input_belief_map_images_path
        )

        mosaic_blend_input_belief_map_kp_images = dream.image_proc.mosaic_images(
            blend_input_belief_map_kp_images, rows=2, cols=n_cols, inner_padding_px=10
        )
        mosaic_blend_input_belief_map_kp_images_path = os.path.join(
            output_dir,
            image_prefix
            + "_blend_kp_rank_{}_id_{}.png".format(sample_rank, sample_name),
        )
        mosaic_blend_input_belief_map_kp_images.save(
            mosaic_blend_input_belief_map_kp_images_path
        )

    # This just a mosaic of all the inputs in raw form
    mosaic = dream.image_proc.mosaic_images(
        raw_images_or_image_paths, rows=1, cols=n_sample_range, inner_padding_px=10
    )
    mosaic_path = os.path.join(output_dir, image_prefix + ".png")
    mosaic.save(mosaic_path)

    # This is a mosaic of the net input images, with and without KP overlays
    mosaic_net_input = dream.image_proc.mosaic_images(
        images_net_input, rows=1, cols=n_sample_range, inner_padding_px=10
    )
    mosaic_net_input_path = os.path.join(output_dir, image_prefix + "_net_input.png")
    mosaic_net_input.save(mosaic_net_input_path)

    mosaic_net_input_overlay = dream.image_proc.mosaic_images(
        images_net_input_overlay, rows=1, cols=n_sample_range, inner_padding_px=10
    )
    mosaic_net_input_overlay_path = os.path.join(
        output_dir, image_prefix + "_net_input_kp.png"
    )
    mosaic_net_input_overlay.save(mosaic_net_input_overlay_path)
