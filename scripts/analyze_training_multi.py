# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import dream.analysis as dream_analysis

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def analyze_training(args):

    # Input argument handling
    assert os.path.exists(
        args.input_dir
    ), 'Expected input directory "{}" to exist, but it does not.'.format(args.input_dir)

    save_results = True if args.output_dir else False

    if save_results:
        dream.utilities.makedirs(args.output_dir, exist_ok=args.force_overwrite)

    dir_list = [
        this_dir
        for this_dir in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, this_dir))
    ]
    dir_list.sort(key=lambda x: os.path.getmtime(os.path.join(args.input_dir, x)))

    train_epochs = None
    all_losses_list = []
    all_validation_losses_list = []
    random_seeds = []
    for d in dir_list:
        results_path = os.path.join(args.input_dir, d, "training_log.pkl")
        with open(results_path, "rb") as f:
            training_log = pickle.load(f)

        # Process log
        epochs = training_log["epochs"]
        losses = training_log["losses"]
        this_random_seed = training_log["random_seed"]

        if train_epochs:
            assert train_epochs == epochs
        else:
            train_epochs = epochs

        all_losses_list.append(losses)

        if "validation_losses" in training_log:
            all_validation_losses_list.append(training_log["validation_losses"])

        random_seeds.append(this_random_seed)

        print("{}: Random seed: {}".format(d, this_random_seed))

    # Process
    all_losses = np.array(all_losses_list)
    all_losses_std = np.std(all_losses, axis=0)
    all_losses_mean = np.mean(all_losses, axis=0)
    all_losses_med = np.median(all_losses, axis=0)
    all_losses_max = np.max(all_losses, axis=0)
    all_losses_min = np.min(all_losses, axis=0)

    all_validation_losses = np.array(all_validation_losses_list)

    # Determine the worst and best results
    n_traces = len(all_losses_list)
    n_epochs = len(train_epochs)
    all_losses_lasthalf = all_losses[:, (n_epochs // 2) :]
    all_losses_lasthalf_sum = np.sum(all_losses_lasthalf, axis=1)
    x_worst = np.argmax(all_losses_lasthalf_sum)
    x_best = np.argmin(all_losses_lasthalf_sum)
    x_sort = np.argsort(all_losses_lasthalf_sum)
    x_median = x_sort[n_traces // 2]

    print("Training Loss Performance")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Best instance for training loss: {}".format(dir_list[x_best]))
    print("Median instance for training loss: {}".format(dir_list[x_median]))
    print("Worst instance for training loss: {}".format(dir_list[x_worst]))
    print("")

    # Plot
    fig, ax = plt.subplots()
    ax.plot(train_epochs, np.transpose(all_losses), ".-")
    ax.plot(
        train_epochs,
        all_losses[x_worst, :],
        "-",
        linewidth=8,
        alpha=0.667,
        label="Worst training result",
    )
    ax.plot(
        train_epochs,
        all_losses[x_best, :],
        "-",
        linewidth=8,
        alpha=0.667,
        label="Best training result",
    )
    ax.plot(
        train_epochs,
        all_losses[x_median, :],
        "-",
        linewidth=8,
        alpha=0.667,
        label="Median training result",
    )
    ax.grid()
    plt.xlabel("Training epoch")
    plt.ylabel("Training loss")
    plt.xlim((train_epochs[0], train_epochs[-1]))
    plt.title("All training results ({} instances)".format(n_traces))
    ax.legend(loc="best")
    if save_results:
        training_results_path = os.path.join(
            args.output_dir, "training_results_instances.png"
        )
        plt.savefig(training_results_path)

    fig, ax = plt.subplots()
    ax.fill_between(
        train_epochs,
        all_losses_mean - all_losses_std,
        all_losses_mean + all_losses_std,
        alpha=0.333,
        label="Aggregate mean +- 1 std dev",
    )
    ax.plot(train_epochs, all_losses_mean, ".-", label="Aggregate mean")
    ax.plot(train_epochs, all_losses_med, ".-", label="Aggregate median")
    ax.plot(train_epochs, all_losses_min, ".-", label="Aggregate min")
    ax.plot(train_epochs, all_losses_max, ".-", label="Aggregate max")
    ax.grid()
    plt.xlabel("Training epoch")
    plt.ylabel("Training loss")
    plt.xlim((train_epochs[0], train_epochs[-1]))
    plt.title("Aggregate (epoch-wise) training results ({} instances)".format(n_traces))
    ax.legend(loc="best")
    if save_results:
        training_results_agg_path = os.path.join(
            args.output_dir, "training_results_aggregate.png"
        )
        plt.savefig(training_results_agg_path)

    if not save_results:
        plt.show()

    # Now show validation losses
    if len(all_validation_losses) > 0:

        # Determine the best overall validation loss
        min_validation_losses_traces = np.min(all_validation_losses, axis=1)
        min_valid_loss_all_traces = np.min(min_validation_losses_traces)
        x_best_valid_loss = np.argmin(min_validation_losses_traces)
        x_epoch = np.argmin(all_validation_losses[x_best_valid_loss])

        print("Validation Loss Performance:")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(
            "Best instance for validation loss: {} ({} after epoch {})".format(
                dir_list[x_best_valid_loss],
                min_valid_loss_all_traces,
                train_epochs[x_epoch],
            )
        )

        for n in range(n_traces):

            save_plot_path = (
                os.path.join(args.output_dir, "train_valid_loss_{}".format(dir_list[n]))
                if save_results
                else None
            )
            dream_analysis.plot_train_valid_loss(
                train_epochs,
                all_losses[n, :],
                all_validation_losses[n, :],
                dataset_name=dir_list[n],
                save_plot_path=save_plot_path,
            )

        if not save_results:
            plt.show()


if __name__ == "__main__":

    # Parse input arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        metavar="input_dir",
        required=True,
        help="Path to directory of training results to analyze.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        metavar="output_dir",
        default=None,
        help="Path to output directory to save analysis results.",
    )
    parser.add_argument(
        "-f",
        "--force-overwrite",
        action="store_true",
        default=False,
        help="Forces overwriting of analysis results in the provided directory.",
    )
    args = parser.parse_args()

    # Analyze training results
    analyze_training(args)
