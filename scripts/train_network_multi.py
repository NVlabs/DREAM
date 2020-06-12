# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import argparse
import os
import subprocess

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Totally developmental script


def train_network_multi(args):

    # Input argument handling
    assert (
        isinstance(args.num_instances, int) and args.num_instances > 0
    ), "Expected num_instances to be an integer greater than 0."

    # Generate output directories
    dream.utilities.makedirs(args.output_dir)
    output_dirs = [
        os.path.join(args.output_dir, "train_{}".format(n))
        for n in range(args.num_instances)
    ]

    for output_dir in output_dirs:

        # TBD: do this without subprocess
        train_command_output = (
            "python scripts/train_network.py "
            + args.train_command
            + ' -o "{}"'.format(output_dir)
        )
        subprocess.run(train_command_output, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-n",
        "--num-instances",
        metavar="num_instances",
        type=int,
        required=True,
        help="Number of training instances to run.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        metavar="output_dir",
        required=True,
        help="Output directory for training instances.",
    )
    parser.add_argument(
        "-c",
        "--train-command",
        metavar="train_command",
        required=True,
        help="Command line options for training.",
    )
    args = parser.parse_args()

    # Run train network, multiple instances version
    train_network_multi(args)
