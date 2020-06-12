# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-whitegrid")


# Example of running the script
# python oks_plots.py --data all_dataset_keypoints.csv all_dataset_keypoints.csv --labels 1 2

# pythonw oks_plots.py --data deep-arm-cal-paper/data/dope/3cam_real_keypoints.csv deep-arm-cal-paper/data/dream_hg/3cam_real_keypoints.csv deep-arm-cal-paper/data/dream_hg_deconv/3cam_real_keypoints.csv deep-arm-cal-paper/data/resimple/3cam_real_keypoints.csv --labels DOPE DREAM AE resnet
parser = argparse.ArgumentParser(description="OKS for DREAM")
parser.add_argument(
    "--data", nargs="+", default="[all_dataset_keypoints.csv]", help="list of csv files"
)

parser.add_argument(
    "--labels",
    nargs="+",
    default=None,
    help="names for each dataset to be added as label",
)

parser.add_argument("--styles", nargs="+", default=None, help="")
parser.add_argument("--threshold", default=0.1)

parser.add_argument("--colours", nargs="+", default=None, help="")

parser.add_argument("--pixel", default=20)

parser.add_argument("--output", default="output.pdf")

parser.add_argument("--show", default=False, action="store_true")

parser.add_argument("--divide", default=False, action="store_true")


parser.add_argument("--title", default=None)

args = parser.parse_args()
print(args)


fig = plt.figure()
ax = plt.axes()

for i_csv, csv_file in enumerate(args.data):
    print(csv_file)

    if csv_file == "666":
        plt.plot([], [], " ", label=args.labels[i_csv].replace("_", " "))
        continue

    name_csv = csv_file.replace(".csv", "")

    df = pd.read_csv(csv_file)

    if args.divide is True:
        add = np.array(df["add"].tolist()) / 100
    else:
        add = np.array(df["add"].tolist())

    pnp_sol_found_magic_number = -9.99 if args.divide else -999.0

    n_inframe_gt_projs = np.array(df["n_inframe_gt_projs"].tolist())
    n_pnp_possible_frames = len(np.where(n_inframe_gt_projs >= 4)[0])
    add_pnp_found = add[np.where(add > pnp_sol_found_magic_number)]
    n_pnp_found = len(add_pnp_found)

    delta_threshold = 0.00001
    add_threshold_values = np.arange(0.0, args.threshold, delta_threshold)

    counts = []
    for value in add_threshold_values:
        under_threshold = (
            len(np.where(add_pnp_found <= value)[0]) / n_pnp_possible_frames
        )
        counts.append(under_threshold)

    auc = np.trapz(counts, dx=delta_threshold) / args.threshold

    # TODO: consolidate above calculations with pnp_metrics
    from dream import analysis as dream_analysis

    pnp_metrics = dream_analysis.pnp_metrics(df["add"], df["n_inframe_gt_projs"])
    assert pnp_metrics["add_auc"] == auc
    assert pnp_metrics["add_mean"] == np.mean(
        add[np.where(add > pnp_sol_found_magic_number)]
    )
    assert pnp_metrics["add_median"] == np.median(
        add[np.where(add > pnp_sol_found_magic_number)]
    )
    assert pnp_metrics["add_std"] == np.std(
        add[np.where(add > pnp_sol_found_magic_number)]
    )
    assert pnp_metrics["num_pnp_found"] == n_pnp_found
    assert pnp_metrics["num_pnp_possible"] == n_pnp_possible_frames

    # divide might screw this up .... to check!
    print("auc", auc)
    print("found", n_pnp_found / n_pnp_possible_frames)
    print("mean", np.mean(add[np.where(add > pnp_sol_found_magic_number)]))
    print("median", np.median(add[np.where(add > pnp_sol_found_magic_number)]))
    print("std", np.std(add[np.where(add > pnp_sol_found_magic_number)]))

    try:
        label = args.labels[i_csv]
    except:
        label = name_csv
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    try:

        colour = cycle[int(args.colours[i_csv])]
    except:
        colour = ""

    try:
        style = args.styles[i_csv]
        if style == "0":
            style = "-"
        elif style == "1":
            style = "--"
        elif style == "2":
            style = ":"

        else:
            style = "-"
    except:
        style = "-"

    label = f"{label} ({auc:.3f})"
    ax.plot(add_threshold_values, counts, style, color=colour, label=label)

plt.xlabel("ADD threshold distance (mm)")
plt.ylabel("Accuracy")
plt.title(args.title)
ax.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.8)


legend = ax.get_legend()
for i, t in enumerate(legend.get_texts()):
    if args.data[i] == "666":
        t.set_ha("left")  # ha is alias for horizontalalignment
        t.set_position((-30, 0))

ax.set_ylim(0, 1)
ax.set_xlim(0, float(args.threshold))
ax.set_xticklabels([0, 20, 40, 60, 80, 100])

plt.savefig(args.output)
if args.show:
    plt.show()
