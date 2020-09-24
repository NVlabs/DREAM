# DREAM: Deep Robot-to-Camera Extrinsics for Articulated Manipulators

This is the official implementation of ["Camera-to-robot pose estimation from a single image"](https://arxiv.org/abs/1911.09231) (ICRA 2020).  The DREAM system uses a robot-specific deep neural network to detect keypoints (typically joint locations) in the RGB image of a robot manipulator. Using these keypoint locations along with the robot forward kinematics, the camera pose with respect to the
robot is estimated using a perspective-n-point (PnP) algorithm.  For more details, please see our [paper](https://arxiv.org/abs/1911.09231) and [video](https://youtu.be/O1qAFboFQ8A).

![DREAM in operation](dream-franka.png)


## **Installation**

We have tested on Ubuntu 16.04 and 18.04 with an NVIDIA GeForce RTX 2080 and Titan X, with both Python 2.7 and Python 3.6.  The code may work on other systems.

Install the DREAM package and its dependencies using `pip`:

```
pip install . -r requirements.txt
```

Download the pre-trained models and (optionally) data.  In the scripts below, be sure to comment out files you do not want, as they are very large.  Alternatively, you can download files [manually](https://drive.google.com/drive/folders/1Krp-fCT9ffEML3IpweSOgWiMHHBw6k2Z?usp=sharing)
```
cd trained_models; ./DOWNLOAD.sh; cd ..
cd data; ./DOWNLOAD.sh; cd ..
```

Unit tests are implemented in the `pytest` framework. Verify your installation by running them:  `pytest test/`


## **Offline inference**

There are three scripts for offline inference:

```
# Run the network on a single image to display detected 2D keypoints.
python scripts/network_inference.py -i <path/to/network.pth> -m <path/to/image.png> 

# Process a dataset to save both 2D keypoints and 3D poses.
python scripts/network_inference_dataset.py -i <path/to/network.pth> -d <path/to/dataset_dir/> -o <path/to/output_dir/> -b 16 -w 8

# Run the network on an image sequence 
# (either a dataset or a directory of images, e.g., from a video),
# and saves the resulting visualizations as videos.
python scripts/visualize_network_inference.py -i <path/to/network.pth> -d <path/to/dataset_dir/> -o <path/to/output_dir/> -s <start_frame_name> -e <end_frame_name>
```

Pass `-h` for help on command line arguments.  Datasets are assumed to be in [NDDS](https://github.com/NVIDIA/Dataset_Synthesizer) format.


#### **Example for single-image inference**

Single-image inference from one frame of the Panda-3Cam RealSense dataset using the DREAM-vgg-Q network:

```
python scripts/network_inference.py -i trained_models/panda_dream_vgg_q.pth -m data/real/panda-3cam_realsense/000000.rgb.jpg
```

You should see the detected keypoints printed to the screen as well as overlaid on the Panda robot. (See note below regarding the Panda keypoint locations.)


#### **Example for dataset inference**

Inference on the Panda-3Cam RealSense dataset using the DREAM-vgg-Q network:

```
python scripts/network_inference_dataset.py -i trained_models/panda_dream_vgg_q.pth -d data/real/panda-3cam_realsense/ -o <path/to/output_results> -b 16 -w 8
```

The analysis will print to both the screen and file. You should see that the percentage of correct keypoints (PCK) and the area under the curve (AUC) is about 0.720, and the average distance (ADD) AUC is about 0.792. Various visualizations will also be saved to disk.

#### **Example for generating inference visualizations**

Generating visualizations for one of the sequences in the Panda-3Cam RealSense dataset using the DREAM-vgg-Q network:

```
python scripts/visualize_network_inference.py -i trained_models/panda_dream_vgg_q.pth -d data/real/panda-3cam_realsense -o <path/to/output_results> -fps 120.0 -s 004151
```
This creates videos at 4x normal camera framerate.

## **Online inference using ROS**

A ROS node is provided for real-time camera pose estimation.
Some values, such as ROS topic names, may
need to be changed for your application.  Because of incompatabilities between
ROS (before Noetic) and Python 3, the DREAM ROS node is implemented using Python 2.7.  For ease
of use, we have provided a [Docker setup](docker/) containing all the necessary
compoinents to run DREAM with ROS Kinetic.

Example to run the DREAM ROS node (in verbose mode):

```
python scripts/launch_dream_ros.py -i trained_models/baxter_dream_vgg_q.pth -b torso -v
```


## **Training**

Below is an example for training a DREAM-vgg-Q model for the Franka Emika Panda robot:

```
python scripts/train_network.py -i data/synthetic/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_hourglass_example.yaml -e 25 -lr 0.00015 -b 128 -w 16 -o <path/to/output_dir/>
```

The models below are defined in the following architecture files:
- DREAM-vgg-Q: `arch_configs/dream_vgg_q.yaml`
- DREAM-vgg-F: `arch_configs/deam_vgg_f.yaml`
- DREAM-resnet-H: `arch_configs/dream_resnet_h.yaml`
- DREAM-resnet-F: `arch_configs/dream_resnet_f.yaml` (very large network and unwieldy to train)

## **Note on Panda keypoints**

By default, keypoints are defined at the joint locations as defined by the robot URDF file. In the case of the Panda robot, the URDF file defines the joints at non-intuitive locations. As a result, visualizations of keypoint detections may appear to be wrong when they are in fact correct (see [our video](https://youtu.be/O1qAFboFQ8A)).  We have since modified the URDF to place the keypoints at the actual joint locations (see Fig. 5a of our paper), but for simplicity we are not releasing the modified URDF at this time.


## **Note on reproducing results**

The experiments in the paper used the image preprocessing type `shrink-and-crop`, which preserves the same aspect ratio of the input image, but crops the width to send 400 x 400 resolution to the network (which is the resolution used during training). In order to allow for full-frame inference, the models we released have the default image preprocessing type `resize`, which prevents this cropping.  Careful analysis has shown almost no difference in quantitative results, but if you are looking to reproduce our ICRA results exactly, please change the `architecture/image_processing` value to `shrink-and-crop`.

The PCK and ADD plots in the paper are generated from `oks_plots.py` and `add_plots.py`. The AUC in these figures (and in Table 1) are in the `analysis_results.txt` file that is produced by `scripts/network_inference_dataset.py`.

# Further information

For an example of how DREAM can be used in practice for vision-based object manipulation, please refer to ["Indirect Object-to-Robot Pose Estimation from an External Monocular RGB Camera"](https://research.nvidia.com/publication/2020-07_Indirect-Object-Pose) by Jonathan Tremblay, Stephen Tyree, Terry Mosier, and Stan Birchfield.

For more information on how DREAM learns robot keypoint detection from sim-to-real transfer using only synthetic domain randomized images, please refer to our [sim2realAI blog post](https://sim2realai.github.io/dream-camera-calibration-sim2real/).


# License

DREAM is licensed under the [NVIDIA Source Code License - Non-commercial](LICENSE.md).


# Citation

Please cite our work if you use it for your research. Thank you!

```
@inproceedings{lee2020icra:dream,
  title={Camera-to-Robot Pose Estimation from a Single Image},
  author={Lee, Timothy E and Tremblay, Jonathan and To, Thang and Cheng, Jia and Mosier, Terry and Kroemer, Oliver and Fox, Dieter and Birchfield, Stan},
  booktitle={International Conference on Robotics and Automation (ICRA)},
  year=2020,
  url={https://arxiv.org/abs/1911.09231}
}
```

# Acknowledgment
Thanks to Jeffrey Smith (jeffreys@nvidia.com) for assistance in preparing this release.
