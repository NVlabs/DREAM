# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

#!/usr/bin/env python2

import argparse
import os

import cv2
import numpy as np
from PIL import Image as PILImage
from pyrr import Quaternion
from ruamel.yaml import YAML
import torch
import webcolors

import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Empty
import tf2_ros as tf2
import tf

import dream

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# USERS: Please change the ROS topics / frames below as desired.

# ROS topic for listening to RGB images
image_topic = "/camera/color/image_raw"

# ROS topic for listening to camera intrinsics
camera_info_topic = "/camera/color/camera_info"

# ROS service for sending request to capture frame
capture_frame_service_topic = "/dream/capture_frame"

# ROS service for sending request to clear buffer
clear_buffer_service_topic = "/dream/clear_buffer"

# ROS topics for outputs
topic_out_net_input_image = "/dream/net_input_image"
topic_out_keypoint_overlay = "/dream/keypoint_overlay"
topic_out_belief_maps = "/dream/belief_maps"
topic_out_keypoint_belief_overlay = "/dream/keypoint_belief_overlay"
topic_out_keypoint_names = "/dream/keypoint_names"
topic_out_keypoint_frame_overlay = "/dream/keypoint_frame_overlay"

# ROS frames for the output of DREAM
# tform_out_basename is now set by the user - previously was 'dream/base_frame'
tform_out_childname = "dream/camera_rgb_frame"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class DreamInferenceROS:
    def __init__(self, args, single_frame_mode=True, compute_2d_to_3d_transform=False):
        """Initialize inference engine.

            single_frame_mode:  Set this to True.  (Appears to be some sort of future-proofing by Tim.)
        """
        self.cv_image = None
        self.camera_K = None

        # TODO: -- continuous mode produces a TF at each frame
        # not continuous mode allows for several frames before producing an estimate
        self.single_frame_mode = single_frame_mode
        self.capture_frame_srv = rospy.Service(
            capture_frame_service_topic, Empty, self.on_capture_frame
        )
        self.clear_buffer_srv = rospy.Service(
            clear_buffer_service_topic, Empty, self.on_clear_buffer
        )
        self.kp_projs_raw_buffer = np.array([])
        self.kp_positions_buffer = np.array([])
        self.pnp_solution_found = False
        self.capture_frame_max_kps = True

        self.compute_2d_to_3d_transform = compute_2d_to_3d_transform

        # Create subscribers
        self.image_sub = rospy.Subscriber(
            image_topic, Image, self.on_image, queue_size=1
        )
        self.bridge = CvBridge()

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
        self.dream_network = dream.create_network_from_config_data(network_config)

        print(
            "Loading network with weights from:  {} ...".format(args.input_params_path)
        )
        self.dream_network.model.load_state_dict(torch.load(args.input_params_path))
        self.dream_network.enable_evaluation()

        # Use image preprocessing specified by config by default, unless user specifies otherwise
        if args.image_preproc_override:
            self.image_preprocessing = args.image_preproc_override
        else:
            self.image_preprocessing = self.dream_network.image_preprocessing()

        # Output names used to look up keypoints in TF tree
        self.keypoint_tf_frames = self.dream_network.ros_keypoint_frames
        print("ROS keypoint frames: {}".format(self.keypoint_tf_frames))

        # Define publishers
        self.net_input_image_pub = rospy.Publisher(
            topic_out_net_input_image, Image, queue_size=1
        )
        self.image_overlay_pub = rospy.Publisher(
            topic_out_keypoint_overlay, Image, queue_size=1
        )
        self.belief_image_pub = rospy.Publisher(
            topic_out_belief_maps, Image, queue_size=1
        )
        self.kp_belief_overlay_pub = rospy.Publisher(
            topic_out_keypoint_belief_overlay, Image, queue_size=1
        )
        self.kp_frame_overlay_pub = rospy.Publisher(
            topic_out_keypoint_frame_overlay, Image, queue_size=1
        )

        # Store the base frame for the TF lookup
        self.base_tf_frame = args.base_frame

        # Define TFs
        self.tfBuffer = tf2.Buffer()
        self.tf_broadcaster = tf2.TransformBroadcaster()
        self.listener = tf2.TransformListener(self.tfBuffer)
        self.camera_pose_tform = TransformStamped()

        self.camera_pose_tform.header.frame_id = self.base_tf_frame
        self.camera_pose_tform.child_frame_id = tform_out_childname

        # Subscriber for camera intrinsics topic
        self.camera_info_sub = rospy.Subscriber(
            camera_info_topic, CameraInfo, self.on_camera_info, queue_size=1
        )

        # Verbose mode
        self.verbose = args.verbose

    def on_capture_frame(self, req):
        print("Capturing frame.")
        found_kp_projs_net_input = dream_ros.process_image()
        print(found_kp_projs_net_input)
        (
            kp_projs_raw_good_sample,
            kp_positions_good_sample,
        ) = dream_ros.keypoint_correspondences(found_kp_projs_net_input)
        if self.capture_frame_max_kps and kp_projs_raw_good_sample is not None:
            n_found_keypoints = kp_projs_raw_good_sample.shape[0]
            if n_found_keypoints != self.dream_network.n_keypoints:
                print(
                    "Only found {} keypoints -- not continuing. Try again.".format(
                        n_found_keypoints
                    )
                )
                return []
        if (
            kp_projs_raw_good_sample is not None
            and kp_positions_good_sample is not None
        ):
            dream_ros.solve_pnp_buffer(
                kp_projs_raw_good_sample, kp_positions_good_sample
            )
        return []

    def on_clear_buffer(self, req):
        print("Clearing frame buffer.")
        self.kp_projs_raw_buffer = np.array([])
        self.kp_positions_buffer = np.array([])
        self.pnp_solution_found = False
        return []

    def on_image(self, image):
        self.cv_image = self.bridge.imgmsg_to_cv2(image, "rgb8")

    def on_camera_info(self, camera_info):
        # Create camera intrinsics matrix
        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]
        self.camera_K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    def process_image(self):
        """Performs inference on the image most recently captured.

        Input (none)
        self.cv_image:  Holds the image to be processed.

        Returns
        detected_keypoints:  Array of 2D keypoint coords wrt original image, possibly including missing keypoints
        """

        if self.cv_image is None:
            return

        # Determine if we need debug content from single-image inference based on subscribers
        num_connect_belief_image_pub = self.belief_image_pub.get_num_connections()
        num_connect_net_input_image_pub = self.net_input_image_pub.get_num_connections()
        num_connect_image_overlay_pub = self.image_overlay_pub.get_num_connections()
        num_connect_kp_belief_overlay_pub = (
            self.kp_belief_overlay_pub.get_num_connections()
        )

        if (
            num_connect_belief_image_pub > 0
            or num_connect_net_input_image_pub > 0
            or num_connect_image_overlay_pub > 0
            or num_connect_kp_belief_overlay_pub > 0
        ):
            debug_inference = True
        else:
            debug_inference = False

        # Detect keypoints from single-image inference
        image_raw = PILImage.fromarray(self.cv_image)
        detection_result = self.dream_network.keypoints_from_image(
            image_raw,
            image_preprocessing_override=self.image_preprocessing,
            debug=debug_inference,
        )
        detected_keypoints = detection_result["detected_keypoints"]

        # Publish debug topics - only do this if something is subscribed
        # TODO: clean up so some of the intermediate processing isn't duplicated
        if num_connect_belief_image_pub > 0:
            belief_maps = detection_result["belief_maps"]
            belief_images = dream.image_proc.images_from_belief_maps(
                belief_maps, normalization_method=6
            )
            belief_image_mosaic = dream.image_proc.mosaic_images(
                belief_images, rows=1, cols=len(belief_images), inner_padding_px=5
            )
            cv_belief_image = np.array(belief_image_mosaic)
            cv_belief_image = cv_belief_image[:, :, ::-1].copy()
            belief_msg = self.bridge.cv2_to_imgmsg(cv_belief_image, encoding="bgr8")
            self.belief_image_pub.publish(belief_msg)

        if num_connect_net_input_image_pub > 0:
            net_input_image = detection_result["image_rgb_net_input"]
            cv_input_image = np.array(net_input_image)
            cv_input_image = cv_input_image[:, :, ::-1].copy()
            net_input_image_msg = self.bridge.cv2_to_imgmsg(
                cv_input_image, encoding="bgr8"
            )
            self.net_input_image_pub.publish(net_input_image_msg)

        if num_connect_image_overlay_pub > 0:
            # TODO: fix color cycle
            if self.dream_network.n_keypoints == 7:
                point_colors = [
                    "red",
                    "blue",
                    "green",
                    "yellow",
                    "black",
                    "cyan",
                    "white",
                ]
            else:
                point_colors = "red"

            image_overlay = dream.image_proc.overlay_points_on_image(
                image_raw,
                detected_keypoints,
                self.dream_network.friendly_keypoint_names,
                annotation_color_dot=point_colors,
                annotation_color_text=point_colors,
            )
            cv_image_overlay = np.array(image_overlay)
            cv_image_overlay = cv_image_overlay[:, :, ::-1].copy()
            image_overlay_msg = self.bridge.cv2_to_imgmsg(
                cv_image_overlay, encoding="bgr8"
            )
            self.image_overlay_pub.publish(image_overlay_msg)

        if num_connect_kp_belief_overlay_pub > 0:
            image_raw_resolution = (self.cv_image.shape[1], self.cv_image.shape[0])
            net_input_resolution = detection_result["image_rgb_net_input"].size
            belief_maps = detection_result["belief_maps"]
            flattened_belief_tensor = belief_maps.sum(dim=0)
            flattened_belief_image = dream.image_proc.image_from_belief_map(
                flattened_belief_tensor, colormap="hot", normalization_method=6
            )
            flattened_belief_image_netin = dream.image_proc.convert_image_to_netin_from_netout(
                flattened_belief_image, net_input_resolution
            )
            flattened_belief_image_raw = dream.image_proc.inverse_preprocess_image(
                flattened_belief_image_netin,
                image_raw_resolution,
                self.image_preprocessing,
            )
            flattened_belief_image_raw_blend = PILImage.blend(
                image_raw, flattened_belief_image_raw, alpha=0.5
            )

            # Overlay keypoints
            # TODO: fix color cycle
            if self.dream_network.n_keypoints == 7:
                point_colors = [
                    "red",
                    "blue",
                    "green",
                    "yellow",
                    "black",
                    "cyan",
                    "white",
                ]
            else:
                point_colors = "red"

            kp_belief_overlay = dream.image_proc.overlay_points_on_image(
                flattened_belief_image_raw_blend,
                detected_keypoints,
                self.dream_network.friendly_keypoint_names,
                annotation_color_dot=point_colors,
                annotation_color_text=point_colors,
            )
            cv_kp_belief_overlay = np.array(kp_belief_overlay)
            cv_kp_belief_overlay = cv_kp_belief_overlay[:, :, ::-1].copy()
            kp_belief_overlay_msg = self.bridge.cv2_to_imgmsg(
                cv_kp_belief_overlay, encoding="bgr8"
            )
            self.kp_belief_overlay_pub.publish(kp_belief_overlay_msg)

        return detected_keypoints

    def keypoint_correspondences(self, detected_kp_projs):
        """Convert 2D keypoint coords to (2D, 3D) pairs of correspondences.

        Input:
        detected_kp_projs:  Array of 2D keypoint coords wrt original image, possibly including missing keypoints

        Returns:
        kp_projs_raw_good_sample:  Array of 2D keypoint coords wrt original image
        kp_positions_good_sample:  Array of 3D keypoint coords
        """

        if not self.compute_2d_to_3d_transform:
            return None, None

        all_kp_positions = []

        for i in range(len(self.keypoint_tf_frames)):
            keypoint_tf_frame = self.keypoint_tf_frames[i]
            if self.verbose:
                print(
                    "Attempting transform lookup between {} and {}...".format(
                        self.base_tf_frame, keypoint_tf_frame
                    )
                )
            try:
                tform = self.tfBuffer.lookup_transform(
                    self.base_tf_frame, keypoint_tf_frame, rospy.Time()
                )
                this_pos = np.array(
                    [
                        tform.transform.translation.x,
                        tform.transform.translation.y,
                        tform.transform.translation.z,
                    ]
                )
                all_kp_positions.append(this_pos)

            except tf2.TransformException as e:
                print("TF Exception: {}".format(e))
                return None, None

        # Now determine which to keep
        kp_projs_raw_good_sample = []
        kp_positions_good_sample = []
        for this_kp_proj_est, this_kp_position in zip(
            detected_kp_projs, all_kp_positions
        ):
            if (
                this_kp_proj_est is not None
                and this_kp_proj_est[0]
                and this_kp_proj_est[1]
                and not (this_kp_proj_est[0] < -999.0 and this_kp_proj_est[1] < 0.999)
            ):
                # This keypoint is defined to exist within the image frame, so we keep it
                kp_projs_raw_good_sample.append(this_kp_proj_est)
                kp_positions_good_sample.append(this_kp_position)

        kp_projs_raw_good_sample = np.array(kp_projs_raw_good_sample)
        kp_positions_good_sample = np.array(kp_positions_good_sample)

        return kp_projs_raw_good_sample, kp_positions_good_sample

    def solve_pnp_buffer(self, candidate_kp_projs_raw, candidate_kp_positions):

        if self.camera_K is None:
            self.pnp_solution_found = False
            return

        kp_projs_raw_to_try = np.array(
            self.kp_projs_raw_buffer.tolist() + candidate_kp_projs_raw.tolist()
        )
        kp_positions_to_try = np.array(
            self.kp_positions_buffer.tolist() + candidate_kp_positions.tolist()
        )

        if self.verbose:
            print("\nSolving for PNP... ~~~~~~~~~~~~~~~~~~~~")
            print("2D Detected KP projections in raw:")
            print(kp_projs_raw_to_try)

            print("3D KP positions:")
            print(kp_positions_to_try)

        pnp_retval, tvec, quat = dream.geometric_vision.solve_pnp(
            kp_positions_to_try, kp_projs_raw_to_try, self.camera_K
        )

        if pnp_retval:
            self.pnp_solution_found = True

            if self.verbose:
                print("Camera-from-robot pose, found by PNP:")
                print(tvec)
                print(quat)

            # Update transform
            T_cam_from_robot = tf.transformations.quaternion_matrix(quat)
            T_cam_from_robot[:3, -1] = tvec
            T_robot_from_cam = tf.transformations.inverse_matrix(T_cam_from_robot)

            robot_from_cam_pos = T_robot_from_cam[:3, -1]

            R_robot_from_cam = T_robot_from_cam[:3, :3]
            temp = tf.transformations.identity_matrix()
            temp[:3, :3] = R_robot_from_cam
            robot_from_cam_quat = tf.transformations.quaternion_from_matrix(temp)

            # Update transform msg for publication
            self.camera_pose_tform.transform.translation.x = robot_from_cam_pos[0]
            self.camera_pose_tform.transform.translation.y = robot_from_cam_pos[1]
            self.camera_pose_tform.transform.translation.z = robot_from_cam_pos[2]

            self.camera_pose_tform.transform.rotation.x = robot_from_cam_quat[0]
            self.camera_pose_tform.transform.rotation.y = robot_from_cam_quat[1]
            self.camera_pose_tform.transform.rotation.z = robot_from_cam_quat[2]
            self.camera_pose_tform.transform.rotation.w = robot_from_cam_quat[3]

            # Save buffer since we got a PNP solution if we're not in single-frame mode
            if not self.single_frame_mode:
                self.kp_projs_raw_buffer = kp_projs_raw_to_try
                self.kp_positions_buffer = kp_positions_to_try
                print(
                    "Adding to buffer! New buffer size: {}".format(
                        self.kp_positions_buffer.shape[0]
                    )
                )

        else:
            print("PnP failed to provide a solution.")
            self.pnp_solution_found = False

    def publish_pose(self):
        self.camera_pose_tform.header.stamp = rospy.Time().now()
        self.tf_broadcaster.sendTransform(self.camera_pose_tform)

        # Generate and publish keypoint frame overlay
        if self.kp_frame_overlay_pub.get_num_connections() > 0:

            if self.cv_image is None or self.camera_K is None:
                return

            all_kp_transforms = []
            for i in range(len(self.keypoint_tf_frames)):
                keypoint_tf_frame = self.keypoint_tf_frames[i]
                # Lookup transform between dream published frame and keypoint frames
                try:
                    tform = self.tfBuffer.lookup_transform(
                        self.camera_pose_tform.child_frame_id,
                        keypoint_tf_frame,
                        rospy.Time(),
                    )
                    all_kp_transforms.append(tform)

                except tf2.TransformException as e:
                    print("TF Exception: {}".format(e))
                    return None, None

            cv_image_overlay = self.cv_image.copy()

            frame_len = 0.1
            frame_thickness = 3
            frame_triad_pts = np.array(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [frame_len, 0.0, 0.0, 1.0],
                    [0.0, frame_len, 0.0, 1.0],
                    [0.0, 0.0, frame_len, 1.0],
                ]
            )
            shift = 4
            factor = 1 << shift
            point_radius = 4.0

            for kp_tform in all_kp_transforms:
                pos = [
                    kp_tform.transform.translation.x,
                    kp_tform.transform.translation.y,
                    kp_tform.transform.translation.z,
                ]
                quat = [
                    kp_tform.transform.rotation.x,
                    kp_tform.transform.rotation.y,
                    kp_tform.transform.rotation.z,
                    kp_tform.transform.rotation.w,
                ]
                T = tf.transformations.quaternion_matrix(quat)
                T[:3, -1] = pos

                frame_triad_positions_homog = np.transpose(
                    np.matmul(T, np.transpose(frame_triad_pts))
                )
                frame_triad_positions = [
                    dream.geometric_vision.hnormalized(v).tolist()
                    for v in frame_triad_positions_homog
                ]
                frame_triad_projs = dream.geometric_vision.point_projection_from_3d(
                    self.camera_K, frame_triad_positions
                )

                # Overlay line on image
                point0_fixedpt = (
                    int(frame_triad_projs[0][0] * factor),
                    int(frame_triad_projs[0][1] * factor),
                )
                point1_fixedpt = (
                    int(frame_triad_projs[1][0] * factor),
                    int(frame_triad_projs[1][1] * factor),
                )
                point2_fixedpt = (
                    int(frame_triad_projs[2][0] * factor),
                    int(frame_triad_projs[2][1] * factor),
                )
                point3_fixedpt = (
                    int(frame_triad_projs[3][0] * factor),
                    int(frame_triad_projs[3][1] * factor),
                )

                # x-axis
                cv_image_overlay = cv2.line(
                    cv_image_overlay,
                    point0_fixedpt,
                    point1_fixedpt,
                    webcolors.name_to_rgb("red"),
                    thickness=frame_thickness,
                    shift=shift,
                )
                # y-axis
                cv_image_overlay = cv2.line(
                    cv_image_overlay,
                    point0_fixedpt,
                    point2_fixedpt,
                    webcolors.name_to_rgb("green"),
                    thickness=frame_thickness,
                    shift=shift,
                )
                # z-axis
                cv_image_overlay = cv2.line(
                    cv_image_overlay,
                    point0_fixedpt,
                    point3_fixedpt,
                    webcolors.name_to_rgb("blue"),
                    thickness=frame_thickness,
                    shift=shift,
                )
                # center of frame triad
                radius_fixedpt = int(point_radius * factor)
                cv_image_overlay = cv2.circle(
                    cv_image_overlay,
                    point0_fixedpt,
                    radius_fixedpt,
                    webcolors.name_to_rgb("black"),
                    thickness=-1,
                    shift=shift,
                )

            cv_image_overlay = cv_image_overlay[:, :, ::-1]
            image_overlay_msg = self.bridge.cv2_to_imgmsg(
                cv_image_overlay, encoding="bgr8"
            )
            self.kp_frame_overlay_pub.publish(image_overlay_msg)


if __name__ == "__main__":

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
        "-b",
        "--base-frame",
        required=True,
        help="The ROS TF name for the base frame of the robot, which serves as the canonical frame for PnP.",
    )
    parser.add_argument(
        "-r",
        "--node-rate",
        type=float,
        default=10.0,
        help="The rate in Hz for this node to run.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Outputs all diagnostic information to the screen.",
    )
    parser.add_argument(
        "-p",
        "--image-preproc-override",
        default=None,
        help="Overrides the image preprocessing specified by the network. (Debug argument.)",
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

    # Initialize ROS node
    rospy.init_node("dream")

    # Create DREAM inference engine
    single_frame_mode = True
    mode_str = "single-frame mode" if single_frame_mode else "multi-frame mode"
    dream_ros = DreamInferenceROS(
        args, single_frame_mode, compute_2d_to_3d_transform=True
    )
    print("DREAM ~ online ~ " + mode_str)

    # Main loop
    rate = rospy.Rate(args.node_rate)
    while not rospy.is_shutdown():

        # Find keypoints in image
        found_kp_projs_net_input = dream_ros.process_image()

        # Solve PNP (if in single frame mode)
        if single_frame_mode and found_kp_projs_net_input is not None:
            (
                kp_projs_raw_good_sample,
                kp_positions_good_sample,
            ) = dream_ros.keypoint_correspondences(found_kp_projs_net_input)
            if (
                kp_projs_raw_good_sample is not None
                and kp_positions_good_sample is not None
            ):
                dream_ros.solve_pnp_buffer(
                    kp_projs_raw_good_sample, kp_positions_good_sample
                )

        # Publish pose (if found)
        if dream_ros.pnp_solution_found:
            dream_ros.publish_pose()

        rate.sleep()
