# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import os
from PIL import Image as PILImage

import numpy as np
import ruamel.yaml
import torch
import torchvision.transforms as TVTransforms

from .spatial_softmax import SoftArgmaxPavlo
import dream

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

KNOWN_ARCHITECTURES = [
    "vgg",
    "resnet",
]

KNOWN_OPTIMIZERS = [
    "adam",  # the Adam optimizer
    "sgd",
]  # the Stochastic Gradient Descent optimizer


def create_network_from_config_file(config_file_path, network_params_path=None):

    # Input argument handling
    assert os.path.exists(
        config_file_path
    ), 'Expected config_file_path "{}" to exist, but it does not.'.format(
        config_file_path
    )

    if network_params_path:
        load_network_parameters = True
        assert os.path.exists(
            network_params_path
        ), 'If provided, expected network_params_path "{}" to exist, but it does not.'.format(
            network_params_path
        )
    else:
        load_network_parameters = False

    # Create parser
    data_parser = ruamel.yaml.YAML(typ="safe")

    print('Loading network config file "{}"'.format(config_file_path))
    with open(config_file_path, "r") as f:
        network_config = data_parser.load(f)

    # Create the network
    dream_network = create_network_from_config_data(network_config)

    # Optionally load weights of the network
    if load_network_parameters:
        print('Loading network weights file "{}"'.format(network_params_path))
        dream_network.model.load_state_dict(torch.load(network_params_path))

    return dream_network


def create_network_from_config_data(network_config_data):

    # Create the network
    dream_network = DreamNetwork(network_config_data)
    return dream_network


class DreamNetwork:
    def __init__(self, network_config):

        # Assert input
        assert (
            "architecture" in network_config
        ), 'Required key "architecture" is missing from network configuration.'
        assert (
            "type" in network_config["architecture"]
        ), 'Required key "type" in dictionary "architecture" is missing from network configuration.'
        assert (
            "manipulator" in network_config
        ), 'Required key "manipulator" is missing from network configuration.'
        assert (
            "name" in network_config["manipulator"]
        ), 'Required key "name" in dictionary "manipulator" is missing from network configuration.'
        assert (
            "keypoints" in network_config["manipulator"]
        ), 'Required key "keypoints" in dictionary "manipulator" is missing from network configuration.'

        # Parse keypoint specification
        self.keypoint_names = []
        self.friendly_keypoint_names = []
        self.ros_keypoint_frames = []
        for kp_def in network_config["manipulator"]["keypoints"]:
            assert "name" in kp_def, 'Keypoint specification is missing key "name".'
            kp_name = kp_def["name"]
            self.keypoint_names.append(kp_name)

            friendly_kp_name = (
                kp_def["friendly_name"] if "friendly_name" in kp_def else kp_name
            )
            self.friendly_keypoint_names.append(friendly_kp_name)

            ros_kp_frame = kp_def["ros_frame"] if "ros_frame" in kp_def else kp_name
            self.ros_keypoint_frames.append(ros_kp_frame)

        self.network_config = network_config

        # TBD: the following "getters" should all be methods
        self.manipulator_name = self.network_config["manipulator"]["name"]
        self.n_keypoints = len(self.keypoint_names)
        self.architecture_type = self.network_config["architecture"]["type"]

        # print robot info
        print("`network.py`.  `DreamNetwork:__init()` ----------")
        print("  Manipulator: {}".format(self.manipulator_name))
        print("  Keypoint names: {}".format(self.keypoint_names))
        print("  Friendly keypoint names: {}".format(self.friendly_keypoint_names))
        print("  Architecture type: {}".format(self.architecture_type))

        # Parse normalization
        assert (
            "image_normalization" in self.network_config["architecture"]
        ), 'Required key "image_normalization" in dictionary "architecture" is missing from network configuration.'
        self.image_normalization = self.network_config["architecture"][
            "image_normalization"
        ]

        # Parse image preprocessing: how to handle what the network does between the input image and the network input layer
        assert (
            "image_preprocessing" in self.network_config["architecture"]
        ), 'Required key "image_preprocessing" in dictionary "architecture" is missing from network configuration.'
        assert (
            self.image_preprocessing() in dream.image_proc.KNOWN_IMAGE_PREPROC_TYPES
        ), 'Image preprocessing type "{}" is not recognized.'.format(
            self.image_preprocessing()
        )

        # Assert the output heads have been specified
        assert (
            "output_heads" in self.network_config["architecture"]
        ), 'Required key "output_heads" in dictionary "architecture" is missing from network configuration.'

        assert (
            self.architecture_type in KNOWN_ARCHITECTURES
        ), 'Expected architecture type "{}" to be in the list of known network architectures, but it is not.'.format(
            self.architecture_type
        )

        # Assert that the input heads have been specified
        assert (
            "input_heads" in self.network_config["architecture"]
        ), 'Required key "input_heads" in dictionary "architecture" is missing from network configuration.'
        assert (
            self.network_config["architecture"]["input_heads"][0] == "image_rgb"
        ), 'First input head must be "image_rgb".'

        # Assert that the trained network input resolution has been specified
        # The output resolution is calculated at the end, after the model is created
        assert (
            "training" in self.network_config
        ), 'Required key "training" is missing from network configuration.'
        assert (
            "config" in self.network_config["training"]
        ), 'Required key "config" in dictionary "training" is missing from network configuration.'
        assert (
            "net_input_resolution" in self.network_config["training"]["config"]
        ), 'Required key "net_input_resolution" is missing from training configuration.'
        len_trained_net_input_res = len(
            self.network_config["training"]["config"]["net_input_resolution"]
        )
        assert (
            len_trained_net_input_res == 2
        ), "Expected trained net input resolution to have length 2, but it has length {}.".format(
            len_trained_net_input_res
        )

        assert (
            "platform" in self.network_config["training"]
        ), 'Required key "platform" in dictionary "training" is missing from network configuration.'
        gpu_ids = self.network_config["training"]["platform"]["gpu_ids"]
        data_parallel_device_ids = gpu_ids if gpu_ids else None

        # Options to sort belief map
        # Use belief map scores to try to determine cases in multiple instances
        self.use_belief_peak_scores = True
        # This in the scale of the belief map, which is 0 - 1.0.
        self.belief_peak_next_best_score = 0.25

        # Create architectures
        if self.architecture_type == "vgg":
            vgg_kwargs = {}
            if "spatial_softmax" in self.network_config["architecture"]:
                assert self.network_config["architecture"]["output_heads"] == [
                    "belief_maps",
                    "keypoints",
                ]
                vgg_kwargs = {
                    "internalize_spatial_softmax": True,
                    "learned_beta": self.network_config["architecture"][
                        "spatial_softmax"
                    ]["learned_beta"],
                    "initial_beta": self.network_config["architecture"][
                        "spatial_softmax"
                    ]["initial_beta"],
                }
            else:
                assert self.network_config["architecture"]["output_heads"] == [
                    "belief_maps"
                ]
                vgg_kwargs = {"internalize_spatial_softmax": False}

            # Check if using new decoder -- assume output is 100x100 if not
            if (
                "deconv_decoder" in self.network_config["architecture"]
                and not "full_output" in self.network_config["architecture"]
            ):
                use_deconv_decoder = self.network_config["architecture"][
                    "deconv_decoder"
                ]
                vgg_kwargs["deconv_decoder"] = use_deconv_decoder
            elif "full_output" in self.network_config["architecture"]:
                use_deconv_decoder = self.network_config["architecture"][
                    "deconv_decoder"
                ]
                vgg_kwargs["deconv_decoder"] = use_deconv_decoder
                vgg_kwargs["full_output"] = True

                if "n_stages" in self.network_config["architecture"]:
                    vgg_kwargs["n_stages"] = self.network_config[
                        "architecture"
                    ]["n_stages"]

            # Check if skip connections -- use default if not
            if "skip_connections" in self.network_config["architecture"]:
                vgg_kwargs["skip_connections"] = self.network_config[
                    "architecture"
                ]["skip_connections"]

            if "n_stages" in self.network_config["architecture"]:
                self.model = torch.nn.DataParallel(
                    dream.models.DreamHourglassMultiStage(
                        self.n_keypoints, **vgg_kwargs
                    ),
                    device_ids=data_parallel_device_ids,
                ).cuda()
            else:
                self.model = torch.nn.DataParallel(
                    dream.models.DreamHourglass(
                        self.n_keypoints, **vgg_kwargs
                    ),
                    device_ids=data_parallel_device_ids,
                ).cuda()

            loss_type = self.network_config["architecture"]["loss"]["type"]

            if loss_type == "mse":
                self.criterion = torch.nn.MSELoss()
            elif loss_type == "huber":
                self.criterion = torch.nn.SmoothL1Loss()
            else:
                assert False, "Loss not yet implemented."

        elif self.architecture_type == "resnet":

            # Assume we're only training on belief maps
            assert self.network_config["architecture"]["output_heads"] == [
                "belief_maps"
            ]

            resnet_kwargs = {}

            # Check if using "full" output -- assume output is "half" if not
            if "full_decoder" in self.network_config["architecture"]:
                use_full_decoder = self.network_config["architecture"]["full_decoder"]
                resnet_kwargs["full"] = use_full_decoder

            self.model = torch.nn.DataParallel(
                dream.models.ResnetSimple(self.n_keypoints, **resnet_kwargs),
                device_ids=data_parallel_device_ids,
            ).cuda()

            loss_type = self.network_config["architecture"]["loss"]["type"]

            if loss_type == "mse":
                self.criterion = torch.nn.MSELoss()
            elif loss_type == "huber":
                self.criterion = torch.nn.SmoothL1Loss()
            else:
                assert False, "Loss not yet implemented."

        else:
            assert False, 'Network architecture type "{}" not defined.'.format(
                self.architecture_type
            )

        # Optimizer is created in a separate call, because this isn't needed unless we're training
        self.optimizer = None

        # Now determine trained network output resolution, if not specified, or assert consistency if so
        trained_net_out_res_from_model = self.net_output_resolution_from_input_resolution(
            self.trained_net_input_resolution()
        )
        trained_net_out_res_from_model_as_list = list(trained_net_out_res_from_model)
        if "net_output_resolution" in self.network_config["training"]["config"]:
            assert (
                self.network_config["training"]["config"]["net_output_resolution"]
                == trained_net_out_res_from_model_as_list
            ), "Network model and config file disagree for trained network output resolution."
        else:
            # Add to config
            self.network_config["training"]["config"][
                "net_output_resolution"
            ] = trained_net_out_res_from_model_as_list

    def trained_net_input_resolution(self):
        return tuple(self.network_config["training"]["config"]["net_input_resolution"])

    def trained_net_output_resolution(self):
        return tuple(self.network_config["training"]["config"]["net_output_resolution"])

    def image_preprocessing(self):
        return self.network_config["architecture"]["image_preprocessing"]

    def train(self, network_input_heads, target):
        assert self.optimizer, "Optimizer must be defined. Use enable_training() first."

        self.optimizer.zero_grad()

        loss = self.loss(network_input_heads, target)

        loss.backward()
        self.optimizer.step()

        return loss

    def loss(self, network_input_heads, target):
        network_output_heads = self.model(network_input_heads[0])

        if self.network_config["architecture"]["output_heads"] == ["belief_maps"]:

            if "n_stages" in self.network_config["architecture"]:
                # print('hello')
                n_stages = len(network_output_heads)
                expanded_size = [n_stages] + [-1] * target.dim()
                target_expanded = target.unsqueeze(0).expand(expanded_size)
                loss = self.criterion(
                    torch.stack(network_output_heads), target_expanded
                )

                # This was the loss from before
                # loss = torch.tensor(0).float().cuda()
                # for belief in network_output_heads: loss += ((belief - target) * (belief - target)).mean()

            else:
                loss = self.criterion(network_output_heads[0], target)

        else:
            assert False, "Not yet implemented."

        return loss

    # image_raw_resolution: (width, height) in pixels
    # calls resolution_after_preprocessing under the hood, using network trained resolution as the reference resolution
    def net_resolutions_from_image_raw_resolution(
        self, image_raw_resolution, image_preprocessing_override=None
    ):

        # Input argument handling
        assert (
            len(image_raw_resolution) == 2
        ), 'Expected "image_raw_resolution" to have length 2, but it has length {}.'.format(
            len(image_raw_resolution)
        )

        image_preprocessing = (
            image_preprocessing_override
            if image_preprocessing_override
            else self.image_preprocessing()
        )

        # Calculate image resolution at network input layer, after preprocessing
        net_input_resolution = dream.image_proc.resolution_after_preprocessing(
            image_raw_resolution,
            self.trained_net_input_resolution(),
            image_preprocessing,
        )
        net_output_resolution = self.net_output_resolution_from_input_resolution(
            net_input_resolution
        )

        return net_input_resolution, net_output_resolution

    def net_output_resolution_from_input_resolution(self, net_input_resolution):

        # Input argument handling
        assert (
            len(net_input_resolution) == 2
        ), 'Expected "net_input_resolution" to have length 2, but it has length {}.'.format(
            len(net_input_resolution)
        )

        netin_width, netin_height = net_input_resolution

        # Construct test input and send thru network to get the output
        # This assumes there is only one input head, which is the RGB image
        with torch.no_grad():
            net_input_as_tensor_batch = torch.zeros(
                1, 3, netin_height, netin_width
            ).cuda()
            net_output_as_tensor_batch = self.model(net_input_as_tensor_batch)
            net_output_shape = net_output_as_tensor_batch[0][0].shape
            net_output_resolution = (net_output_shape[2], net_output_shape[1])

        return net_output_resolution

    # Wrapper for inferences from a PIL image directly
    # Returns keypoints in the input image (not necessarily network input) frame
    # Allows for an optional overwrite
    def keypoints_from_image(
        self, input_rgb_image_as_pil, image_preprocessing_override=None, debug=False
    ):

        # Input handling
        assert isinstance(
            input_rgb_image_as_pil, PILImage.Image
        ), 'Expected "input_rgb_image_as_pil" to be a PIL Image, but it is {}.'.format(
            type(input_rgb_image_as_pil)
        )
        input_image_resolution = input_rgb_image_as_pil.size

        # do preprocessing
        image_preprocessing = (
            image_preprocessing_override
            if image_preprocessing_override
            else self.image_preprocessing()
        )

        input_image_preproc_before_norm = dream.image_proc.preprocess_image(
            input_rgb_image_as_pil,
            self.trained_net_input_resolution(),
            image_preprocessing,
        )

        netin_res_inf = input_image_preproc_before_norm.size
        tensor_from_image_tform = TVTransforms.Compose(
            [
                TVTransforms.ToTensor(),
                TVTransforms.Normalize(
                    self.image_normalization["mean"], self.image_normalization["stdev"]
                ),
            ]
        )
        input_rgb_image_as_tensor = tensor_from_image_tform(
            input_image_preproc_before_norm
        )

        # Inference
        with torch.no_grad():
            input_rgb_image_as_tensor_batch = input_rgb_image_as_tensor.unsqueeze(
                0
            ).cuda()
            belief_maps_net_out_batch, detected_kp_projs_net_out_batch = self.inference(
                input_rgb_image_as_tensor_batch
            )

        belief_maps_net_out = belief_maps_net_out_batch[0]
        detected_kp_projs_net_out = np.array(
            detected_kp_projs_net_out_batch[0], dtype=float
        )

        # Get network output resolution for this inference based on belief maps
        belief_map = belief_maps_net_out[0]
        netout_res_inf = (belief_map.shape[1], belief_map.shape[0])

        # Convert keypoints from net-output to net-input
        detected_kp_projs_net_in = dream.image_proc.convert_keypoints_to_netin_from_netout(
            detected_kp_projs_net_out, netout_res_inf, netin_res_inf
        )
        detected_kp_projs = dream.image_proc.convert_keypoints_to_raw_from_netin(
            detected_kp_projs_net_in,
            netin_res_inf,
            input_image_resolution,
            image_preprocessing,
        )

        detection_result = {"detected_keypoints": detected_kp_projs}
        if debug:
            detection_result["image_rgb_net_input"] = input_image_preproc_before_norm
            detection_result["belief_maps"] = belief_maps_net_out
            detection_result[
                "detected_keypoints_net_output"
            ] = detected_kp_projs_net_out
            detection_result["detected_keypoints_net_input"] = detected_kp_projs_net_in

        return detection_result

    # Inference is designed to return the best output of belief_maps and keypoints
    # This is an abstraction layer so even if multiple stages are used, this only produces one set of outputs (for the final stage)
    def inference(self, network_input):

        if self.architecture_type == "dream_legacy":
            network_head_outputs = self.model(network_input)
            belief_maps = network_head_outputs[0]
            detected_kp_projections = self.soft_argmax(belief_maps)
            network_output = [belief_maps, detected_kp_projections]

        elif self.network_config["architecture"]["output_heads"] == [
            "belief_maps",
            "keypoints",
        ]:
            network_output = self.model(network_input)

        elif self.network_config["architecture"]["output_heads"] == ["belief_maps"]:

            network_head_output = self.model(network_input)

            # use the last layer
            belief_maps_batch = network_head_output[-1]

            # # If we need to transpose the belief map -- this is done here:
            # for x in range(belief_maps_batch.shape[0]):
            #     bmaps = belief_maps_batch[x]
            #     belief_maps_batch[x] = torch.cat([torch.t(b).unsqueeze(0) for b in bmaps])

            peaks_from_belief_maps_kwargs = {}
            (
                trained_net_output_width,
                trained_net_output_height,
            ) = self.trained_net_output_resolution()
            if trained_net_output_width >= 400 and trained_net_output_height >= 400:
                peaks_from_belief_maps_kwargs["offset_due_to_upsampling"] = 0.0
            else:
                # heuristic based on previous work with smaller belief maps
                peaks_from_belief_maps_kwargs["offset_due_to_upsampling"] = 0.4395

            detected_kp_projs_batch = []
            for belief_maps in belief_maps_batch:
                peaks = dream.image_proc.peaks_from_belief_maps(
                    belief_maps, **peaks_from_belief_maps_kwargs
                )

                # Determine keypoints from this
                detected_kp_projs = []
                for peak in peaks:

                    if len(peak) == 1:
                        detected_kp_projs.append([peak[0][0], peak[0][1]])
                    else:
                        if self.use_belief_peak_scores and len(peak) > 1:
                            # Try to use the belief map scores
                            peak_sorted_by_score = sorted(
                                peak, key=lambda x: x[2], reverse=True
                            )
                            if (
                                peak_sorted_by_score[0][2] - peak_sorted_by_score[1][2]
                                >= self.belief_peak_next_best_score
                            ):
                                # Keep the best score
                                detected_kp_projs.append(
                                    [
                                        peak_sorted_by_score[0][0],
                                        peak_sorted_by_score[0][1],
                                    ]
                                )
                            else:
                                # Can't determine -- return no detection
                                # Can't use None because we need to return it as a tensor
                                detected_kp_projs.append([-999.999, -999.999])

                        else:
                            # Can't determine -- return no detection
                            # Can't use None because we need to return it as a tensor
                            detected_kp_projs.append([-999.999, -999.999])

                detected_kp_projs_batch.append(detected_kp_projs)

            detected_kp_projs_batch = torch.tensor(detected_kp_projs_batch).float()

            network_output = [belief_maps_batch, detected_kp_projs_batch]

        else:
            assert (
                False
            ), "Could not determine how to conduct inference on this network."

        return network_output

    def save_network_config(self, config_file_path, overwrite=False):

        if not overwrite:
            assert not os.path.exists(
                config_file_path
            ), 'Output file already exists in "{}".'.format(config_file_path)

        # Create saver
        data_saver = ruamel.yaml.YAML()
        data_saver.default_flow_style = False
        data_saver.explicit_start = False

        with open(config_file_path, "w") as f:
            # TBD - convert to ruamel.yaml.comments.CommentedMap to get rid of !!omap in yaml
            data_saver.dump(self.network_config, f)

    def save_network_params(self, network_params_path, overwrite=False):

        if not overwrite:
            assert not os.path.exists(
                network_params_path
            ), 'Output file already exists in "{}".'.format(network_params_path)

        # Save weights
        torch.save(self.model.state_dict(), network_params_path)

    def save_network(
        self, output_dir, output_filename_without_extension, overwrite=False
    ):

        dream.utilities.makedirs(output_dir, exist_ok=overwrite)

        network_config_dir = os.path.join(
            output_dir, output_filename_without_extension + ".yaml"
        )
        self.save_network_config(network_config_dir, overwrite)

        network_params_path = os.path.join(
            output_dir, output_filename_without_extension + ".pth"
        )
        self.save_network_params(network_params_path, overwrite)

    def enable_training(self):

        # Load optimizer if needed
        if not self.optimizer:

            assert (
                "optimizer" in self.network_config["training"]["config"]
            ), 'Required key "optimizer" in dictionary "config" is missing from network configuration.'
            assert (
                "type" in self.network_config["training"]["config"]["optimizer"]
            ), 'Required key "type" in dictionary "optimizer" is missing from network configuration.'

            network_parameters = filter(
                lambda p: p.requires_grad, self.model.parameters()
            )
            optimizer_type = self.network_config["training"]["config"]["optimizer"][
                "type"
            ]

            assert (
                optimizer_type in KNOWN_OPTIMIZERS
            ), 'Expected optimizer_type "{}" to be in the list of known optimizers, but it is not.'.format(
                optimizer_type
            )

            if optimizer_type == "adam":

                assert (
                    "learning_rate"
                    in self.network_config["training"]["config"]["optimizer"]
                ), 'Required key "learning_rate" in dictionary "optimizer" is missing to use the Adam optimizer.'

                self.optimizer = torch.optim.Adam(
                    network_parameters,
                    lr=self.network_config["training"]["config"]["optimizer"][
                        "learning_rate"
                    ],
                )

            elif optimizer_type == "sgd":

                assert (
                    "learning_rate"
                    in self.network_config["training"]["config"]["optimizer"]
                ), 'Required key "learning_rate" in dictionary "optimizer" is missing to use the SGD optimizer.'

                self.optimizer = torch.optim.SGD(
                    network_parameters,
                    lr=self.network_config["training"]["config"]["optimizer"][
                        "learning_rate"
                    ],
                )

            else:
                assert False, 'Optimizer "{}" is not defined.'.format(optimizer_type)

        # Enable training mode
        self.model.train()

    def enable_evaluation(self):

        # Enable evaluation mode
        self.model.eval()
