#!/bin/bash

# This script makes it easy to download pretrained model weights.
# (Files are at https://drive.google.com/open?id=1Krp-fCT9ffEML3IpweSOgWiMHHBw6k2Z)

# Each robot contains two files (.pth and .yaml).
# Comment out any model you do not want.

#----- Rethink Robotics' Baxter 

# Baxter VGG-Q (baxter_dream_vgg_q)
gdown --id 1Ia4UxSdilXH9SwyPqem0rS13Mha9pN7F  # .pth (85 MB)
gdown --id 1TNhYuOm_-UH5z1rEVm16mnRA7hB7AT1X  # .yaml


#----- Kuka LBR iiwa 7 R800

# Kuka ResNet-H (kuka_dream_resnet_h)
gdown --id 1Ctoh01q1IvLHP9pf5Os8eIzJ8fQBgYpJ  # .pth (207 MB)
gdown --id 1MLWDTq7yQF9UeV1T3REDk60GYne32OXJ  # .yaml


#----- Franka Emika Panda

# Panda VGG-Q (panda_dream_vgg_q) -- recommended
gdown --id 1zS-kQ73dOYMXS8Wku_OUN0q7MvEUm2fZ  # .pth (85 MB)
gdown --id 1MKDiknxDzXErd4Gwdv0uMoL65IYjxO0Q  # .yaml

# Panda VGG-F (panda_dream_vgg_f)
gdown --id 1pz-gXux8TxB4pOYnYy5DH7vp-3-mTJFu  # .pth (86 MB)
gdown --id 191Pgu_C0qzKpOSoicOOSLq-bR7cg2KVO  # .yaml

# Panda ResNet-H (panda_dream_resnet_h)
gdown --id 16fyv6ps3om0H8dnXRDHj0w4dfEKPSpDW  # .pth (207 MB)
gdown --id 1gCpigRIqm1rAw-o7oXpRO2ZTQkHyYF-k  # .yaml

# Panda ResNet-F (panda_dream_resnet_f) -- difficult to train
gdown --id 1d8UfrgQb4ohIAfpRGvDBjabSKuP9LCpy  # .pth (211 MB)
gdown --id 1IWdXSmmIq2-eimtNK_ywJZRH4omesSDq  # .yaml
