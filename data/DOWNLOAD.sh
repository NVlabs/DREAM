#!/bin/bash

# This script makes it easy to download real/synthetic images with annotations.
# (Files are at https://drive.google.com/open?id=1Krp-fCT9ffEML3IpweSOgWiMHHBw6k2Z)

# Comment out any data you do not want.

echo 'Warning:  Files are *very* large.  Be sure to comment out any files you do not want.'


#----- real test data -----------------------------------

mkdir -p real
cd real

gdown --id 10Tpx8jAfzP6g44WXfvjlVywbIlxZ4BRx  # Panda-3Cam, Azure (368 MB)
gdown --id 14TJ9o9QOdb25zlZ3onsOJlSb7-tGrvKz  # Panda-3Cam, Kinect360 (295 MB)
gdown --id 1FFAFpJFwzsjD83S9-Y1ODwDWiWlh1X6P  # Panda-3Cam, RealSense (343 MB)
gdown --id 1kL7Goibx4lwKQoO-UQ4gm94f_XdEKTUZ  # Panda-Orb, RealSense (2 GB)

cd ..


#----- synthetic training / validation data -----------------------------------

mkdir -p synthetic
cd synthetic

#----- Rethink Robotics' Baxter
gdown --id 1MSRwQpg690RvuvtjNuGYA1ILGipX16dW  # test DR (541 MB)
gdown --id 1SzUPYmNxe1OsbGyWdpdkoRjWJurs-NAF  # train (9 GB)

#----- Kuka LBR iiwa 7 R800
gdown --id 1kGvSlVScmMohZStS-_NfCpCa5SBAcx_i  # test DR (445 MB)
gdown --id 1ChF4jAGMPbPwe2dOZYPJ2t2rCSR0Xw9R  # test non-DR (222 MB)
gdown --id 1HTW3YEGDO22zOT56jFWxfizznw4aGMpU  # train (8 GB)

#----- Franka Emika Panda
gdown --id 12bvDmr6cZZaCNadOWf_4EOHI9sp97yah  # test DR (450 MB)
gdown --id 11pK1BqfQkzVnTjyQHVRZ6ZkX4oyxbEQP  # test non-DR (190 MB)
gdown --id 1ZXzseMa7aMIKxK4BNH2gacmm3_XGJvxm  # train (8 GB)

cd ..
