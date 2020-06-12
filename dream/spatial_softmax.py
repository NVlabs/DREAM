# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

# This code is adapted from
# https://gitlab-master.nvidia.com/pmolchanov/lpr-3d-hand-pose-rgb-demo/blob/master/handpose/models/image_heatmaps_pose2dZrel_softargmax_slim.py


class SoftArgmaxPavlo(torch.nn.Module):
    def __init__(self, n_keypoints=5, learned_beta=False, initial_beta=25.0):
        super(SoftArgmaxPavlo, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(7, stride=1, padding=3)
        if learned_beta:
            self.beta = torch.nn.Parameter(torch.ones(n_keypoints) * initial_beta)
        else:
            self.beta = (torch.ones(n_keypoints) * initial_beta).cuda()

    def forward(self, heatmaps, size_mult=1.0):

        epsilon = 1e-8
        bch, ch, n_row, n_col = heatmaps.size()
        n_kpts = ch

        beta = self.beta

        # input has the shape (#bch, n_kpts+1, img_sz[0], img_sz[1])
        # +1 is for the Zrel
        heatmaps2d = heatmaps[:, :n_kpts, :, :]
        heatmaps2d = self.avgpool(heatmaps2d)

        # heatmaps2d has the shape (#bch, n_kpts, img_sz[0]*img_sz[1])
        heatmaps2d = heatmaps2d.contiguous().view(bch, n_kpts, -1)

        # getting the max value of the maps across each 2D matrix
        map_max = torch.max(heatmaps2d, dim=2, keepdim=True)[0]

        # reducing the max from each map
        # this will make the max value zero and all other values
        # will be negative.
        # max_reduced_maps has the shape (#bch, n_kpts, img_sz[0]*img_sz[1])
        heatmaps2d = heatmaps2d - map_max

        beta_ = beta.view(1, n_kpts, 1).repeat(1, 1, n_row * n_col)
        # due to applying the beta value, the non-max values will be further
        # pushed towards zero after applying the exp function
        exp_maps = torch.exp(beta_ * heatmaps2d)
        # normalizing the exp_maps by diving it to the sum of elements
        # exp_maps_sum has the shape (#bch, n_kpts, 1)
        exp_maps_sum = torch.sum(exp_maps, dim=2, keepdim=True)
        exp_maps_sum = exp_maps_sum.view(bch, n_kpts, 1, 1)
        normalized_maps = exp_maps.view(bch, n_kpts, n_row, n_col) / (
            exp_maps_sum + epsilon
        )

        col_vals = torch.arange(0, n_col) * size_mult
        col_repeat = col_vals.repeat(n_row, 1)
        col_idx = col_repeat.view(1, 1, n_row, n_col).cuda()
        # col_mat gives a column measurement matrix to be used for getting
        # 'x'. It is a matrix where each row has the sequential values starting
        # from 0 up to n_col-1:
        # 0,1,2, ..., n_col-1
        # 0,1,2, ..., n_col-1
        # 0,1,2, ..., n_col-1

        row_vals = torch.arange(0, n_row).view(n_row, -1) * size_mult
        row_repeat = row_vals.repeat(1, n_col)
        row_idx = row_repeat.view(1, 1, n_row, n_col).cuda()
        # row_mat gives a row measurement matrix to be used for getting 'y'.
        # It is a matrix where each column has the sequential values starting
        # from 0 up to n_row-1:
        # 0,0,0, ..., 0
        # 1,1,1, ..., 1
        # 2,2,2, ..., 2
        # ...
        # n_row-1, ..., n_row-1

        col_idx = Variable(col_idx, requires_grad=False)
        weighted_x = normalized_maps * col_idx.float()
        weighted_x = weighted_x.view(bch, n_kpts, -1)
        x_vals = torch.sum(weighted_x, dim=2)

        row_idx = Variable(row_idx, requires_grad=False)
        weighted_y = normalized_maps * row_idx.float()
        weighted_y = weighted_y.view(bch, n_kpts, -1)
        y_vals = torch.sum(weighted_y, dim=2)

        out = torch.stack((x_vals, y_vals), dim=2)

        return out


if __name__ == "__main__":
    from torch.autograd import Variable

    data = Variable(torch.zeros([1, 3, 3, 3]))
    data[0, 0, 0, 1] = 10
    data[0, 1, 1, 1] = 10
    data[0, 2, 1, 2] = 10
    layer = SpatialSoftmax(3, 3, 3, temperature=1)
    print(layer(data))
