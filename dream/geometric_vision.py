# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import cv2
import numpy as np
from pyrr import Quaternion

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def convert_rvec_to_quaternion(rvec):
    """Convert rvec (which is log quaternion) to quaternion"""
    theta = np.sqrt(
        rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]
    )  # in radians
    raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

    # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html
    quaternion = Quaternion.from_axis_rotation(raxis, theta)
    quaternion.normalize()
    return quaternion


def hnormalized(vector):
    hnormalized_vector = (vector / vector[-1])[:-1]
    return hnormalized_vector


def point_projection_from_3d(camera_K, points):
    projections = []
    for p in points:
        p_unflattened = np.matmul(camera_K, p)
        projection = hnormalized(p_unflattened)
        projections.append(projection)
    projections = np.array(projections)
    return projections


def solve_pnp(
    canonical_points,
    projections,
    camera_K,
    method=cv2.SOLVEPNP_EPNP,
    refinement=True,
    dist_coeffs=np.array([]),
):

    n_canonial_points = len(canonical_points)
    n_projections = len(projections)
    assert (
        n_canonial_points == n_projections
    ), "Expected canonical_points and projections to have the same length, but they are length {} and {}.".format(
        n_canonial_points, n_projections
    )

    # Process points to remove any NaNs
    canonical_points_proc = []
    projections_proc = []
    for canon_pt, proj in zip(canonical_points, projections):

        if (
            canon_pt is None
            or len(canon_pt) == 0
            or canon_pt[0] is None
            or canon_pt[1] is None
            or proj is None
            or len(proj) == 0
            or proj[0] is None
            or proj[1] is None
        ):
            continue

        canonical_points_proc.append(canon_pt)
        projections_proc.append(proj)

    # Return if no valid points
    if len(canonical_points_proc) == 0:
        return False, None, None

    canonical_points_proc = np.array(canonical_points_proc)
    projections_proc = np.array(projections_proc)

    # Use cv2's PNP solver
    try:
        pnp_retval, rvec, tvec = cv2.solvePnP(
            canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
            projections_proc.reshape(projections_proc.shape[0], 1, 2),
            camera_K,
            dist_coeffs,
            flags=method,
        )

        if refinement:
            pnp_retval, rvec, tvec = cv2.solvePnP(
                canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
                projections_proc.reshape(projections_proc.shape[0], 1, 2),
                camera_K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=True,
                rvec=rvec,
                tvec=tvec,
            )
        translation = tvec[:, 0]
        quaternion = convert_rvec_to_quaternion(rvec[:, 0])

    except:
        pnp_retval = False
        translation = None
        quaternion = None

    return pnp_retval, translation, quaternion


def solve_pnp_ransac(
    canonical_points,
    projections,
    camera_K,
    method=cv2.SOLVEPNP_EPNP,
    inlier_thresh_px=5.0,  # this is the threshold for each point to be considered an inlier
    dist_coeffs=np.array([]),
):

    n_canonial_points = len(canonical_points)
    n_projections = len(projections)
    assert (
        n_canonial_points == n_projections
    ), "Expected canonical_points and projections to have the same length, but they are length {} and {}.".format(
        n_canonial_points, n_projections
    )

    # Process points to remove any NaNs
    canonical_points_proc = []
    projections_proc = []
    for canon_pt, proj in zip(canonical_points, projections):

        if (
            canon_pt is None
            or len(canon_pt) == 0
            or canon_pt[0] is None
            or canon_pt[1] is None
            or proj is None
            or len(proj) == 0
            or proj[0] is None
            or proj[1] is None
        ):
            continue

        canonical_points_proc.append(canon_pt)
        projections_proc.append(proj)

    # Return if no valid points
    if len(canonical_points_proc) == 0:
        return False, None, None, None

    canonical_points_proc = np.array(canonical_points_proc)
    projections_proc = np.array(projections_proc)

    # Use cv2's PNP solver
    try:
        pnp_retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
            projections_proc.reshape(projections_proc.shape[0], 1, 2),
            camera_K,
            dist_coeffs,
            reprojectionError=inlier_thresh_px,
            flags=method,
        )

        translation = tvec[:, 0]
        quaternion = convert_rvec_to_quaternion(rvec[:, 0])

    except:
        pnp_retval = False
        translation = None
        quaternion = None
        inliers = None

    return pnp_retval, translation, quaternion, inliers


def add_from_pose(translation, quaternion, keypoint_positions_wrt_cam_gt, camera_K):
    transform = np.eye(4)
    transform[:3, :3] = quaternion.matrix33.tolist()
    transform[:3, -1] = translation
    kp_pos_gt_homog = np.hstack(
        (
            keypoint_positions_wrt_cam_gt,
            np.ones((keypoint_positions_wrt_cam_gt.shape[0], 1)),
        )
    )
    kp_pos_aligned = np.transpose(np.matmul(transform, np.transpose(kp_pos_gt_homog)))[
        :, :3
    ]
    # The below lines were useful when debugging pnp ransac, so left here for now
    # projs = point_projection_from_3d(camera_K, kp_pos_aligned)
    # temp = np.linalg.norm(kp_projs_est_pnp - projs, axis=1) # all of these should be below the inlier threshold above!
    kp_3d_errors = kp_pos_aligned - keypoint_positions_wrt_cam_gt
    kp_3d_l2_errors = np.linalg.norm(kp_3d_errors, axis=1)
    add = np.mean(kp_3d_l2_errors)
    return add
