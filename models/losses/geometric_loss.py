# -*- coding: utf-8 -*-


import numpy as np
import cv2

# h_high, w_high = 192, 256
# h_low, w_low = 48, 64

def normalize_3D(points: np.array):
    centroid = np.mean(points)
    points -= centroid
    distances = np.linalg.norm(points, axis=1)
    maximum_extent = np.max(distances)
    points /= maximum_extent
    points.astype(np.float64)
    return points


def calculate_focal(points: np.array, height: int, width: int):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    focal = (0.5 * np.min(z) * min(height, width)) / max(np.max(x), np.max(y))
    return focal.astype(np.float64)


def project_matrix(focal: int, height: int, width: int):
    pm = np.array([
        [focal, 0, height / 2],
        [0, focal, width / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    return pm.astype(np.float64)


def project_3D_to_2D(points: np.array, rvec: np.array, camera_matrix: np.array):
    dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float64)
    tvec = np.array([0, 0, 0], dtype=np.float64)
    
    image_points, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
    return image_points.astype(np.float64)


def calculate_loss(gt_points: np.array, pred_points: np.array, height: int, width: int):
    gt_points = np.rint(gt_points)
    pred_points = np.rint(pred_points)

    gt_pixels = np.zeros((height, width))
    pred_pixels = np.zeros((height, width))

    for h in range(-(height // 2), (height // 2)):
        for w in range(-(width // 2), (width // 2)):
            if [h, w] in gt_points:
                gt_pixels[h, w] = 1
            elif [h, w] in pred_points:
                pred_pixels[h, w] = 1

    loss = np.mean(np.square(gt_pixels - pred_pixels))
    return loss


def geometric_loss(gt_points: np.array, pred_points: np.array, height: int, width: int):
    total_loss = 0
    for i in range(len(gt_points)):
        for views in range(3):
            rvec = np.random.randint(low=0, high=2 * 3.14, size=(3,)).astype(np.float64)

            gt_normal = normalize_3D(gt_points[i])
            gt_focal = calculate_focal(gt_normal, height, width)
            gt_pm = project_matrix(gt_focal, height, width)
            gt_projection = project_3D_to_2D(gt_normal, rvec, gt_pm)

            pred_normal = normalize_3D(pred_points[i])
            pred_focal = calculate_focal(pred_normal, height, width)
            pred_pm = project_matrix(pred_focal, height, width)
            pred_projection = project_3D_to_2D(pred_normal, rvec, pred_pm)

            loss = calculate_loss(gt_projection, pred_projection, height, width)
            total_loss += loss

    return total_loss / len(gt_points)