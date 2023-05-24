# -*- coding: utf-8 -*-
import numpy as np
import torch
import cv2


def normalize_3D(points: torch.tensor):
    centroid = torch.mean(points)
    points = points - centroid
    distances = torch.linalg.norm(points, dim=1)
    maximum_extent = torch.max(distances)
    points = points / maximum_extent
    return points


def calculate_focal(points: torch.tensor, height: int, width: int):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    focal = (0.5 * torch.min(z) * min(height, width)) / max(torch.max(x), torch.max(y))
    return focal.item()


def project_matrix(focal: int, height: int, width: int):
    pm = torch.tensor([
        [focal, 0, height / 2],
        [0, focal, width / 2],
        [0, 0, 1]
    ], dtype=torch.float32)
    return pm


def project_3D_to_2D(points: torch.tensor, rvec: torch.tensor, camera_matrix: np.array):
    points = points.detach().cpu().numpy()
    camera_matrix = camera_matrix.detach().cpu().numpy()

    dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)

    # rvec = np.array([0, 0, 0], dtype=np.float32)
    tvec = np.array([0, 0, 0], dtype=np.float32)

    image_points, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
    return torch.from_numpy(image_points)


def find_pixels(points: torch.tensor, height: int, width: int):
    pixels = torch.zeros((height, width))

    x_min = -height // 2
    x_max = height // 2
    y_min = -width // 2
    y_max = width // 2

    mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (
            points[:, 1] <= y_max)
    x_indices = (points[mask, 0] + x_max - 1).to(torch.long)
    y_indices = (points[mask, 1] + y_max - 1).to(torch.long)
    pixels[x_indices, y_indices] = 1

    return pixels


def calculate_loss(gt_points: torch.tensor, pred_points: torch.tensor, height: int, width: int):
    gt_points = torch.round(gt_points)
    gt_points = gt_points.reshape(-1, 2)

    pred_points = torch.round(pred_points)
    pred_points = pred_points.reshape(-1, 2)

    gt_pixels = find_pixels(gt_points, height, width)
    pred_pixels = find_pixels(pred_points, height, width)

    loss = torch.mean(torch.square(gt_pixels - pred_pixels))
    return loss


def geometric_loss(gt_points: torch.tensor, pred_points: torch.tensor, height: int, width: int):
    total_loss = 0

    for replication in range(len(gt_points)):
        for views in range(3):
            rvec = np.random.randint(low=0, high=2 * 3.14, size=(3,)).astype(np.float64)

            # print(f"rotation vector is: ", rvec)

            gt_normal = normalize_3D(gt_points[replication])
            gt_focal = calculate_focal(gt_normal, height, width)
            gt_pm = project_matrix(gt_focal, height, width)
            gt_projection = project_3D_to_2D(gt_normal, rvec, gt_pm)

            pred_normal = normalize_3D(pred_points[replication])
            pred_focal = calculate_focal(pred_normal, height, width)
            pred_pm = project_matrix(pred_focal, height, width)
            pred_projection = project_3D_to_2D(pred_normal, rvec, pred_pm)

            loss = calculate_loss(gt_projection, pred_projection, height, width)
            total_loss += loss

        return total_loss.item()