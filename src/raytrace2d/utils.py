"""
Utilities for raytrace2d project.
"""
import numpy as np
from typing import Tuple

def douglas_peucker(depths: np.ndarray, speeds: np.ndarray, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplify sound speed profile using the Douglas-Peucker algorithm.
    Args:
        depths: 1D array of depth values.
        speeds: 1D array of corresponding sound speeds.
        epsilon: threshold T for maximum allowed deviation.
    Returns:
        Tuple of (simplified_depths, simplified_speeds).
    """
    if depths.shape != speeds.shape:
        raise ValueError("depths and speeds must have the same shape")
    # Stack into Nx2 points
    points = np.vstack((depths, speeds)).T

    def _perp_distance(pt, line_start, line_end):
        # Compute perpendicular distance from pt to line (line_start, line_end)
        if np.all(line_start == line_end):
            return np.linalg.norm(pt - line_start)
        return np.abs(np.cross(line_end - line_start, line_start - pt) / 
                      np.linalg.norm(line_end - line_start))

    def _recursive_dp(pts):
        # Base case: if only two points, return them
        if pts.shape[0] <= 2:
            return pts
        # Find point with max distance
        start, end = pts[0], pts[-1]
        distances = np.array([_perp_distance(p, start, end) for p in pts[1:-1]])
        idx = np.argmax(distances)
        dmax = distances[idx] if distances.size > 0 else 0.0
        if dmax > epsilon:
            # Recursively simplify
            idx += 1  # account for offset
            left = _recursive_dp(pts[:idx+1])
            right = _recursive_dp(pts[idx:])
            # Combine, avoid duplicate
            return np.vstack((left[:-1], right))
        else:
            # Discard intermediate points
            return np.vstack((start, end))

    simplified = _recursive_dp(points)
    return simplified[:,0], simplified[:,1]
