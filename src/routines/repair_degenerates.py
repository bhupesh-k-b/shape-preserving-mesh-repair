"""
Module for removing degenerate and duplicate faces from a mesh.
"""

import numpy as np
from typing import Tuple

from src.core.repair_config import RepairConfig, RepairLog

def repair_degenerate_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    config: RepairConfig,
    log: RepairLog
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove zero-area and repeated-vertex faces. Returns (faces, normals)."""
    if len(faces) == 0:
        return faces, normals

    repeated = (
        (faces[:, 0] == faces[:, 1]) |
        (faces[:, 1] == faces[:, 2]) |
        (faces[:, 0] == faces[:, 2])
    )
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = np.linalg.norm(cross, axis=1) * 0.5
    keep_mask = (~repeated) & (area > config.area_tolerance)
    removed = int(len(faces) - np.count_nonzero(keep_mask))
    log.degenerate_removed += removed
    return faces[keep_mask], normals[keep_mask]


def repair_duplicate_faces(
    faces: np.ndarray,
    normals: np.ndarray,
    log: RepairLog
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only the first occurrence of each unique face (vertex-set)."""
    if len(faces) == 0:
        return faces, normals

    canonical = np.sort(faces, axis=1)
    _, keep_idx = np.unique(canonical, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)
    removed = int(len(faces) - len(keep_idx))
    log.duplicate_removed += removed
    return faces[keep_idx], normals[keep_idx]
