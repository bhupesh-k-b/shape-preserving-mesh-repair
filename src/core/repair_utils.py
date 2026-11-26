"""
Core utility functions for the Mesh Repair Pipeline.
"""

import numpy as np
from typing import Tuple, Dict

def _compact_mesh(vertices: np.ndarray,
                  faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return vertices[:0].copy(), faces.copy()
    used = np.unique(faces.reshape(-1))
    remap = -np.ones(len(vertices), dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return vertices[used].copy(), remap[faces]

def _face_area(vertices: np.ndarray, face: np.ndarray) -> float:
    a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))

def _face_normal(vertices: np.ndarray, face: np.ndarray) -> np.ndarray:
    a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    n = np.cross(b - a, c - a)
    ln = float(np.linalg.norm(n))
    return n / ln if ln > 0 else np.array([0., 0., 1.])

def _recompute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Vectorised normal computation — ~100x faster than per-face loop."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return cross / norms

def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Signed volume via the divergence theorem.
    Negative → normals point inward (mesh is inside-out).
    """
    if len(faces)==0:
        return 0.0
    # Translation-invariant origin avoids cancellation for distant models.
    origin=vertices[faces[0,0]]
    v0 = vertices[faces[:, 0]]-origin
    v1 = vertices[faces[:, 1]]-origin
    v2 = vertices[faces[:, 2]]-origin
    # Signed volume of tetrahedra formed with the origin
    return float(np.sum(v0 * np.cross(v1, v2))) / 6.0

def compute_edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute lengths of all 3 edges for every face (3*F total)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    
    return np.concatenate([e0, e1, e2])
