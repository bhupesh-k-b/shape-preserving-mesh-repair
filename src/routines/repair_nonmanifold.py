"""
Module for resolving non-manifold edges.
"""

import numpy as np
from typing import Tuple, Set, Dict

from src.core.repair_config import RepairConfig, RepairLog
from src.core.repair_utils import _face_normal
from src.core.mesh_analyzer import build_edge_map


def _dihedral_score(vertices: np.ndarray, face: np.ndarray,
                    edge_map: Dict, all_faces: np.ndarray) -> float:
    """
    Score a face by the average dihedral angle with its neighbours.
    Lower score = flatter join = prefer keeping this face.
    """
    scores = []
    fn = _face_normal(vertices, face)
    for j in range(3):
        a, b = int(face[j]), int(face[(j + 1) % 3])
        key = (min(a, b), max(a, b))
        for nb_fi in edge_map.get(key, []):
            nb_face = all_faces[nb_fi]
            if np.array_equal(nb_face, face):
                continue
            nn = _face_normal(vertices, nb_face)
            cos_a = float(np.clip(np.dot(fn, nn), -1, 1))
            scores.append(cos_a)
    return float(np.mean(scores)) if scores else 0.0


def repair_non_manifold(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    config: RepairConfig,
    log: RepairLog
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each edge with 3+ faces, keep the `max_faces_per_edge` faces that
    form the flattest (most co-planar) dihedral angles.  
    The rest are removed.
    """
    edge_map = build_edge_map(faces)

    # Find which faces must be removed
    faces_to_remove: Set[int] = set()
    for edge, face_list in edge_map.items():
        if len(face_list) <= config.max_faces_per_edge:
            continue
        # Score each face candidate
        scored = sorted(
            face_list,
            key=lambda fi: _dihedral_score(vertices, faces[fi], edge_map, faces),
            reverse=True,            # higher cos = flatter = better
        )
        # Remove all but the top N
        for fi in scored[config.max_faces_per_edge:]:
            faces_to_remove.add(fi)

    config.check_deadline()
    if faces_to_remove and len(faces_to_remove) < len(faces):
        mask=np.ones(len(faces),dtype=bool)
        mask[list(faces_to_remove)]=False
        faces=faces[mask]; normals=normals[mask]
        log.non_manifold_removed += len(faces_to_remove)
    # Boundary triangles are modeled geometry, not automatically removable fins.
    return faces, normals
