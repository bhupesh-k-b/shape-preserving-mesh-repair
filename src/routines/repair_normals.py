"""
Module for fixing inverted normals and ensuring consistent face winding.
"""

import numpy as np
from collections import deque
from typing import List, Tuple

from src.core.repair_config import RepairLog
from src.core.repair_utils import _signed_volume
from src.core.mesh_analyzer import build_edge_map


def improved_orientation(
    vertices: np.ndarray,
    faces: np.ndarray,
    log: RepairLog
) -> np.ndarray:
    if len(faces) == 0:
        return faces

    faces = faces.copy()
    edge_map = build_edge_map(faces)
    # The adjacency is built once from the original edge map and is intentionally
    # not rebuilt after mid-BFS flips, because the direction check uses the live
    # faces array directly.
    adjacency: List[List[Tuple[int, Tuple[int, int]]]] = [[] for _ in range(len(faces))]
    for edge, face_ids in edge_map.items():
        if len(face_ids) != 2:
            continue
        a, b = int(face_ids[0]), int(face_ids[1])
        adjacency[a].append((b, edge))
        adjacency[b].append((a, edge))

    def edge_direction(face: np.ndarray, edge: Tuple[int, int]) -> int:
        u, v = edge
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            if a == u and b == v:
                return 1
            if a == v and b == u:
                return -1
        return 0

    visited = np.zeros(len(faces), dtype=bool)
    flipped = 0
    for start in range(len(faces)):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        component: List[int] = []
        while queue:
            fi = queue.popleft()
            component.append(fi)
            for nb, edge in adjacency[fi]:
                if visited[nb]:
                    continue
                dir_f = edge_direction(faces[fi], edge)
                dir_nb = edge_direction(faces[nb], edge)
                if dir_f != 0 and dir_f == dir_nb:
                    faces[nb, [1, 2]] = faces[nb, [2, 1]]
                    flipped += 1
                visited[nb] = True
                queue.append(nb)

        comp_idx = np.array(component, dtype=np.int32)
        if len(comp_idx) and _signed_volume(vertices, faces[comp_idx]) < 0:
            faces[comp_idx] = faces[comp_idx][:, [0, 2, 1]]
            flipped += len(comp_idx)

    log.faces_reoriented += flipped
    return faces


def repair_inverted_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    log: RepairLog
) -> np.ndarray:
    return improved_orientation(vertices, faces, log)
