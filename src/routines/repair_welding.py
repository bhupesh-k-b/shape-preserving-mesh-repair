"""
Module for global vertex spatial welding and boundary sliver welding.
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Set

from src.core.repair_config import RepairConfig, RepairLog
from src.core.repair_utils import _compact_mesh, _recompute_normals
from src.routines.repair_degenerates import repair_degenerate_faces, repair_duplicate_faces
from src.core.mesh_analyzer import build_edge_map, check_naked_and_nonmanifold

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:  # pragma: no cover
    KDTree = None


def _spatial_weld_groups(points: np.ndarray,
                         tolerance: float) -> List[List[int]]:
    if len(points) == 0 or tolerance <= 0:
        return []

    parent = np.arange(len(points), dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra > rb:
            ra, rb = rb, ra
        parent[rb] = ra

    tol_sq = float(tolerance) * float(tolerance)

    if KDTree is not None and len(points) > 1:
        tree = KDTree(points)
        for i, j in tree.query_pairs(tolerance, output_type="ndarray"):
            union(int(i), int(j))
    else:
        cell = max(float(tolerance), 1e-15)
        buckets: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        keys = np.floor(points / cell).astype(np.int64)
        for i, key in enumerate(keys):
            buckets[tuple(int(v) for v in key)].append(i)
        offsets = [(dx, dy, dz)
                   for dx in (-1, 0, 1)
                   for dy in (-1, 0, 1)
                   for dz in (-1, 0, 1)]
        for i, key in enumerate(keys):
            key_t = tuple(int(v) for v in key)
            for off in offsets:
                nb_key = (key_t[0] + off[0], key_t[1] + off[1], key_t[2] + off[2])
                for j in buckets.get(nb_key, []):
                    if j > i and float(np.dot(points[i] - points[j], points[i] - points[j])) <= tol_sq:
                        union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(points)):
        groups[find(i)].append(i)
    return list(groups.values())


def global_vertex_weld(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    config: RepairConfig,
    log: RepairLog,
    boundary_only: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Global welding fixes tiny open seams that boundary-only welding misses.
    It safely remaps faces through a union-find, then removes any triangles
    collapsed by the merge so indices cannot become corrupt.
    """
    if len(vertices) == 0 or len(faces) == 0:
        return vertices, faces, normals

    vertices = vertices.copy()
    tolerance = (config.boundary_weld_tolerance if boundary_only
                 else config.merge_tolerance)
    if tolerance <= 0:
        return vertices, faces, normals

    if boundary_only:
        edge_map = build_edge_map(faces)
        naked, _ = check_naked_and_nonmanifold(edge_map)
        candidate_vertices = np.array(sorted({v for e in naked for v in e}),
                                      dtype=np.int32)
        if len(candidate_vertices) < 2:
            return vertices, faces, normals
    else:
        candidate_vertices = np.arange(len(vertices), dtype=np.int32)

    groups = _spatial_weld_groups(vertices[candidate_vertices], tolerance)
    if not groups:
        return vertices, faces, normals

    # Build incident-face sets only for vertices that may actually be welded.
    candidate_mask = np.zeros(len(vertices), dtype=bool)
    candidate_mask[candidate_vertices] = True
    incident_faces: Dict[int, Set[int]] = defaultdict(set)
    for fi, face in enumerate(faces):
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        if candidate_mask[a]:
            incident_faces[a].add(fi)
        if candidate_mask[b]:
            incident_faces[b].add(fi)
        if candidate_mask[c]:
            incident_faces[c].add(fi)

    remap = np.arange(len(vertices), dtype=np.int32)
    moved = 0
    tol_sq = float(tolerance) * float(tolerance)

    v_to_group = np.arange(len(vertices), dtype=np.int32)
    for group in groups:
        if len(group) < 2:
            continue
        g_id = int(min((int(v) for v in candidate_vertices[np.array(group, dtype=np.int32)])))
        for idx in group:
            v_to_group[candidate_vertices[idx]] = g_id

    def cluster_is_compatible(cluster: List[int], vertex_idx: int) -> bool:
        v_pos = vertices[vertex_idx]
        v_faces = incident_faces.get(vertex_idx, set())
        for other in cluster:
            delta = vertices[other] - v_pos
            if float(np.dot(delta, delta)) > tol_sq:
                return False
            if incident_faces.get(other, set()) & v_faces:
                return False

        # Topology Check: prevent merge if it would create a non-manifold edge.
        # We must map adjacent vertices to their *intended* global spatial group
        # so that separated neighbors are recognized as being effectively coincidental.
        v_adj_faces = defaultdict(int)
        for fi in v_faces:
            for v in faces[fi]:
                if v != vertex_idx:
                    v_adj_faces[v_to_group[int(v)]] += 1
                    
        for other in cluster:
            for fi in incident_faces.get(other, set()):
                for v in faces[fi]:
                    if v != other:
                        v_adj_faces[v_to_group[int(v)]] += 1
                        
        for vg, count in v_adj_faces.items():
            if count > 2:
                # Count how many of these faces came from vertex_idx
                count_v = 0
                for fi in v_faces:
                    for v in faces[fi]:
                        if v != vertex_idx and v_to_group[int(v)] == vg:
                            count_v += 1
                            
                # If both sides contributed faces to this edge, the merge CAUSES non-manifoldness
                if count_v > 0 and (count - count_v) > 0:
                    return False

        return True

    for group in groups:
        if len(group) < 2:
            continue

        originals = sorted(
            (int(v) for v in candidate_vertices[np.array(group, dtype=np.int32)]),
            key=int
        )

        # Split coarse groups into safe cliques
        safe_clusters: List[List[int]] = []
        for vertex_idx in originals:
            placed = False
            for cluster in safe_clusters:
                if cluster_is_compatible(cluster, vertex_idx):
                    cluster.append(vertex_idx)
                    placed = True
                    break
            if not placed:
                safe_clusters.append([vertex_idx])

        for cluster in safe_clusters:
            if len(cluster) < 2:
                continue
            canonical = int(min(cluster))
            merged_position = vertices[np.array(cluster, dtype=np.int32)].mean(axis=0)
            vertices[canonical] = merged_position
            for old in cluster:
                if old != canonical:
                    remap[old] = canonical
                    moved += 1

    if moved == 0:
        return vertices, faces, normals

    faces = remap[faces]
    log.vertices_welded += moved

    deduped_vertices, deduped_faces = _compact_mesh(vertices, faces)
    normals = _recompute_normals(deduped_vertices, deduped_faces) if len(deduped_faces) else normals[:0]
    
    dedupe_log = RepairLog()
    deduped_faces, normals = repair_degenerate_faces(
        deduped_vertices, deduped_faces, normals, config, dedupe_log)
    log.degenerate_removed += dedupe_log.degenerate_removed

    dup_log = RepairLog()
    deduped_faces, normals = repair_duplicate_faces(deduped_faces, normals, dup_log)
    log.duplicate_removed += dup_log.duplicate_removed
    
    return deduped_vertices, deduped_faces, normals


def repair_boundary_slivers(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    config: RepairConfig,
    log: RepairLog
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return global_vertex_weld(
        vertices, faces, normals, config, log, boundary_only=True
    )
