"""
================================================================================
  MESH REPAIR PIPELINE  —  Phase 2: Repairer
  Repairs: Degenerate Faces, Duplicate Faces, Non-Manifold Edges,
           Inverted Normals, Planar Holes (ear-clip), Non-Planar Holes (fan)
  Zero paid dependencies — pure numpy.
================================================================================
"""

import numpy as np
import struct
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
from src.core.mesh_analyzer import (
    MeshData, MeshAnalyzer, AnalysisResult,
    build_edge_map, check_naked_and_nonmanifold, classify_holes
)

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:  # pragma: no cover - scipy is optional
    KDTree = None


# ─────────────────────────────────────────────────────────────────────────────
#  Repair Config & Log
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RepairConfig:
    area_tolerance:       float = 1e-12   # below this → degenerate face
    merge_tolerance:      float = 5e-5    # vertex merge distance
    planar_tolerance:     float = 1e-4    # SVD flatness ratio
    normal_angle_tol_deg: float = 90.0    # stored-vs-computed flip threshold
    # Non-manifold: how many faces to keep per edge (usually 2)
    max_faces_per_edge:   int   = 2
    # Boundary sliver welding: merge naked-edge vertices closer than this
    boundary_weld_tolerance: float = 0.1    


@dataclass
class RepairLog:
    degenerate_removed:   int = 0
    duplicate_removed:    int = 0
    non_manifold_removed: int = 0
    faces_reoriented:     int = 0
    vertices_welded:      int = 0
    planar_holes_filled:  int = 0
    non_planar_filled:    int = 0
    faces_added:          int = 0
    repair_time_ms:       float = 0.0

    def summary(self) -> str:
        lines = [
            "━" * 60,
            "  REPAIR LOG",
            "━" * 60,
            f"  Degenerate faces removed   : {self.degenerate_removed}",
            f"  Duplicate faces removed    : {self.duplicate_removed}",
            f"  Non-manifold extra removed : {self.non_manifold_removed}",
            f"  Faces reoriented (normals) : {self.faces_reoriented}",
            f"  Boundary vertices welded   : {self.vertices_welded}",
            f"  Planar holes filled        : {self.planar_holes_filled}",
            f"  Non-planar holes filled    : {self.non_planar_filled}",
            f"  New faces added (patches)  : {self.faces_added}",
            f"  Repair time                : {self.repair_time_ms:.1f} ms",
            "━" * 60,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compact_mesh(vertices: np.ndarray,
                  faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return vertices[:0].copy(), faces.copy()
    used = np.unique(faces.reshape(-1))
    remap = -np.ones(len(vertices), dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return vertices[used].copy(), remap[faces]


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

    def cluster_is_compatible(cluster: List[int], vertex_idx: int) -> bool:
        v_pos = vertices[vertex_idx]
        v_faces = incident_faces.get(vertex_idx, set())
        for other in cluster:
            delta = vertices[other] - v_pos
            if float(np.dot(delta, delta)) > tol_sq:
                return False
            if incident_faces.get(other, set()) & v_faces:
                return False
        return True

    for group in groups:
        if len(group) < 2:
            continue

        originals = sorted(
            (int(v) for v in candidate_vertices[np.array(group, dtype=np.int32)]),
            key=int
        )

        # The spatial grouping can be transitive (A close to B, B close to C)
        # even when A and C are too far apart. Split each coarse group into
        # safe cliques so welding stays within the actual tolerance ball.
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
    return global_vertex_weld(vertices, faces, normals, config, log)

def _face_area(vertices: np.ndarray, face: np.ndarray) -> float:
    a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def _face_normal(vertices: np.ndarray, face: np.ndarray) -> np.ndarray:
    a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    n = np.cross(b - a, c - a)
    ln = float(np.linalg.norm(n))
    return n / ln if ln > 1e-12 else np.array([0., 0., 1.])


def _recompute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Vectorised normal computation — ~100x faster than per-face loop."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)   # avoid /0
    return cross / norms


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Signed volume via the divergence theorem.
    Negative → normals point inward (mesh is inside-out).
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    # Signed volume of tetrahedra formed with the origin
    return float(np.sum(v0 * np.cross(v1, v2))) / 6.0


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
            nn = _face_normal(vertices, nb_face)
            cos_a = float(np.clip(np.dot(fn, nn), -1, 1))
            scores.append(cos_a)
    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 1-A: Remove Degenerate Faces
# ─────────────────────────────────────────────────────────────────────────────

def repair_degenerate_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    config: RepairConfig,
    log: RepairLog
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove zero-area and repeated-vertex faces. Returns (faces, normals)."""
    if len(faces) == 0:
        log.degenerate_removed = 0
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
    area2 = np.linalg.norm(cross, axis=1) * 0.5
    keep_mask = (~repeated) & (area2 >= config.area_tolerance)
    removed = int(len(faces) - np.count_nonzero(keep_mask))
    log.degenerate_removed = removed
    return faces[keep_mask], normals[keep_mask]


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 1-B: Remove Duplicate Faces
# ─────────────────────────────────────────────────────────────────────────────

def repair_duplicate_faces(
    faces: np.ndarray,
    normals: np.ndarray,
    log: RepairLog
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only the first occurrence of each unique face (vertex-set)."""
    if len(faces) == 0:
        log.duplicate_removed = 0
        return faces, normals

    canonical = np.sort(faces, axis=1)
    _, keep_idx = np.unique(canonical, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)
    removed = int(len(faces) - len(keep_idx))
    log.duplicate_removed = removed
    return faces[keep_idx], normals[keep_idx]


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 2-A: Resolve Non-Manifold Edges
# ─────────────────────────────────────────────────────────────────────────────

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

    keep = [fi for fi in range(len(faces)) if fi not in faces_to_remove]
    log.non_manifold_removed = len(faces_to_remove)

    if not keep:
        return faces, normals
    mask = np.array(keep, dtype=int)
    return faces[mask], normals[mask]


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 2-B: Fix Inverted Normals via BFS Winding Propagation
#             + Global Signed-Volume Orientation Test
# ─────────────────────────────────────────────────────────────────────────────

def repair_inverted_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    log: RepairLog
) -> np.ndarray:
    """
    Two-stage normal repair:

    Stage 1 — BFS Winding Consistency:
      Walk face-adjacency via canonical edges.  For each shared edge,
      verify the two faces traverse it in opposite directions (manifold rule).
      If both traverse (u→v), the neighbour is flipped.

    Stage 2 — Global Orientation (Signed-Volume Test):
      After BFS, the mesh is *consistently* wound but might be entirely
      inside-out.  Compute the signed volume: if negative, flip every face.
      This matches what PrusaSlicer / Netfabb / Meshmixer do.

    Returns the corrected faces array (normals are recomputed after).
    """
    n = len(faces)
    if n == 0:
        return faces

    faces = faces.copy()

    # ── Build canonical edge map + per-face directed edge sets ─────────
    edge_map = build_edge_map(faces)  # canonical (min,max) → [fi, ...]

    # Per-face set of directed edges: face_directed[fi] = {(u,v), (v,w), (w,u)}
    face_directed: List[Set[Tuple[int, int]]] = []
    for fi, face in enumerate(faces):
        edges = set()
        for j in range(3):
            edges.add((int(face[j]), int(face[(j + 1) % 3])))
        face_directed.append(edges)

    # ── Stage 1: BFS winding consistency ───────────────────────────────
    visited = np.zeros(n, dtype=bool)
    flipped = 0

    for start in range(n):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True

        while queue:
            fi = queue.pop(0)  # BFS (FIFO)
            face = faces[fi]
            for j in range(3):
                u = int(face[j])
                v = int(face[(j + 1) % 3])
                canon = (min(u, v), max(u, v))

                for nb in edge_map.get(canon, []):
                    if nb == fi or visited[nb]:
                        continue

                    # Does the neighbour traverse this edge in the SAME
                    # direction as face fi?  If (u,v) ∈ face_directed[nb]
                    # then both faces go u→v → inconsistent → flip nb.
                    if (u, v) in face_directed[nb]:
                        # Flip nb: swap vertices 1 and 2
                        faces[nb, 1], faces[nb, 2] = (
                            int(faces[nb, 2]), int(faces[nb, 1]))
                        # Rebuild directed-edge set for nb
                        face_directed[nb] = set()
                        for k in range(3):
                            face_directed[nb].add(
                                (int(faces[nb][k]),
                                 int(faces[nb][(k + 1) % 3])))
                        flipped += 1

                    visited[nb] = True
                    queue.append(nb)

    # ── Stage 2: Global orientation via signed volume ─────────────────
    vol = _signed_volume(vertices, faces)
    if vol < 0:
        # Entire mesh is inside-out — flip every face
        faces[:, [1, 2]] = faces[:, [2, 1]]
        flipped = len(faces)  # all were wrong

    log.faces_reoriented = flipped
    return faces


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 3 Helpers: Boundary Loop Ordering
# ─────────────────────────────────────────────────────────────────────────────

def _order_boundary_loop(edges):
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    start = edges[0][0]
    loop = [start]
    prev = None
    cur = start

    while True:
        neighbors = adj[cur]
        next_v = None

        for nb in neighbors:
            if nb != prev:
                next_v = nb
                break

        if next_v is None:
            break

        if next_v == start:
            return loop

        loop.append(next_v)
        prev, cur = cur, next_v

        if len(loop) > len(edges) + 5:
            break

    return loop if len(loop) >= 3 else None


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 3-A: Fill Planar Holes — Ear-Clipping Triangulation
# ─────────────────────────────────────────────────────────────────────────────

def _project_to_2d(pts_3d: np.ndarray, normal: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an orthonormal coordinate frame on the plane defined by `normal`.
    Returns (pts_2d, axis_u, axis_v).
    """
    n   = normal / (np.linalg.norm(normal) + 1e-15)
    ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    u   = np.cross(ref, n);  u /= (np.linalg.norm(u) + 1e-15)
    v   = np.cross(n, u);    v /= (np.linalg.norm(v) + 1e-15)
    center = pts_3d.mean(axis=0)
    pts_2d = np.column_stack([pts_3d.dot(u), pts_3d.dot(v)])
    return pts_2d, u, v


def _cross2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _point_in_triangle_2d(p, a, b, c) -> bool:
    """Return True if point p is strictly inside triangle abc (2D)."""
    d1 = _cross2d(p, a, b)
    d2 = _cross2d(p, b, c)
    d3 = _cross2d(p, c, a)
    has_neg = (d1 < -1e-10) or (d2 < -1e-10) or (d3 < -1e-10)
    has_pos = (d1 >  1e-10) or (d2 >  1e-10) or (d3 >  1e-10)
    return not (has_neg and has_pos)


def _ear_clip(loop_2d: np.ndarray, loop_indices: List[int]
              ) -> List[Tuple[int, int, int]]:
    """
    Ear-clipping triangulation of a simple polygon.
    loop_2d       : (N,2) projected 2D points
    loop_indices  : original 3D vertex indices
    Returns list of (i,j,k) face index triples.
    """
    n      = len(loop_2d)
    tris   = []
    idxs   = list(range(n))   # working index list into loop_2d

    # Ensure CCW orientation (positive area)
    area = sum(
        _cross2d(loop_2d[idxs[i]], loop_2d[idxs[(i+1) % n]], loop_2d[idxs[(i+2) % n]])
        for i in range(len(idxs))
    )
    if area < 0:
        idxs.reverse()

    max_iters = n * n          # safety cap
    iters     = 0

    while len(idxs) > 3 and iters < max_iters:
        iters  += 1
        n_cur   = len(idxs)
        ear_found = False

        for i in range(n_cur):
            prev_i = idxs[(i - 1) % n_cur]
            curr_i = idxs[i]
            next_i = idxs[(i + 1) % n_cur]

            A = loop_2d[prev_i]
            B = loop_2d[curr_i]
            C = loop_2d[next_i]

            # Must be a convex vertex (left turn in CCW polygon)
            cross = _cross2d(A, B, C)
            if cross <= 1e-10:
                continue   # reflex vertex, not an ear

            # No other polygon vertex inside the triangle
            inside = False
            for j in range(n_cur):
                ji = idxs[j]
                if ji in (prev_i, curr_i, next_i):
                    continue
                if _point_in_triangle_2d(loop_2d[ji], A, B, C):
                    inside = True
                    break
            if inside:
                continue

            # Valid ear — add triangle and remove vertex
            tris.append((
                loop_indices[prev_i],
                loop_indices[curr_i],
                loop_indices[next_i],
            ))
            idxs.pop(i)
            ear_found = True
            break

        if not ear_found:
            break   # degenerate polygon, give up remaining

    # Last triangle
    if len(idxs) == 3:
        tris.append((
            loop_indices[idxs[0]],
            loop_indices[idxs[1]],
            loop_indices[idxs[2]],
        ))

    return tris


def _hole_normal(loop: List[Tuple], vertices: np.ndarray,
                 mesh_faces: np.ndarray, edge_map: Dict) -> np.ndarray:
    """
    Estimate the outward normal for a hole by averaging the normals of faces
    adjacent to the boundary edges, then inverting (hole faces face inward).
    """
    adj_normals = []
    for a, b in loop:
        key = (min(a, b), max(a, b))
        for fi in edge_map.get(key, []):
            face = mesh_faces[fi]
            n    = _face_normal(vertices, face)
            adj_normals.append(n)
    if not adj_normals:
        return np.array([0., 0., 1.])
    avg = np.mean(adj_normals, axis=0)
    ln  = np.linalg.norm(avg)
    return -avg / ln if ln > 1e-12 else np.array([0., 0., 1.])


def repair_planar_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    config: RepairConfig,
    log: RepairLog
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return improved_hole_fill(vertices, faces, normals, config, log)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 3-B: Fill Non-Planar Holes — Centroid Fan + Advancing Front
# ─────────────────────────────────────────────────────────────────────────────

def _advancing_front_patch(
    loop_verts: List[int],
    vertices: np.ndarray,
    edge_map: Dict,
    mesh_faces: np.ndarray
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Simple advancing front strategy for non-planar holes:
    1. Add a centroid vertex at the centre of the boundary.
    2. Build a fan of triangles from centroid to each boundary edge.
    3. For boundary edges with large dihedral mismatches, subdivide once.
    Returns (new_vertices, new_face_arrays).
    """
    pts       = vertices[loop_verts]
    centroid  = pts.mean(axis=0)
    new_v_idx = len(vertices)

    verts_out    = np.vstack([vertices, centroid[np.newaxis]])
    new_faces    = []

    # Determine preferred outward direction from adjacent face normals
    hole_normal_acc = np.zeros(3)
    n_adj = 0
    for i, a in enumerate(loop_verts):
        b = loop_verts[(i + 1) % len(loop_verts)]
        key = (min(a, b), max(a, b))
        for fi in edge_map.get(key, []):
            hole_normal_acc += _face_normal(vertices, mesh_faces[fi])
            n_adj += 1

    if n_adj > 0:
        hole_dir = -hole_normal_acc / n_adj   # invert: patch faces outward
    else:
        hole_dir = np.array([0., 0., 1.])

    for i in range(len(loop_verts)):
        a = loop_verts[i]
        b = loop_verts[(i + 1) % len(loop_verts)]
        face = np.array([a, b, new_v_idx], dtype=np.int32)
        n    = _face_normal(verts_out, face)
        if float(np.dot(n, hole_dir)) < 0:
            face[[1, 2]] = face[[2, 1]]
        new_faces.append(face)

    return verts_out, new_faces


def repair_non_planar_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    config: RepairConfig,
    log: RepairLog
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return vertices, faces, normals


def _vertex_turn_key(prev_idx: Optional[int],
                     cur_idx: int,
                     cand_idx: int,
                     vertices: np.ndarray) -> Tuple[float, float, int]:
    if prev_idx is None:
        edge = vertices[cand_idx] - vertices[cur_idx]
        return (0.0, float(np.linalg.norm(edge)), int(cand_idx))
    incoming = vertices[cur_idx] - vertices[prev_idx]
    outgoing = vertices[cand_idx] - vertices[cur_idx]
    in_len = float(np.linalg.norm(incoming))
    out_len = float(np.linalg.norm(outgoing))
    if in_len < 1e-15 or out_len < 1e-15:
        return (float("inf"), out_len, int(cand_idx))
    cosine = float(np.clip(np.dot(incoming, outgoing) / (in_len * out_len), -1.0, 1.0))
    return (-cosine, out_len, int(cand_idx))


def robust_boundary_loop(
    naked_edges: List[Tuple[int, int]],
    vertices: np.ndarray
) -> List[Dict[str, object]]:
    """
    Deterministic traversal over all boundary edges. This fixes open-edge
    misses by consuming every naked edge exactly once, including open chains.
    """
    if not naked_edges:
        return []

    adjacency: Dict[int, Set[int]] = defaultdict(set)
    unique_edges: Set[Tuple[int, int]] = set()
    for a, b in naked_edges:
        a_i, b_i = int(a), int(b)
        if a_i == b_i:
            continue
        edge = (a_i, b_i) if a_i < b_i else (b_i, a_i)
        if edge in unique_edges:
            continue
        unique_edges.add(edge)
        adjacency[a_i].add(b_i)
        adjacency[b_i].add(a_i)

    unused = set(unique_edges)
    result: List[Dict[str, object]] = []

    def consume_walk(start: int, prev: Optional[int]) -> List[int]:
        walk = [start]
        cur = start
        prev_local = prev
        safety = len(unique_edges) + 2
        while safety > 0:
            safety -= 1
            candidates = []
            for nb in adjacency[cur]:
                edge = (cur, nb) if cur < nb else (nb, cur)
                if edge in unused and nb != prev_local:
                    candidates.append(nb)
            if not candidates and prev_local is not None:
                for nb in adjacency[cur]:
                    edge = (cur, nb) if cur < nb else (nb, cur)
                    if edge in unused:
                        candidates.append(nb)
            if not candidates:
                break
            next_v = min(
                candidates,
                key=lambda nb: _vertex_turn_key(prev_local, cur, nb, vertices)
            )
            edge = (cur, next_v) if cur < next_v else (next_v, cur)
            unused.remove(edge)
            prev_local, cur = cur, next_v
            if cur == start:
                walk.append(cur)
                break
            walk.append(cur)
        return walk

    while unused:
        seed = next(iter(unused))
        component = set()
        stack = [seed[0], seed[1]]
        while stack:
            v = stack.pop()
            if v in component:
                continue
            component.add(v)
            stack.extend(list(adjacency[v]))

        while True:
            component_edges = [e for e in unused if e[0] in component or e[1] in component]
            if not component_edges:
                break
            odd_vertices = sorted(v for v in component if len(adjacency[v]) % 2 == 1)
            start = odd_vertices[0] if odd_vertices else min(min(e) for e in component_edges)
            walk = consume_walk(start, None)
            if len(walk) < 2:
                unused.discard(component_edges[0])
                continue
            closed = len(walk) > 2 and walk[0] == walk[-1]
            ordered = walk[:-1] if closed else walk
            result.append({
                "vertices": ordered,
                "closed": closed,
                "edges": len(ordered) if closed else max(len(ordered) - 1, 0),
            })

    return result


def _polygon_area_2d(loop_2d: np.ndarray) -> float:
    x = loop_2d[:, 0]
    y = loop_2d[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _clean_loop_vertices(loop_vertices: List[int]) -> List[int]:
    cleaned: List[int] = []
    for v in loop_vertices:
        vi = int(v)
        if not cleaned or cleaned[-1] != vi:
            cleaned.append(vi)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    if len(cleaned) < 3:
        return cleaned

    unique_loop: List[int] = []
    seen: Set[int] = set()
    for v in cleaned:
        if v in seen:
            break
        unique_loop.append(v)
        seen.add(v)
    return unique_loop


def _newell_normal(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    shifted = np.roll(points, -1, axis=0)
    nx = np.sum((points[:, 1] - shifted[:, 1]) * (points[:, 2] + shifted[:, 2]))
    ny = np.sum((points[:, 2] - shifted[:, 2]) * (points[:, 0] + shifted[:, 0]))
    nz = np.sum((points[:, 0] - shifted[:, 0]) * (points[:, 1] + shifted[:, 1]))
    normal = np.array([nx, ny, nz], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    return normal / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _safe_ear_clip(loop_2d: np.ndarray,
                   loop_indices: List[int]) -> Optional[List[Tuple[int, int, int]]]:
    if len(loop_indices) < 3:
        return None
    work = list(range(len(loop_indices)))
    if _polygon_area_2d(loop_2d) < 0:
        work.reverse()
    triangles: List[Tuple[int, int, int]] = []
    safety = len(work) * len(work)
    while len(work) > 3 and safety > 0:
        safety -= 1
        ear_found = False
        for i in range(len(work)):
            i0 = work[(i - 1) % len(work)]
            i1 = work[i]
            i2 = work[(i + 1) % len(work)]
            a = loop_2d[i0]
            b = loop_2d[i1]
            c = loop_2d[i2]
            if _cross2d(a, b, c) <= 1e-12:
                continue
            contains_vertex = False
            for j in work:
                if j in (i0, i1, i2):
                    continue
                if _point_in_triangle_2d(loop_2d[j], a, b, c):
                    contains_vertex = True
                    break
            if contains_vertex:
                continue
            triangles.append((loop_indices[i0], loop_indices[i1], loop_indices[i2]))
            work.pop(i)
            ear_found = True
            break
        if not ear_found:
            return None
    if len(work) == 3:
        triangles.append((loop_indices[work[0]], loop_indices[work[1]], loop_indices[work[2]]))
    return triangles if len(triangles) == len(loop_indices) - 2 else None


def _hole_reference_normal(loop_vertices: List[int],
                           vertices: np.ndarray,
                           faces: np.ndarray,
                           edge_map: Dict) -> np.ndarray:
    acc = np.zeros(3, dtype=np.float64)
    count = 0
    for i, a in enumerate(loop_vertices):
        b = loop_vertices[(i + 1) % len(loop_vertices)]
        edge = (a, b) if a < b else (b, a)
        for fi in edge_map.get(edge, []):
            acc += _face_normal(vertices, faces[fi])
            count += 1
    if count > 0:
        acc /= float(count)
    if np.linalg.norm(acc) < 1e-12:
        pts = vertices[np.array(loop_vertices, dtype=np.int32)]
        acc = _newell_normal(pts)
    norm = float(np.linalg.norm(acc))
    return -acc / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _orient_patch_faces(vertices: np.ndarray,
                        patch_faces: List[np.ndarray],
                        hole_normal: np.ndarray) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for face in patch_faces:
        face_arr = np.array(face, dtype=np.int32)
        if _face_area(vertices, face_arr) < 1e-15:
            continue
        normal = _face_normal(vertices, face_arr)
        if float(np.dot(normal, hole_normal)) < 0:
            face_arr[[1, 2]] = face_arr[[2, 1]]
        out.append(face_arr)
    return out


def _centroid_fan_patch(loop_vertices: List[int],
                        vertices: np.ndarray,
                        hole_normal: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    pts = vertices[np.array(loop_vertices, dtype=np.int32)]
    centroid = pts.mean(axis=0)
    # Project centroid back toward the estimated hole plane so fallback fans
    # do not create unnecessary sliver triangles on nearly planar holes.
    plane_normal = hole_normal / (np.linalg.norm(hole_normal) + 1e-15)
    plane_offset = np.dot(pts[0] - centroid, plane_normal)
    centroid = centroid + plane_offset * plane_normal
    new_vertices = np.vstack([vertices, centroid[np.newaxis, :]])
    center_idx = len(vertices)
    patch = [
        np.array([loop_vertices[i], loop_vertices[(i + 1) % len(loop_vertices)], center_idx],
                 dtype=np.int32)
        for i in range(len(loop_vertices))
    ]
    return new_vertices, _orient_patch_faces(new_vertices, patch, hole_normal)


def _stitch_collinear_loop(
    loop_vertices: List[int],
    vertices: np.ndarray,
    faces: np.ndarray
) -> Tuple[np.ndarray, bool]:
    """
    Repair zero-area "holes" that are really T-junction seams:
    a collinear boundary chain closed by one long chord edge. We fix these by
    splitting the triangle incident to the chord edge across the intermediate
    chain vertices instead of trying to add a degenerate patch.
    """
    if len(loop_vertices) < 3:
        return faces, False

    pts = vertices[np.array(loop_vertices, dtype=np.int32)]
    centered = pts - pts.mean(axis=0)
    _, svals, _ = np.linalg.svd(centered, full_matrices=False)
    if len(svals) < 2 or float(svals[1]) > 1e-8 * max(float(svals[0]), 1e-15):
        return faces, False

    edges = [(loop_vertices[i], loop_vertices[(i + 1) % len(loop_vertices)])
             for i in range(len(loop_vertices))]
    lengths_sq = [
        float(np.dot(vertices[b] - vertices[a], vertices[b] - vertices[a]))
        for a, b in edges
    ]
    if not lengths_sq:
        return faces, False

    chord_idx = int(np.argmax(lengths_sq))
    chord_a, chord_b = edges[chord_idx]
    chain = [chord_b]
    j = (chord_idx + 1) % len(loop_vertices)
    while True:
        chain.append(loop_vertices[(j + 1) % len(loop_vertices)])
        if loop_vertices[(j + 1) % len(loop_vertices)] == chord_a:
            break
        j = (j + 1) % len(loop_vertices)
        if len(chain) > len(loop_vertices) + 1:
            return faces, False

    ordered_chain = chain
    if len(ordered_chain) < 3:
        return faces, False

    edge_map = build_edge_map(faces)
    chord_key = (min(chord_a, chord_b), max(chord_a, chord_b))
    adjacent = edge_map.get(chord_key, [])
    if len(adjacent) != 1:
        return faces, False

    fi = int(adjacent[0])
    face = faces[fi]
    third_candidates = [int(v) for v in face if int(v) not in chord_key]
    if len(third_candidates) != 1:
        return faces, False
    third = third_candidates[0]

    new_tris = []
    for i in range(len(ordered_chain) - 1):
        tri = np.array([third, ordered_chain[i], ordered_chain[i + 1]], dtype=np.int32)
        if _face_area(vertices, tri) < 1e-15:
            return faces, False
        ref_n = _face_normal(vertices, face)
        tri_n = _face_normal(vertices, tri)
        if float(np.dot(ref_n, tri_n)) < 0:
            tri[[1, 2]] = tri[[2, 1]]
        new_tris.append(tri)

    keep_mask = np.ones(len(faces), dtype=bool)
    keep_mask[fi] = False
    return np.vstack([faces[keep_mask], np.array(new_tris, dtype=np.int32)]), True


def improved_hole_fill(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    config: RepairConfig,
    log: RepairLog
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_map = build_edge_map(faces)
    naked, _ = check_naked_and_nonmanifold(edge_map)
    if not naked:
        return vertices, faces, normals

    boundary_sets = robust_boundary_loop(naked, vertices)
    closed = [item for item in boundary_sets if item["closed"] and len(item["vertices"]) >= 3]
    if not closed:
        return vertices, faces, normals

    stitched = 0
    stitched_face_delta = 0
    remaining_closed = []
    for item in closed:
        loop_vertices = _clean_loop_vertices(list(item["vertices"]))
        updated_faces, fixed = _stitch_collinear_loop(loop_vertices, vertices, faces)
        if fixed:
            stitched += 1
            stitched_face_delta += len(updated_faces) - len(faces)
            faces = updated_faces
        else:
            remaining_closed.append(item)

    if stitched:
        normals = _recompute_normals(vertices, faces)
        log.planar_holes_filled += stitched
        log.faces_added += stitched_face_delta

    if not remaining_closed:
        return vertices, faces, normals

    edge_map = build_edge_map(faces)
    new_faces: List[np.ndarray] = []
    planar_filled = 0
    fan_filled = 0

    for item in remaining_closed:
        loop_vertices = _clean_loop_vertices(list(item["vertices"]))
        if len(loop_vertices) < 3 or len(set(loop_vertices)) < 3:
            continue
        pts_3d = vertices[np.array(loop_vertices, dtype=np.int32)]
        edge_lengths = np.linalg.norm(pts_3d - np.roll(pts_3d, -1, axis=0), axis=1)
        if np.any(edge_lengths <= 1e-15):
            continue
        hole_normal = _hole_reference_normal(loop_vertices, vertices, faces, edge_map)
        centered = pts_3d - pts_3d.mean(axis=0)
        _, svals, _ = np.linalg.svd(centered, full_matrices=False)
        planar_ratio = float(svals[-1] / max(svals[0], 1e-15))

        patch: Optional[List[np.ndarray]] = None
        used_fallback = True
        if planar_ratio <= config.planar_tolerance:
            pts_2d, _, _ = _project_to_2d(pts_3d, hole_normal)
            if abs(_polygon_area_2d(pts_2d)) <= 1e-15:
                pts_2d = None
            else:
                keep = []
                n2d = len(pts_2d)
                for i in range(n2d):
                    prev_p = pts_2d[(i - 1) % n2d]
                    curr_p = pts_2d[i]
                    next_p = pts_2d[(i + 1) % n2d]
                    if abs(_cross2d(prev_p, curr_p, next_p)) > 1e-14:
                        keep.append(i)
                if len(keep) >= 3 and len(keep) < len(loop_vertices):
                    loop_vertices = [loop_vertices[i] for i in keep]
                    pts_3d = pts_3d[np.array(keep, dtype=np.int32)]
                    pts_2d = pts_2d[np.array(keep, dtype=np.int32)]
                elif len(keep) < 3:
                    pts_2d = None
            if pts_2d is not None:
                tris = _safe_ear_clip(pts_2d, loop_vertices)
            else:
                tris = None
            if tris is not None:
                patch = _orient_patch_faces(
                    vertices,
                    [np.array(tri, dtype=np.int32) for tri in tris],
                    hole_normal
                )
                if len(patch) == len(loop_vertices) - 2:
                    used_fallback = False
                else:
                    patch = None

        if patch is None:
            vertices, patch = _centroid_fan_patch(loop_vertices, vertices, hole_normal)
            used_fallback = True

        if not patch:
            continue

        new_faces.extend(patch)
        if used_fallback:
            fan_filled += 1
        else:
            planar_filled += 1

    if not new_faces:
        return vertices, faces, normals

    faces = np.vstack([faces, np.array(new_faces, dtype=np.int32)])
    normals = _recompute_normals(vertices, faces)
    log.faces_added += len(new_faces)
    log.planar_holes_filled += planar_filled
    log.non_planar_filled += fan_filled
    return vertices, faces, normals


def improved_orientation(
    vertices: np.ndarray,
    faces: np.ndarray,
    log: RepairLog
) -> np.ndarray:
    if len(faces) == 0:
        return faces

    faces = faces.copy()
    edge_map = build_edge_map(faces)
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

# ─────────────────────────────────────────────────────────────────────────────
#  STL Writer  (binary, preserves exact normals)
# ─────────────────────────────────────────────────────────────────────────────

def write_stl(path: str, vertices: np.ndarray,
              faces: np.ndarray, normals: np.ndarray) -> None:
    """Write a binary STL file, recomputing normals from winding first."""
    # Always recompute normals from winding for a clean output
    recomputed = _recompute_normals(vertices, faces)
    with open(path, "wb") as f:
        hdr = b"Repaired by MeshRepairPipeline"
        f.write((hdr + b"\x00" * 80)[:80])                  # exactly 80 bytes
        f.write(struct.pack("<I", len(faces)))
        for fi, face in enumerate(faces):
            n = recomputed[fi]
            f.write(struct.pack("<3f", *n))
            for vi in face:
                f.write(struct.pack("<3f", *vertices[vi]))
            f.write(struct.pack("<H", 0))


# ─────────────────────────────────────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class MeshRepairPipeline:
    """
    Orchestrates all repair stages in dependency order:
      1. Remove degenerate faces   (safe, no topology side-effects)
      2. Remove duplicate faces    (safe)
      3. Resolve non-manifold      (needs clean mesh first)
      4. Fix inverted normals      (BFS needs manifold topology)
      5. Fill planar holes         (ear-clipping)
      6. Fill non-planar holes     (centroid fan)
    """

    def __init__(self, config: Optional[RepairConfig] = None):
        self.config = config or RepairConfig()

    def _count_defects(self, vertices, faces):
        """Quick defect tally without going through file I/O."""
        edge_map = build_edge_map(faces)
        naked, non_manif = check_naked_and_nonmanifold(edge_map)

        if len(faces) == 0:
            return len(naked), len(non_manif), 0, 0

        repeated = (
            (faces[:, 0] == faces[:, 1]) |
            (faces[:, 1] == faces[:, 2]) |
            (faces[:, 0] == faces[:, 2])
        )
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        area2 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        n_degen = int(np.count_nonzero(repeated | (area2 < self.config.area_tolerance)))

        canonical = np.sort(faces, axis=1)
        n_unique = np.unique(canonical, axis=0).shape[0]
        n_dup = int(len(faces) - n_unique)

        return len(naked), len(non_manif), n_degen, n_dup

    def repair(self,
               input_path: str,
               output_path: str,
               verbose: bool = True,
               max_iterations: int = 5) -> Tuple[MeshData, RepairLog]:

        t0  = time.perf_counter()
        log = RepairLog()

        # ── Load ────────────────────────────────────────────────────
        if verbose:
            print(f"\n  Loading  : {input_path}")
        analyzer  = MeshAnalyzer(self.config.merge_tolerance)
        mesh, pre = analyzer.analyze(input_path)

        if verbose:
            print(pre.summary())
            print("\n  Repairing ...\n")

        vertices = mesh.vertices.copy()
        faces    = mesh.faces.copy()
        normals  = mesh.normals.copy()

        # ── Stage 1-A: Degenerate ────────────────────────────────────
        faces, normals = repair_degenerate_faces(
            vertices, faces, normals, self.config, log)
        if verbose and log.degenerate_removed:
            print(f"  [1/6] Removed {log.degenerate_removed} degenerate face(s)")

        # ── Stage 1-B: Duplicates ────────────────────────────────────
        faces, normals = repair_duplicate_faces(faces, normals, log)
        if verbose and log.duplicate_removed:
            print(f"  [2/6] Removed {log.duplicate_removed} duplicate face(s)")

        # ── Stage 2-A: Non-Manifold ──────────────────────────────────
        faces, normals = repair_non_manifold(
            vertices, faces, normals, self.config, log)
        if verbose and log.non_manifold_removed:
            print(f"  [3/6] Removed {log.non_manifold_removed} non-manifold extra face(s)")

        # ── Stage 2-B: Inverted Normals ──────────────────────────────
        faces = improved_orientation(vertices, faces, log)
        normals = _recompute_normals(vertices, faces)
        if verbose and log.faces_reoriented:
            print(f"  [4/6] Reoriented {log.faces_reoriented} face(s)")

        # ── Stage 2-C: Boundary Sliver Welding ─────────────────────────
        vertices, faces, normals = global_vertex_weld(
            vertices, faces, normals, self.config, log)
        if verbose and log.vertices_welded:
            print(f"  [5/7] Welded {log.vertices_welded} vertex pair(s)")

        # ── Stage 3-A: Planar Holes ──────────────────────────────────
        prev_planar = log.planar_holes_filled
        prev_non_planar = log.non_planar_filled
        prev_added = log.faces_added
        vertices, faces, normals = repair_planar_holes(
            vertices, faces, normals, self.config, log)
        if verbose and (log.planar_holes_filled > prev_planar or log.non_planar_filled > prev_non_planar):
            print(f"  [6/7] Filled {log.planar_holes_filled - prev_planar} planar hole(s) and "
                  f"{log.non_planar_filled - prev_non_planar} fallback hole(s) "
                  f"({log.faces_added - prev_added} new face(s))")

        # ── Stage 3-B: Non-Planar Holes ──────────────────────────────
        vertices, faces, normals = repair_non_planar_holes(
            vertices, faces, normals, self.config, log)

        # ─────────────────────────────────────────────────────────────
        #  ITERATIVE CONVERGENCE LOOP
        #  After the first pass, repairs can create secondary defects:
        #   - Removing non-manifold faces opens new holes
        #   - Hole-filling patches can be degenerate
        #  Keep cleaning + filling until nothing changes.
        # ─────────────────────────────────────────────────────────────
        prev_total_defects = float('inf')

        for iteration in range(1, max_iterations + 1):
            n_naked, n_nm, n_degen, n_dup = self._count_defects(vertices, faces)
            total_defects = n_naked + n_nm + n_degen + n_dup

            if total_defects == 0:
                if verbose:
                    print(f"  [✓] Mesh is clean after pass {iteration}. No remaining defects.")
                break

            if total_defects >= prev_total_defects:
                if verbose:
                    print(f"  [!] Convergence stalled at pass {iteration} "
                          f"({total_defects} defects remain). Stopping.")
                break

            prev_total_defects = total_defects

            if verbose:
                print(f"\n  ── Iterative pass {iteration} "
                      f"(naked={n_naked}, non-manifold={n_nm}, "
                      f"degenerate={n_degen}, duplicate={n_dup}) ──")

            # -- Clean degenerate faces created by hole filling ---------
            if n_degen > 0:
                iter_log = RepairLog()
                faces, normals = repair_degenerate_faces(
                    vertices, faces, normals, self.config, iter_log)
                log.degenerate_removed += iter_log.degenerate_removed
                if verbose and iter_log.degenerate_removed:
                    print(f"    Removed {iter_log.degenerate_removed} degenerate face(s)")

            # -- Clean duplicates that slipped through ------------------
            if n_dup > 0:
                iter_log = RepairLog()
                faces, normals = repair_duplicate_faces(faces, normals, iter_log)
                log.duplicate_removed += iter_log.duplicate_removed
                if verbose and iter_log.duplicate_removed:
                    print(f"    Removed {iter_log.duplicate_removed} duplicate face(s)")

            # -- Re-check for non-manifold edges after cleanup ----------
            if n_nm > 0:
                iter_log = RepairLog()
                faces, normals = repair_non_manifold(
                    vertices, faces, normals, self.config, iter_log)
                log.non_manifold_removed += iter_log.non_manifold_removed
                if verbose and iter_log.non_manifold_removed:
                    print(f"    Removed {iter_log.non_manifold_removed} non-manifold extra face(s)")

            # -- Re-orient normals after cleanup -------------------------
            iter_log_n = RepairLog()
            faces = repair_inverted_normals(vertices, faces, iter_log_n)
            normals = _recompute_normals(vertices, faces)
            log.faces_reoriented += iter_log_n.faces_reoriented
            if verbose and iter_log_n.faces_reoriented:
                print(f"    Reoriented {iter_log_n.faces_reoriented} face(s)")

            # -- Weld boundary slivers before attempting hole fill -------
            edge_map_now = build_edge_map(faces)
            naked_now, _ = check_naked_and_nonmanifold(edge_map_now)

            if naked_now:
                iter_log_w = RepairLog()
                vertices, faces, normals = repair_boundary_slivers(
                    vertices, faces, normals, self.config, iter_log_w)
                log.vertices_welded += iter_log_w.vertices_welded
                log.degenerate_removed += iter_log_w.degenerate_removed
                if verbose and iter_log_w.vertices_welded:
                    print(f"    Welded {iter_log_w.vertices_welded} vertex pair(s)")

                # Re-check after welding
                edge_map_now = build_edge_map(faces)
                naked_now, _ = check_naked_and_nonmanifold(edge_map_now)

            # -- Re-fill any newly opened holes -------------------------
            if naked_now:
                # Try planar holes first
                iter_log_p = RepairLog()
                vertices, faces, normals = repair_planar_holes(
                    vertices, faces, normals, self.config, iter_log_p)
                log.planar_holes_filled += iter_log_p.planar_holes_filled
                log.non_planar_filled += iter_log_p.non_planar_filled
                log.faces_added += iter_log_p.faces_added
                if verbose and iter_log_p.faces_added:
                    print(f"    Filled {iter_log_p.planar_holes_filled} planar hole(s) and "
                          f"{iter_log_p.non_planar_filled} fallback hole(s) "
                          f"({iter_log_p.faces_added} new face(s))")

                # Then non-planar
                iter_log_np = RepairLog()
                vertices, faces, normals = repair_non_planar_holes(
                    vertices, faces, normals, self.config, iter_log_np)
                log.non_planar_filled += iter_log_np.non_planar_filled
                log.faces_added += iter_log_np.faces_added
                if verbose and iter_log_np.non_planar_filled:
                    print(f"    Filled {iter_log_np.non_planar_filled} non-planar hole(s) "
                          f"({iter_log_np.faces_added} new face(s))")

        # ── Final recompute normals & write ───────────────────────────
        normals = _recompute_normals(vertices, faces)
        write_stl(output_path, vertices, faces, normals)
        log.repair_time_ms = (time.perf_counter() - t0) * 1000

        if verbose:
            print(f"\n  Written  : {output_path}")

        # ── Post-repair analysis ─────────────────────────────────────
        repaired_mesh = MeshData(vertices, faces, normals)
        if verbose:
            post_analyzer = MeshAnalyzer(self.config.merge_tolerance)
            _, post = post_analyzer.analyze(output_path)
            print("\n  ── POST-REPAIR VERIFICATION ──")
            print(post.summary())
            print(log.summary())

        return repaired_mesh, log


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os

    # Force UTF-8 output on Windows (cp1252 can't render ━ ✓ ✗)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 3:
        print("Usage: python mesh_repair.py <input.stl> <output.stl>")
        sys.exit(1)

    inp, out = sys.argv[1], sys.argv[2]
    if not os.path.exists(inp):
        print(f"[ERROR] File not found: {inp}")
        sys.exit(1)

    pipeline = MeshRepairPipeline()
    pipeline.repair(inp, out, verbose=True)
