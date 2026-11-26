"""Connected components, containment diagnostics, and explicit Manifold64 union.

Surface connectivity and manifold sheets are distinct. Small closed parts are
preserved. Approximate interior culling and naive merges never establish success.
The full pipeline verifies geometry and the serialized output before publishing.
"""

from __future__ import annotations

import sys
from pathlib import Path
if __package__ in {None,''}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.core.repair_config import RepairConfig, RepairLog
from src.core.repair_utils import _recompute_normals, _compact_mesh, _signed_volume
from src.routines.repair_welding import global_vertex_weld
from src.routines.repair_degenerates import repair_degenerate_faces, repair_duplicate_faces
from src.core.mesh_analyzer import build_edge_map, check_naked_and_nonmanifold


# ─────────────────────────────────────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShellInfo:
    """Metadata for one connected component (shell)."""
    index:        int                  # shell index (0-based)
    face_indices: List[int]            # indices into the global faces array
    face_count:   int = 0
    is_open:      bool = False         # has naked edges
    volume:       float = 0.0         # signed volume (only valid for closed shells)
    total_area:   float = 0.0
    bbox_min:     np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max:     np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Relationship fields filled in by classify_shells()
    parent_index: Optional[int] = None   # index of the smallest shell that contains this one
    depth:        int = 0                # nesting depth (0 = outermost)


@dataclass
class ShellUnionLog:
    shells_found:         int   = 0
    shells_removed:       int   = 0    # inner shells eliminated (solid union)
    shells_flipped:       int   = 0    # inner shells flipped (hollow mode)
    interior_faces_culled: int  = 0
    vertices_welded:      int   = 0
    strategy_used:        str   = "none"
    success:              bool  = False

    def summary(self) -> str:
        lines = [
            "━" * 60,
            "  SHELL UNION LOG",
            "━" * 60,
            f"  Strategy used             : {self.strategy_used}",
            f"  Shells found              : {self.shells_found}",
            f"  Shells removed (solid)    : {self.shells_removed}",
            f"  Shells flipped (hollow)   : {self.shells_flipped}",
            f"  Interior faces culled     : {self.interior_faces_culled}",
            f"  Boundary vertices welded  : {self.vertices_welded}",
            f"  Result                    : {'✓ unified' if self.success else '✗ partial'}",
            "━" * 60,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Shell Detection
# ─────────────────────────────────────────────────────────────────────────────

def _find_shells(
    faces: np.ndarray,
    canonical_map: Dict[Tuple[int, int], List[int]],
    manifold_only: bool = False,
) -> List[List[int]]:
    """Connected surfaces use all shared edges; manifold sheets are explicit."""
    n       = len(faces)
    visited = np.zeros(n, dtype=bool)
    adj: List[List[int]] = [[] for _ in range(n)]
    for flist in canonical_map.values():
        if len(flist) < 2 or (manifold_only and len(flist) != 2):
            continue
        # A star connects a high-degree edge in linear rather than quadratic time.
        a = flist[0]
        for b in flist[1:]:
            adj[a].append(b)
            adj[b].append(a)

    shells: List[List[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        queue = deque([start])
        shell: List[int] = []
        visited[start] = True
        while queue:
            fi = queue.popleft()
            shell.append(fi)
            for nb in adj[fi]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        shells.append(shell)
    return shells


def _build_shell_infos(
    vertices: np.ndarray,
    faces: np.ndarray,
    shells: List[List[int]],
    canonical_map: Dict[Tuple[int, int], List[int]],
) -> List[ShellInfo]:
    infos: List[ShellInfo] = []
    for i, shell in enumerate(shells):
        arr  = np.array(shell, dtype=np.int32)
        vset = np.unique(faces[arr])
        pts  = vertices[vset]
        bmin = pts.min(axis=0)
        bmax = pts.max(axis=0)

        # Naked edge check (scoped to this shell)
        shell_set  = set(shell)
        is_open    = False
        for fi in shell:
            face = faces[fi]
            for j in range(3):
                a, b = int(face[j]), int(face[(j + 1) % 3])
                key  = (min(a, b), max(a, b))
                nbrs = canonical_map.get(key, [])
                if len(nbrs) == 1 or any(f not in shell_set for f in nbrs):
                    is_open = True
                    break
            if is_open:
                break

        vol = 0.0
        if not is_open:
            vol = _signed_volume(vertices, faces[arr])

        # Area calculation
        v0p = vertices[faces[arr, 0]]
        v1p = vertices[faces[arr, 1]]
        v2p = vertices[faces[arr, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1p - v0p, v2p - v0p), axis=1)
        t_area = float(np.sum(areas))

        infos.append(ShellInfo(
            index=i,
            face_indices=shell,
            face_count=len(shell),
            is_open=is_open,
            volume=vol,
            total_area=t_area,
            bbox_min=bmin,
            bbox_max=bmax,
        ))
    return infos


def filter_small_shells(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    config: RepairConfig,
    log: RepairLog,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove any disjoint shells whose surface area is less than 
    threshold_ratio * total_area.
    """
    if len(faces) == 0:
        return vertices, faces, normals
        
    config.check_deadline()
    if not config.shell_filter_enabled:
        return vertices, faces, normals
    canonical_map = build_edge_map(faces)
    raw_shells = _find_shells(faces, canonical_map)
    if len(raw_shells) <= 1:
        return vertices, faces, normals

    infos = _build_shell_infos(vertices, faces, raw_shells, canonical_map)
    total_mesh_area = sum(info.total_area for info in infos)
    
    keep_indices = []
    filtered_count = 0
    for info in infos:
        config.check_deadline()
        if info.is_open and total_mesh_area > 0 and (info.total_area / total_mesh_area) < config.shell_area_threshold:
            filtered_count += 1
            if verbose:
                print(f"    [filter] Removing shell {info.index+1} (Area={info.total_area:.2e}, "
                      f"{(info.total_area/total_mesh_area)*100:.3f}% of total)")
            continue
        keep_indices.extend(info.face_indices)

    if filtered_count > 0 and keep_indices:
        log.shells_filtered += filtered_count
        mask = np.array(keep_indices, dtype=np.int32)
        return vertices, faces[mask], normals[mask]
    
    return vertices, faces, normals


# ─────────────────────────────────────────────────────────────────────────────
#  Containment / Overlap Detection
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_contains(outer: ShellInfo, inner: ShellInfo) -> bool:
    """True if outer's AABB fully encloses inner's AABB."""
    return (np.all(outer.bbox_min <= inner.bbox_min) and
            np.all(outer.bbox_max >= inner.bbox_max))


def _bbox_overlaps(a: ShellInfo, b: ShellInfo) -> bool:
    """True if two AABBs overlap (shared volume)."""
    return (np.all(a.bbox_min <= b.bbox_max) and
            np.all(a.bbox_max >= b.bbox_min))


def _point_inside_shell(
    point: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    shell_face_indices: List[int],
    n_rays: int = 6,
) -> bool:
    """
    Möller–Trumbore ray casting: shoot rays in N axis-aligned directions from
    `point` and count intersections with the shell.  Majority vote.
    Returns True if point is inside the shell.
    """
    shell_arr = np.array(shell_face_indices, dtype=np.int32)
    v0 = vertices[faces[shell_arr, 0]]
    v1 = vertices[faces[shell_arr, 1]]
    v2 = vertices[faces[shell_arr, 2]]
    origin=np.minimum(np.minimum(v0.min(axis=0),v1.min(axis=0)),v2.min(axis=0))
    extent=np.maximum(np.maximum(v0.max(axis=0),v1.max(axis=0)),v2.max(axis=0))-origin
    scale=float(np.linalg.norm(extent))
    if scale==0:
        return False
    v0=(v0-origin)/scale; v1=(v1-origin)/scale; v2=(v2-origin)/scale
    point=(point-origin)/scale
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Use slightly perturbed directions for robustness against edge-hits
    directions = np.array([
        [1, 0.0123, 0.00456], [-1, -0.0123, -0.00456],
        [0.00567, 1, 0.0123], [-0.00567, -1, -0.0123],
        [0.0123, 0.00567, 1], [-0.0123, -0.00567, -1],
    ], dtype=np.float64)

    inside_votes = 0
    for ray_dir in directions[:n_rays]:
        h     = np.cross(ray_dir, edge2)               # (F, 3)
        a_det = np.einsum('ij,ij->i', edge1, h)        # (F,)
        eps   = 1e-8
        valid = np.abs(a_det) > eps
        if not np.any(valid):
            continue
        f_inv      = np.divide(1.0, a_det, out=np.zeros_like(a_det), where=valid)
        s          = point - v0
        u          = f_inv * np.einsum('ij,ij->i', s, h)
        in_u       = valid & (u >= 0.0) & (u <= 1.0)
        q          = np.cross(s, edge1)
        v_bc       = f_inv * np.einsum('j,ij->i', ray_dir, q)
        in_v       = in_u & (v_bc >= 0.0) & ((u + v_bc) <= 1.0)
        t          = f_inv * np.einsum('ij,ij->i', edge2, q)
        hit_count  = int(np.sum(in_v & (t > eps)))
        if hit_count % 2 == 1:
            inside_votes += 1

    return inside_votes > n_rays // 2


def _sample_shell_point(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_indices: List[int],
) -> np.ndarray:
    """Return the centroid of the largest face in this shell (robust sample)."""
    arr  = np.array(face_indices, dtype=np.int32)
    v0   = vertices[faces[arr, 0]]
    v1   = vertices[faces[arr, 1]]
    v2   = vertices[faces[arr, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    best  = int(np.argmax(areas))
    return (v0[best] + v1[best] + v2[best]) / 3.0


def classify_shells(
    vertices: np.ndarray,
    faces: np.ndarray,
    infos: List[ShellInfo],
) -> List[ShellInfo]:
    """
    Determine parent–child (containment) relationships between shells.
    A shell B is *nested inside* shell A when:
      - A's AABB contains B's AABB, AND
      - A sample point from B tests inside A (ray-cast).

    Sets info.parent_index and info.depth on every ShellInfo.
    """
    n = len(infos)
    if n <= 1:
        return infos

    # Sort by descending volume magnitude so we test larger shells as parents first
    order = sorted(range(n), key=lambda i: -abs(infos[i].volume))

    for i in range(n):
        info_i = infos[i]
        best_parent: Optional[int] = None
        best_vol    = float('inf')

        for j in order:
            if j == i:
                continue
            info_j = infos[j]
            if abs(info_j.volume) <= abs(info_i.volume) * (1 + 32*np.finfo(float).eps):
                continue  # equal/coincident shells are not containment parents
            if info_j.is_open:
                continue       # open shells can't reliably contain anything
            if not _bbox_contains(info_j, info_i):
                continue
            # Quick bbox check passed → do ray cast
            sample = _sample_shell_point(vertices, faces, info_i.face_indices)
            if _point_inside_shell(sample, vertices, faces, info_j.face_indices):
                # Prefer the *smallest* enclosing shell as the direct parent
                encl_vol = abs(info_j.volume)
                if encl_vol < best_vol:
                    best_vol    = encl_vol
                    best_parent = j

        info_i.parent_index = best_parent

    # Compute nesting depth
    def _depth(idx: int, visited: Set[int] = set()) -> int:
        if idx in visited:
            return 0
        visited = visited | {idx}
        parent = infos[idx].parent_index
        if parent is None:
            return 0
        return 1 + _depth(parent, visited)

    for i in range(n):
        infos[i].depth = _depth(i)

    return infos


def repair_shell_union(vertices,faces,normals,config,log,*,hollow=None,verbose=True,force_tier=None):
    """Explicit Boolean operation; approximate culling never claims success."""
    from src.core.mesh_analyzer import MeshData
    from src.core.repair_backends import candidate
    from src.core.validation import validate
    cfg=config.for_mesh(vertices,faces)
    cfg.union_enabled=True
    if hollow is not None:
        cfg.union_hollow=hollow
    cfg.preserve_cavities=cfg.union_hollow
    tier=force_tier if force_tier is not None else cfg.union_force_tier
    ulog=ShellUnionLog()
    if not len(faces):
        return vertices,faces,normals,ulog
    em=build_edge_map(faces)
    shells=_find_shells(faces,em,manifold_only=True)
    ulog.shells_found=len(shells)
    if len(shells)==1:
        report=validate(MeshData(vertices,faces,normals),cfg)
        ulog.success=report.complete and report.valid
        ulog.strategy_used='unchanged-single-shell'
        return vertices,faces,normals,ulog
    if tier==3:
        ulog.strategy_used='unchanged-components (union not verified)'
        return vertices,faces,normals,ulog
    try:
        result=candidate('manifold',MeshData(vertices,faces,normals),cfg)
        report=validate(result,cfg)
        ulog.success=report.complete and report.valid
        ulog.strategy_used='Manifold64-boolean'
        ulog.shells_flipped=sum(1 for shell in shells if _signed_volume(vertices,faces[np.array(shell)])<0)
        return result.vertices,result.faces,result.normals,ulog
    except RuntimeError as exc:
        if verbose:
            print(f'    [union] Incomplete: {exc}')
        return vertices,faces,normals,ulog


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience wrapper: analyze only (no repair)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_shells(
    vertices: np.ndarray,
    faces: np.ndarray,
    verbose: bool = True,
) -> List[ShellInfo]:
    """
    Return ShellInfo for every connected component without modifying the mesh.
    Useful for diagnostics before deciding whether to call repair_shell_union().
    """
    canonical_map = build_edge_map(faces)
    raw_shells    = _find_shells(faces, canonical_map)
    infos         = _build_shell_infos(vertices, faces, raw_shells, canonical_map)
    infos         = classify_shells(vertices, faces, infos)

    if verbose:
        print(f"\n  Shell analysis: {len(infos)} shell(s)")
        for info in infos:
            parent_str = (f"inside shell {info.parent_index+1}"
                          if info.parent_index is not None else "outermost")
            open_str   = "open" if info.is_open else "closed"
            vol_str    = f"{info.volume:+.4e}" if not info.is_open else "N/A (open)"
            print(f"    Shell {info.index+1:>3d}: {info.face_count:>6d} faces  "
                  f"vol={vol_str}  {open_str}  {parent_str}  depth={info.depth}")

    return infos


# ─────────────────────────────────────────────────────────────────────────────
#  CLI  — python repair_shell_union.py input.stl output.stl [--hollow] [-v]
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from src.core.mesh_io import load_mesh
    from src.pipelines.mesh_repair_pipeline import MeshRepairPipeline
    parser=argparse.ArgumentParser(description='Verified explicit material union')
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--hollow',action='store_true',help='Declare nested shells as voids')
    parser.add_argument('--analyze',action='store_true')
    parser.add_argument('--timeout',type=float,default=120)
    args=parser.parse_args()
    if args.analyze:
        mesh=load_mesh(args.input)
        analyze_shells(mesh.vertices,mesh.faces)
    else:
        config=RepairConfig(union_enabled=True,union_hollow=args.hollow,
                            preserve_cavities=args.hollow,timeout_seconds=args.timeout)
        _,result=MeshRepairPipeline(config).repair(args.input,args.output)
        raise SystemExit(0 if result.success else 2)
