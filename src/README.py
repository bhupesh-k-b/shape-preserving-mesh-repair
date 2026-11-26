"""
================================================================================
  MESH REPAIR PIPELINE  —  repair_winding.py
  Dedicated Winding-Order Audit & Repair

  Winding order defines whether a triangle's normal points outward or inward.
  Given triangle (A, B, C):
      - Counter-clockwise (CCW) from outside → correct outward normal
      - Clockwise (CW) from outside          → reversed / inside-out normal

  For a watertight mesh every shared edge must be traversed in OPPOSITE
  directions by its two neighbouring faces:
      Face 1: … A → B …      Face 2: … B → A …   ✓ consistent
      Face 1: … A → B …      Face 2: … A → B …   ✗ winding conflict

  The existing improved_orientation() in repair_normals.py performs BFS
  propagation but it:
    1. Does NOT audit or count winding conflicts before touching anything.
    2. Cannot safely orient open / partial shells (signed-volume = 0 or garbage).
    3. Has no fallback for shells where the volume-sign check is ambiguous.

  This module adds:
    ┌─────────────────────────────────────────────────────────────┐
    │  check_winding_consistency  — pure audit, no modifications  │
    │  repair_winding_order       — three-strategy fix + logging  │
    └─────────────────────────────────────────────────────────────┘

  Three-strategy outward orientation (applied per connected shell):
    Strategy A – Signed Volume   : works for any closed, non-degenerate shell.
    Strategy B – Ray Casting     : fallback for open shells or near-zero volume.
    Strategy C – Boundary Normal : last resort; uses boundary-edge neighbourhood.
================================================================================
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

from src.core.repair_config import RepairConfig, RepairLog
from src.core.repair_utils import _recompute_normals, _signed_volume


# ─────────────────────────────────────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WindingAudit:
    """
    Full winding-order audit result returned by check_winding_consistency().
    All lists contain face indices (into the faces array passed in).
    """
    # Edges where both neighbouring faces traverse in the SAME direction
    conflicted_edges:      List[Tuple[int, int]] = field(default_factory=list)

    # Faces that share at least one conflicted edge with a neighbour
    conflicted_faces:      List[int]             = field(default_factory=list)

    # Faces whose computed normal disagrees with the stored STL normal by > tol
    stored_normal_flipped: List[int]             = field(default_factory=list)

    # Connected components (shells); each entry is a list of face indices
    shells:                List[List[int]]       = field(default_factory=list)

    # Per-shell: True if signed-volume says the shell is inside-out
    shells_inverted:       List[bool]            = field(default_factory=list)

    # Per-shell: True if the shell is open (has naked edges)
    shells_open:           List[bool]            = field(default_factory=list)

    # Percentage of manifold edges that are winding-consistent (0–100)
    consistency_pct:       float                 = 100.0

    def summary(self) -> str:
        lines = [
            "━" * 60,
            "  WINDING ORDER AUDIT",
            "━" * 60,
            f"  Conflicted edges          : {len(self.conflicted_edges)}",
            f"  Faces with conflicts      : {len(self.conflicted_faces)}",
            f"  Stored normals flipped    : {len(self.stored_normal_flipped)}",
            f"  Connected shells          : {len(self.shells)}",
            f"  Consistency               : {self.consistency_pct:.1f} %",
        ]
        for i, (shell, inv, open_) in enumerate(
                zip(self.shells, self.shells_inverted, self.shells_open)):
            status = []
            if inv:
                status.append("inside-out")
            if open_:
                status.append("open")
            status_str = ", ".join(status) if status else "ok"
            lines.append(
                f"    Shell {i+1:>3d}: {len(shell):>6d} faces  [{status_str}]"
            )
        lines.append("━" * 60)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_directed_edge_map(
    faces: np.ndarray,
) -> Tuple[Dict[Tuple[int,int], List[int]],
           Dict[Tuple[int,int], List[int]]]:
    """
    Returns two maps:
      canonical_map : (min_v, max_v) → [face_idx, ...]   — for adjacency
      directed_map  : (a, b)         → [face_idx, ...]   — for winding check
    """
    canonical: Dict[Tuple[int,int], List[int]] = defaultdict(list)
    directed:  Dict[Tuple[int,int], List[int]] = defaultdict(list)
    for fi, face in enumerate(faces):
        for j in range(3):
            a, b = int(face[j]), int(face[(j + 1) % 3])
            canonical[(min(a, b), max(a, b))].append(fi)
            directed[(a, b)].append(fi)
    return canonical, directed


def _connected_shells(
    faces: np.ndarray,
    canonical_map: Dict[Tuple[int,int], List[int]],
) -> List[List[int]]:
    """BFS over manifold adjacency to find connected components."""
    n = len(faces)
    visited = np.zeros(n, dtype=bool)
    # face → list of manifold neighbours
    adj: List[List[int]] = [[] for _ in range(n)]
    for flist in canonical_map.values():
        if len(flist) == 2:
            a, b = flist
            adj[a].append(b)
            adj[b].append(a)

    shells: List[List[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        queue  = deque([start])
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


def _shell_has_naked_edges(
    shell_faces: List[int],
    canonical_map: Dict[Tuple[int,int], List[int]],
    faces: np.ndarray,
) -> bool:
    """True if any edge of this shell's faces appears only once (naked)."""
    face_set = set(shell_faces)
    for fi in shell_faces:
        face = faces[fi]
        for j in range(3):
            a, b = int(face[j]), int(face[(j + 1) % 3])
            key = (min(a, b), max(a, b))
            shell_edge_count = sum(1 for f in canonical_map.get(key, []) if f in face_set)
            if shell_edge_count == 1:
                return True
    return False


# ─── Strategy B: ray-casting outward test ────────────────────────────────────

def _raycast_is_inverted(
    vertices: np.ndarray,
    shell_face_indices: np.ndarray,
    faces: np.ndarray,
    n_rays: int = 7,
) -> Optional[bool]:
    """
    Cast rays from the centroid of several faces outward along their computed
    normal.  Count how many times each ray re-enters the shell.
    If most rays hit an even number of back-faces (0, 2, …) → normals point
    inward → shell is inverted.

    Returns True if the shell appears inside-out.
    Falls back to False (no-op) if the geometry is too degenerate.
    """
    if len(shell_face_indices) < 4:
        return None

    shell_verts_idx = shell_face_indices
    v0 = vertices[faces[shell_verts_idx, 0]]
    v1 = vertices[faces[shell_verts_idx, 1]]
    v2 = vertices[faces[shell_verts_idx, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    normals   = np.cross(v1 - v0, v2 - v0)
    norms_len = np.linalg.norm(normals, axis=1, keepdims=True)
    mask      = (norms_len[:, 0] > 1e-12)
    if not np.any(mask):
        return None
    normals[mask] /= norms_len[mask]

    # Pick up to n_rays sample faces spread across the shell
    step      = max(1, len(shell_verts_idx) // n_rays)
    sample_fi = shell_verts_idx[::step][:n_rays]

    epsilon   = 1e-6
    # All triangles in the shell for intersection testing
    all_v0 = vertices[faces[shell_verts_idx, 0]]
    all_v1 = vertices[faces[shell_verts_idx, 1]]
    all_v2 = vertices[faces[shell_verts_idx, 2]]

    inward_votes = 0
    valid_rays   = 0

    for fi_local, fi_global in enumerate(sample_fi):
        ray_orig = centroids[fi_local] + normals[fi_local] * epsilon
        ray_dir  = normals[fi_local]
        if np.linalg.norm(ray_dir) < 1e-12:
            continue

        # Möller–Trumbore against all shell triangles
        edge1 = all_v1 - all_v0
        edge2 = all_v2 - all_v0
        h     = np.cross(ray_dir, edge2)
        a_det = np.einsum('ij,ij->i', edge1, h)

        front_hits = 0   # ray exits through a front face (dot > 0)
        back_hits  = 0   # ray exits through a back face  (dot < 0)

        eps = 1e-8
        valid = np.abs(a_det) > eps
        if not np.any(valid):
            continue

        f_inv = np.zeros_like(a_det)
        f_inv[valid] = 1.0 / a_det[valid]
        s     = ray_orig - all_v0
        u     = f_inv * np.einsum('ij,ij->i', s, h)
        in_u  = valid & (u >= 0.0) & (u <= 1.0)

        q     = np.cross(s, edge1)
        v_bc  = f_inv * np.einsum('ij,j->i', q, ray_dir)
        in_v  = in_u & (v_bc >= 0.0) & ((u + v_bc) <= 1.0)

        t     = f_inv * np.einsum('ij,ij->i', edge2, q)
        hit   = in_v & (t > eps)

        if not np.any(hit):
            continue

        hit_normals = np.cross(edge1[hit], edge2[hit])
        dots        = np.einsum('ij,j->i', hit_normals, ray_dir)
        back_hits   = int(np.sum(dots < 0))
        front_hits  = int(np.sum(dots >= 0))

        valid_rays += 1
        # If more hits are back-facing, ray is exiting through back-faces →
        # normals point inward → inverted
        if back_hits > front_hits:
            inward_votes += 1

    if valid_rays == 0:
        return None
    return inward_votes > valid_rays // 2


# ─── Strategy C: boundary normal heuristic ───────────────────────────────────

def _boundary_normal_is_inverted(
    vertices: np.ndarray,
    shell_face_indices: List[int],
    faces: np.ndarray,
    canonical_map: Dict[Tuple[int,int], List[int]],
) -> bool:
    """
    For open shells: find the face with the highest Z-normal component on the
    topmost boundary edge and check whether that normal points up (+Z) or down.
    Very rough heuristic; returns False (no-op) when inconclusive.
    """
    face_set = set(shell_face_indices)
    shell_arr = np.array(shell_face_indices, dtype=np.int32)

    v0 = vertices[faces[shell_arr, 0]]
    v1 = vertices[faces[shell_arr, 1]]
    v2 = vertices[faces[shell_arr, 2]]
    cross   = np.cross(v1 - v0, v2 - v0)
    norms_l = np.linalg.norm(cross, axis=1, keepdims=True)
    ok      = norms_l[:, 0] > 1e-12
    if not np.any(ok):
        return False
    cross[ok] /= norms_l[ok]

    # Pick the face with the highest centroid Z among boundary faces
    centroids = (v0 + v1 + v2) / 3.0
    top_local = int(np.argmax(centroids[:, 2]))
    top_normal = cross[top_local]

    shell_centroid = centroids.mean(axis=0)
    to_center = shell_centroid - centroids[top_local]
    return bool(np.dot(top_normal, to_center) > 0)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API  —  Audit
# ─────────────────────────────────────────────────────────────────────────────

def check_winding_consistency(
    vertices: np.ndarray,
    faces: np.ndarray,
    stored_normals: Optional[np.ndarray] = None,
    normal_angle_tol_deg: float = 90.0,
) -> WindingAudit:
    """
    Pure audit: examine every manifold edge and every face's stored normal.
    Does NOT modify vertices or faces.

    Parameters
    ----------
    vertices            : (V, 3) float array
    faces               : (F, 3) int array
    stored_normals      : (F, 3) optional — normals read from the STL file.
                          If supplied, faces whose stored normal disagrees with
                          the computed normal by > normal_angle_tol_deg are
                          flagged in audit.stored_normal_flipped.
    normal_angle_tol_deg: threshold in degrees for stored-vs-computed flip test.

    Returns
    -------
    WindingAudit dataclass (no mesh data is changed).
    """
    audit = WindingAudit()
    if len(faces) == 0:
        return audit

    canonical_map, directed_map = _build_directed_edge_map(faces)

    # ── 1. Winding conflict detection ──────────────────────────────
    conflicted_face_set: Set[int] = set()
    manifold_edge_count  = 0

    for (a, b), flist in canonical_map.items():
        if len(flist) != 2:
            continue            # naked or non-manifold — skip winding check
        manifold_edge_count += 1

        # For a consistent pair: one face has directed edge (a→b),
        # the other has (b→a).  If BOTH have the same directed direction
        # (both a→b, or both b→a) → conflict.
        fwd_faces = directed_map.get((a, b), [])
        bwd_faces = directed_map.get((b, a), [])

        # Both share the edge in the same direction → conflict
        both_fwd = all(f in fwd_faces for f in flist)
        both_bwd = all(f in bwd_faces for f in flist)

        if both_fwd or both_bwd:
            audit.conflicted_edges.append((a, b))
            for fi in flist:
                conflicted_face_set.add(fi)

    audit.conflicted_faces = sorted(conflicted_face_set)
    if manifold_edge_count > 0:
        consistent = manifold_edge_count - len(audit.conflicted_edges)
        audit.consistency_pct = 100.0 * consistent / manifold_edge_count
    else:
        audit.consistency_pct = 100.0   # no manifold edges → nothing to check

    # ── 2. Stored normal vs computed normal ──────────────────────────
    if stored_normals is not None and len(stored_normals) == len(faces):
        computed = _recompute_normals(vertices, faces)
        cos_tol  = np.cos(np.radians(normal_angle_tol_deg))
        dots     = np.einsum('ij,ij->i', computed, stored_normals)
        # Normalise stored normals in case they are not unit vectors
        sn_len   = np.linalg.norm(stored_normals, axis=1)
        valid    = sn_len > 1e-12
        dots[valid]  /= sn_len[valid]
        dots[~valid]  = 1.0              # treat zero-length stored normals as ok
        flipped_mask  = dots < cos_tol
        audit.stored_normal_flipped = list(np.where(flipped_mask)[0])

    # ── 3. Connected shells + inversion status ────────────────────────
    shells = _connected_shells(faces, canonical_map)
    audit.shells = shells

    for shell in shells:
        shell_arr  = np.array(shell, dtype=np.int32)
        is_open    = _shell_has_naked_edges(shell, canonical_map, faces)
        audit.shells_open.append(is_open)

        if is_open:
            # Signed volume is unreliable for open meshes
            inverted = False
        else:
            vol = _signed_volume(vertices, faces[shell_arr])
            inverted = bool(vol < 0.0)
        audit.shells_inverted.append(inverted)

    return audit


# ─────────────────────────────────────────────────────────────────────────────
#  Public API  —  Repair
# ─────────────────────────────────────────────────────────────────────────────

def repair_winding_order(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    config: RepairConfig,
    log: RepairLog,
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fix winding-order inconsistencies and inside-out shells.

    Algorithm (applied per connected shell):
      Step 1 – BFS consistency pass:
               Propagate winding from a seed face so every manifold edge is
               traversed in opposite directions by its two neighbours.
               (This is the same logic as improved_orientation but scoped
               per-shell so shells don't "infect" each other.)

      Step 2 – Outward orientation (three strategies):
        A. Signed Volume  — closed shells: if volume < 0 flip the whole shell.
        B. Ray Casting    — open/ambiguous: cast rays along face normals and
                            count re-entries to determine inside-out status.
        C. Boundary Heuristic — last resort for thin / planar open patches.

    Parameters
    ----------
    vertices  : (V, 3)
    faces     : (F, 3)  — will be modified in-place copy
    normals   : (F, 3)  — recomputed from winding at the end
    config    : RepairConfig
    log       : RepairLog  (faces_reoriented is updated)
    verbose   : if True, print per-shell decisions to stdout

    Returns
    -------
    (vertices, faces, normals) — vertices unchanged, faces/normals may differ.
    """
    if len(faces) == 0:
        return vertices, faces, normals

    faces = faces.copy()
    canonical_map, directed_map = _build_directed_edge_map(faces)

    # Pre-build adjacency (manifold edges only)
    adj: List[List[Tuple[int, Tuple[int,int]]]] = [[] for _ in range(len(faces))]
    for (a, b), flist in canonical_map.items():
        if len(flist) != 2:
            continue
        fa, fb = flist[0], flist[1]
        adj[fa].append((fb, (a, b)))
        adj[fb].append((fa, (a, b)))

    def _directed_edge_in_face(
        face: np.ndarray, u: int, v: int
    ) -> int:
        """Return +1 if face traverses u→v, -1 if v→u, 0 if not found."""
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            if a == u and b == v:
                return 1
            if a == v and b == u:
                return -1
        return 0

    shells     = _connected_shells(faces, canonical_map)
    total_flip = 0

    for s_idx, shell in enumerate(shells):
        shell_set = set(shell)
        shell_arr = np.array(shell, dtype=np.int32)

        # ── Step 1: BFS winding consistency within this shell ────────
        visited = {shell[0]: True}
        queue   = deque([shell[0]])
        shell_flipped = 0

        while queue:
            fi = queue.popleft()
            for nb, (a, b) in adj[fi]:
                if nb not in shell_set:
                    continue
                if nb in visited:
                    continue
                # Check traversal directions
                dir_fi = _directed_edge_in_face(faces[fi], a, b)
                dir_nb = _directed_edge_in_face(faces[nb], a, b)
                if dir_fi != 0 and dir_fi == dir_nb:
                    # Same direction → flip neighbour
                    faces[nb, [1, 2]] = faces[nb, [2, 1]]
                    shell_flipped += 1
                visited[nb] = True
                queue.append(nb)

        # Rebuild directed_map for this shell after local BFS
        # (needed for outward-orientation strategies)

        # ── Step 2: Outward orientation ──────────────────────────────
        is_open = _shell_has_naked_edges(shell, canonical_map, faces)
        strategy_used = "none"
        invert_shell  = False

        if not is_open:
            # Strategy A – Signed Volume
            vol = _signed_volume(vertices, faces[shell_arr])
            if abs(vol) > 1e-20:
                invert_shell  = vol < 0.0
                strategy_used = "A (signed-volume)"
            else:
                # Volume is essentially zero (flat / degenerate shell)
                # fall through to Strategy B
                invert_result = _raycast_is_inverted(
                    vertices, shell_arr, faces, n_rays=9)
                strategy_used = "B (ray-cast, flat closed)"
                if invert_result is None and len(shell) < 20:
                    invert_shell  = _boundary_normal_is_inverted(
                        vertices, shell, faces, canonical_map)
                    strategy_used = "C (boundary-heuristic)"
                else:
                    invert_shell = invert_result or False
        else:
            # Strategy B – Ray Casting for open shell
            invert_result = _raycast_is_inverted(
                vertices, shell_arr, faces, n_rays=9)
            strategy_used = "B (ray-cast, open)"

            # Strategy C fallback: if ray-cast is inconclusive and shell
            # is very small or planar, use boundary-normal heuristic.
            if invert_result is None and len(shell) < 20:
                invert_shell  = _boundary_normal_is_inverted(
                    vertices, shell, faces, canonical_map)
                strategy_used = "C (boundary-heuristic)"
            else:
                invert_shell  = invert_result or False

        if invert_shell:
            faces[shell_arr] = faces[shell_arr][:, [0, 2, 1]]
            shell_flipped   += len(shell)

        total_flip += shell_flipped

        if verbose:
            open_tag = "open" if is_open else "closed"
            flip_tag = "flipped" if invert_shell else "ok"
            print(
                f"    Shell {s_idx+1:>3d}: {len(shell):>6d} faces  "
                f"[{open_tag}]  strategy {strategy_used}  → {flip_tag}"
                f"  (consistency BFS fixed {shell_flipped} faces)"
            )

    log.faces_reoriented += total_flip

    # ── Option B: Greedy local flip for residual non-orientable conflicts ──
    # After BFS, some edges may still be conflicted (non-orientable / Möbius).
    # Instead of DELETING these faces (which creates holes), we try flipping
    # the minority face at each conflicted edge.  If a conflict is truly
    # unresolvable, we leave it — a few wrong normals are cosmetic, while
    # holes are structural and prevent watertightness.
    for _flip_pass in range(10):
        final_audit = check_winding_consistency(
            vertices, faces,
            stored_normals=None,
        )
        if not final_audit.conflicted_edges:
            break

        # Count how many conflicted edges each face participates in
        conflict_count: Dict[int, int] = defaultdict(int)
        _, directed_map = _build_directed_edge_map(faces)
        for (a, b) in final_audit.conflicted_edges:
            key = (min(a, b), max(a, b))
            flist = canonical_map.get(key, [])
            if len(flist) != 2:
                continue
            for fi in flist:
                conflict_count[fi] += 1

        # Flip the face with MORE conflicted edges at each conflict
        flipped_this_pass: Set[int] = set()
        for (a, b) in final_audit.conflicted_edges:
            key = (min(a, b), max(a, b))
            flist = canonical_map.get(key, [])
            if len(flist) != 2:
                continue
            fa, fb = flist[0], flist[1]
            if fa in flipped_this_pass or fb in flipped_this_pass:
                continue
            # Flip the face with more conflicts (it's the "odd one out")
            if conflict_count[fa] >= conflict_count[fb]:
                faces[fa, [1, 2]] = faces[fa, [2, 1]]
                flipped_this_pass.add(fa)
                total_flip += 1
            else:
                faces[fb, [1, 2]] = faces[fb, [2, 1]]
                flipped_this_pass.add(fb)
                total_flip += 1

        if not flipped_this_pass:
            break
        # Rebuild canonical map for next pass
        canonical_map, _ = _build_directed_edge_map(faces)

    # Report any remaining conflicts (but never delete faces)
    remaining_audit = check_winding_consistency(vertices, faces, stored_normals=None)
    if remaining_audit.conflicted_edges and verbose:
        print(f"  [i] {len(remaining_audit.conflicted_edges)} non-orientable edge(s) remain "
              f"(cosmetic only — faces preserved to avoid creating holes)")

    normals = _recompute_normals(vertices, faces)
    return vertices, faces, normals


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: audit then repair in one call
# ─────────────────────────────────────────────────────────────────────────────

def audit_and_repair_winding(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    config: RepairConfig,
    log: RepairLog,
    *,
    stored_normals: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, WindingAudit]:
    """
    Run check_winding_consistency() then repair_winding_order() and return
    both the repaired mesh AND the pre-repair audit.
    """
    audit = check_winding_consistency(
        vertices, faces,
        stored_normals=stored_normals,
        normal_angle_tol_deg=config.normal_angle_tol_deg,
    )

    if verbose:
        print(audit.summary())

    vertices, faces, normals = repair_winding_order(
        vertices, faces, normals, config, log, verbose=verbose)

    # Note: Any excision that happened inside repair_winding_order means
    # the returned faces array is shorter. The original pre-repair 'audit' 
    # remains unchanged (referencing the original face indices).
    return vertices, faces, normals, audit


# ─────────────────────────────────────────────────────────────────────────────
#  CLI  — python repair_winding.py input.stl [output.stl]
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os, struct, time

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python repair_winding.py <input.stl> [output.stl]")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 else None

    if not os.path.exists(inp):
        print(f"[ERROR] File not found: {inp}")
        sys.exit(1)

    # Lazy import — only needed when run as a script
    from src.core.mesh_analyzer import MeshData, MeshAnalyzer
    from src.core.repair_config import RepairConfig, RepairLog

    cfg  = RepairConfig()
    t0   = time.perf_counter()
    mesh = MeshData.from_stl(inp, cfg.merge_tolerance)
    print(f"  Loaded  : {inp}  ({len(mesh.faces)} faces, {len(mesh.vertices)} vertices)")

    rlog = RepairLog()
    verts, faces, norms, audit = audit_and_repair_winding(
        mesh.vertices, mesh.faces, mesh.normals,
        cfg, rlog,
        stored_normals=mesh.normals,
        verbose=True,
    )

    print(f"\n  Faces reoriented : {rlog.faces_reoriented}")
    print(f"  Time             : {(time.perf_counter()-t0)*1000:.1f} ms")

    if out:
        # Write a minimal binary STL
        with open(out, "wb") as f:
            hdr = b"Winding-repaired by repair_winding.py"
            f.write((hdr + b"\x00" * 80)[:80])
            f.write(struct.pack("<I", len(faces)))
            for fi, face in enumerate(faces):
                n = norms[fi]
                f.write(struct.pack("<3f", *n))
                for vi in face:
                    f.write(struct.pack("<3f", *verts[vi]))
                f.write(struct.pack("<H", 0))
        print(f"  Written : {out}")
