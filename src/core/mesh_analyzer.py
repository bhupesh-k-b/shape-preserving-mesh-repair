"""
================================================================================
  MESH REPAIR PIPELINE  —  Phase 1: Analyzer
  Detects: Naked Edges, Planar/Non-Planar Holes, Non-Manifold Edges,
           Inverted Normals, Duplicate Faces, Degenerate Faces, Disjoint Shells
================================================================================
"""

import numpy as np
from stl import mesh as stl_mesh
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
import time
import os
import struct


# ─────────────────────────────────────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """Container for all analysis results."""
    # Raw mesh stats
    vertex_count: int = 0
    face_count: int = 0
    edge_count: int = 0
    bounding_box: Optional[np.ndarray] = None

    # Defects
    naked_edges: List[Tuple] = field(default_factory=list)
    planar_holes: List[List[Tuple]] = field(default_factory=list)       # list of edge-loops
    non_planar_holes: List[List[Tuple]] = field(default_factory=list)   # list of edge-loops
    non_manifold_edges: List[Tuple] = field(default_factory=list)
    inverted_normal_faces: List[int] = field(default_factory=list)
    duplicate_face_groups: List[List[int]] = field(default_factory=list)
    degenerate_faces: List[int] = field(default_factory=list)
    disjoint_shells: List[List[int]] = field(default_factory=list)      # list of face-index groups

    # Timing
    analysis_time_ms: float = 0.0

    @property
    def is_watertight(self) -> bool:
        return self.face_count > 0 and not self.naked_edges and not self.non_manifold_edges

    @property
    def hole_count(self) -> int:
        return len(self.planar_holes) + len(self.non_planar_holes)

    @property
    def shell_count(self) -> int:
        return len(self.disjoint_shells)

    def summary(self) -> str:
        lines = [
            "━" * 60,
            "  MESH ANALYSIS REPORT",
            "━" * 60,
            f"  Vertices           : {self.vertex_count}",
            f"  Faces              : {self.face_count}",
            f"  Edges (unique)     : {self.edge_count}",
            f"  Bounding Box       : {self._bbox_str()}",
            "",
            "  ── Defects ──────────────────────────────────────────",
            f"  Naked Edges        : {len(self.naked_edges)}"
            + ("  ✓" if not self.naked_edges else "  ✗ (mesh has holes)"),
            f"  Planar Holes       : {len(self.planar_holes)}",
            f"  Non-Planar Holes   : {len(self.non_planar_holes)}",
            f"  Non-Manifold Edges : {len(self.non_manifold_edges)}"
            + ("  ✓" if not self.non_manifold_edges else "  ✗"),
            f"  Inverted Normals   : {len(self.inverted_normal_faces)} face(s)",
            f"  Duplicate Faces    : {sum(len(g) - 1 for g in self.duplicate_face_groups)} extra face(s)"
            + f"  ({len(self.duplicate_face_groups)} group(s))",
            f"  Degenerate Faces   : {len(self.degenerate_faces)}",
            f"  Disjoint Shells    : {len(self.disjoint_shells)}",
            "",
            f"  Analysis Time      : {self.analysis_time_ms:.1f} ms",
            "━" * 60,
        ]
        return "\n".join(lines)

    def _bbox_str(self) -> str:
        if self.bounding_box is None:
            return "N/A"
        lo, hi = self.bounding_box
        return f"[{lo[0]:.2f},{lo[1]:.2f},{lo[2]:.2f}] → [{hi[0]:.2f},{hi[1]:.2f},{hi[2]:.2f}]"


# ─────────────────────────────────────────────────────────────────────────────
#  STL Loader  →  flat numpy arrays
# ─────────────────────────────────────────────────────────────────────────────

class MeshData:
    """
    Holds the raw mesh as flat numpy arrays.
    vertices : (V, 3)  — unique vertices, deduplicated
    faces    : (F, 3)  — indices into vertices
    normals  : (F, 3)  — stored face normals from file
    """
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray, units=None):
        self.vertices = np.asarray(vertices, dtype=np.float64).reshape(-1,3)
        self.faces    = np.asarray(faces, dtype=np.int64).reshape(-1,3)
        self.normals  = np.asarray(normals, dtype=np.float64).reshape(-1,3)
        self.units = units

    @staticmethod
    def from_stl(path: str, merge_tolerance: float = 0.0) -> "MeshData":
        """Raw import: merge only exactly equal coordinates, irrespective of scale.

        The legacy tolerance argument is retained for API compatibility. Spatial
        welding belongs to a later, preservation-checked repair stage.
        """
        with open(path,'rb') as handle:
            header=handle.read(1024)
        size=os.path.getsize(path)
        binary_count=struct.unpack('<I',header[80:84])[0] if len(header)>=84 else None
        binary_ok=binary_count is not None and size==84+50*binary_count
        ascii_ok=header.lstrip().lower().startswith(b'solid') and b'facet' in header.lower()
        if not binary_ok and not ascii_ok:
            raise ValueError('Malformed or truncated STL')
        raw = stl_mesh.Mesh.from_file(path, calculate_normals=False)

        # raw.vectors: (F, 3, 3) — three vertices per face
        tri_verts = raw.vectors.reshape(-1, 3).astype(np.float64)   # (F*3, 3)
        normals   = raw.normals.astype(np.float64)                   # (F, 3)

        if not len(tri_verts):
            raise ValueError('Empty STL contains no triangles')
        if not np.all(np.isfinite(tri_verts)):
            raise ValueError('STL contains nonfinite coordinates')
        vertices,indices=np.unique(tri_verts,axis=0,return_inverse=True)
        faces    = indices.reshape(-1, 3)                             # (F, 3)

        return MeshData(vertices, faces, normals)


# ─────────────────────────────────────────────────────────────────────────────
#  Edge Utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_edge_map(faces: np.ndarray) -> Dict[Tuple[int,int], List[int]]:
    """
    Returns edge_map: canonical_edge → [face_index, ...]
    Canonical: (min_v, max_v)
    """
    edge_map: Dict[Tuple[int,int], List[int]] = defaultdict(list)
    for fi, face in enumerate(faces):
        for j in range(3):
            a, b = int(face[j]), int(face[(j + 1) % 3])
            key  = (min(a, b), max(a, b))
            edge_map[key].append(fi)
    return edge_map


# ─────────────────────────────────────────────────────────────────────────────
#  Individual Checks
# ─────────────────────────────────────────────────────────────────────────────

def check_naked_and_nonmanifold(
    edge_map: Dict[Tuple[int,int], List[int]]
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Naked edges     → appear in exactly 1 face
    Non-manifold    → appear in 3+ faces
    """
    naked      = []
    non_manif  = []
    for edge, faces in edge_map.items():
        n = len(faces)
        if n == 1:
            naked.append(edge)
        elif n >= 3:
            non_manif.append(edge)
    return naked, non_manif


def _build_naked_adjacency(naked_edges: List[Tuple]) -> Dict[int, List[int]]:
    """Build vertex → neighbouring naked-edge vertices map."""
    adj: Dict[int, List[int]] = defaultdict(list)
    for a, b in naked_edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _trace_boundary_loops(naked_edges: List[Tuple]) -> List[List[Tuple]]:
    """
    Walk naked edges into closed boundary loops.
    Returns a list of loops; each loop is an ordered list of edges.
    """
    adj      = _build_naked_adjacency(naked_edges)
    edge_set = set(map(lambda e: (min(e), max(e)), naked_edges))
    visited_edges: Set[Tuple[int,int]] = set()
    loops: List[List[Tuple]] = []

    for start_a, start_b in edge_set:
        canonical = (min(start_a, start_b), max(start_a, start_b))
        if canonical in visited_edges:
            continue

        loop   = []
        cur    = start_a
        prev   = None
        closed = False

        for _ in range(len(edge_set) + 1):
            neighbors = [v for v in adj[cur] if v != prev]
            if not neighbors:
                break

            # pick an unvisited neighbor if possible
            next_v = None
            for nb in neighbors:
                e = (min(cur, nb), max(cur, nb))
                if e not in visited_edges:
                    next_v = nb
                    break
            if next_v is None:
                break

            e = (min(cur, next_v), max(cur, next_v))
            visited_edges.add(e)
            loop.append((cur, next_v))
            prev  = cur
            cur   = next_v

            if cur == start_a and len(loop) > 1:
                closed = True
                break

        if loop:
            loops.append(loop)

    return loops


def _is_planar(loop_edges: List[Tuple], vertices: np.ndarray,
               tol: float = 1e-4) -> bool:
    """
    Return True if all vertices of a loop are coplanar.
    Uses SVD on centered point cloud: smallest singular value ≈ 0 → planar.
    """
    verts_idx = list({v for e in loop_edges for v in e})
    if len(verts_idx) < 3:
        return True

    pts    = vertices[verts_idx]
    center = pts.mean(axis=0)
    _, sv, _ = np.linalg.svd(pts - center)

    # Normalise smallest singular value by largest to get relative flatness
    ratio = sv[-1] / (sv[0] + 1e-12)
    return float(ratio) < tol


def classify_holes(
    naked_edges: List[Tuple],
    vertices: np.ndarray
) -> Tuple[List[List[Tuple]], List[List[Tuple]]]:
    """Split naked-edge loops into planar and non-planar holes."""
    loops      = _trace_boundary_loops(naked_edges)
    planar     = []
    non_planar = []
    for loop in loops:
        if _is_planar(loop, vertices):
            planar.append(loop)
        else:
            non_planar.append(loop)
    return planar, non_planar


def check_degenerate_faces(
    faces: np.ndarray,
    vertices: np.ndarray,
    area_tol: float = 0.0
) -> List[int]:
    """
    A face is degenerate if:
      - two or more vertex indices are identical, OR
      - computed area < area_tol  (collinear vertices)
    """
    degenerate = []
    for fi, face in enumerate(faces):
        a, b, c = face
        # Repeated index
        if a == b or b == c or a == c:
            degenerate.append(fi)
            continue
        # Zero area
        va, vb, vc = vertices[a], vertices[b], vertices[c]
        area = 0.5 * np.linalg.norm(np.cross(vb - va, vc - va))
        if area <= area_tol:
            degenerate.append(fi)
    return degenerate


def check_duplicate_faces(faces: np.ndarray) -> List[List[int]]:
    """
    Faces are duplicates if they share the same 3 vertices (order-independent).
    Returns groups of face indices; each group has 2+ duplicates.
    """
    canonical: Dict[Tuple, List[int]] = defaultdict(list)
    for fi, face in enumerate(faces):
        key = tuple(sorted(face))
        canonical[key].append(fi)
    return [g for g in canonical.values() if len(g) > 1]


def check_inverted_normals(
    faces: np.ndarray,
    vertices: np.ndarray,
    stored_normals: np.ndarray
) -> List[int]:
    from src.routines.repair_winding import check_winding_consistency
    
    audit = check_winding_consistency(vertices, faces, stored_normals)
    inverted = set(audit.conflicted_faces)
    
    for i, inv in enumerate(audit.shells_inverted):
        if inv:
            inverted.update(audit.shells[i])
            
    if audit.stored_normal_flipped:
        inverted.update(audit.stored_normal_flipped)

    return sorted(list(inverted))

def check_disjoint_shells(
    faces: np.ndarray,
    edge_map: Dict[Tuple[int,int], List[int]]
) -> List[List[int]]:
    """
    BFS / Union-Find on face adjacency to identify connected components (shells).
    Returns a list of shells; each shell is a list of face indices.
    """
    n_faces = len(faces)
    if n_faces == 0:
        return []

    # Build face adjacency from edge_map
    face_adj: List[Set[int]] = [set() for _ in range(n_faces)]
    for faces_sharing_edge in edge_map.values():
        for i in range(len(faces_sharing_edge)):
            for j in range(i + 1, len(faces_sharing_edge)):
                fi, fj = faces_sharing_edge[i], faces_sharing_edge[j]
                face_adj[fi].add(fj)
                face_adj[fj].add(fi)

    # BFS
    visited = np.zeros(n_faces, dtype=bool)
    shells  = []

    for start in range(n_faces):
        if visited[start]:
            continue
        shell   = []
        queue   = [start]
        visited[start] = True
        while queue:
            fi = queue.pop()
            shell.append(fi)
            for nb in face_adj[fi]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        shells.append(shell)

    return shells


# ─────────────────────────────────────────────────────────────────────────────
#  Main Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class MeshAnalyzer:
    """
    Run all defect checks on a loaded MeshData and return an AnalysisResult.
    """

    def __init__(self, merge_tolerance: float = 1e-6):
        self.merge_tolerance = merge_tolerance

    def analyze(self, path: str) -> Tuple["MeshData", AnalysisResult]:
        t0     = time.perf_counter()
        result = AnalysisResult()

        # ── Load ─────────────────────────────────────────────────
        mesh = MeshData.from_stl(path, self.merge_tolerance)

        result.vertex_count = len(mesh.vertices)
        result.face_count   = len(mesh.faces)
        result.bounding_box = (mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)) if len(mesh.vertices) else None

        # ── Build edge map ────────────────────────────────────────
        edge_map = build_edge_map(mesh.faces)
        result.edge_count = len(edge_map)

        # ── Naked + Non-manifold ──────────────────────────────────
        naked, non_manif = check_naked_and_nonmanifold(edge_map)
        result.naked_edges         = naked
        result.non_manifold_edges  = non_manif

        # ── Hole classification ───────────────────────────────────
        if naked:
            planar, non_planar = classify_holes(naked, mesh.vertices)
            result.planar_holes     = planar
            result.non_planar_holes = non_planar

        # ── Degenerate faces ──────────────────────────────────────
        result.degenerate_faces = check_degenerate_faces(mesh.faces, mesh.vertices)

        # ── Duplicate faces ───────────────────────────────────────
        result.duplicate_face_groups = check_duplicate_faces(mesh.faces)

        # ── Inverted normals ──────────────────────────────────────
        result.inverted_normal_faces = check_inverted_normals(
            mesh.faces, mesh.vertices, mesh.normals
        )

        # ── Disjoint shells ───────────────────────────────────────
        shells = check_disjoint_shells(mesh.faces, edge_map)
        result.disjoint_shells = shells

        result.analysis_time_ms = (time.perf_counter() - t0) * 1000
        return mesh, result


# ─────────────────────────────────────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_defects(mesh_data: MeshData, report: AnalysisResult):
    """
    Open up a 3D window highlighting mesh defects.
    Uses 'trimesh' to ensure maximum compatibility even on bleeding-edge Python versions (e.g. 3.14).
    """
    try:
        import trimesh
    except ImportError:
        print("\n  [!] Visualization requires Trimesh and Pyglet.")
        print("      Install them via: pip install trimesh pyglet")
        return

    print("\n  [Visualization] Opening Trimesh 3D Viewer...")
    print("  - Base Mesh      : Light Gray")
    print("  - Naked Edges    : Bright Red lines")
    print("  - Planar Holes   : Bright Green lines")
    print("  - Non-Planar     : Bright Orange lines")
    print("  - Non-Manifold   : Bright Blue lines")
    print("  - Inverted Faces : Solid Yellow faces")

    try:
        scene = trimesh.Scene()
        
        # 1. Base Mesh (Gray)
        base_mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)
        base_mesh.visual.face_colors = [200, 200, 200, 255]
        scene.add_geometry(base_mesh)

        # 2. Naked Edges (Red Lines)
        if report.naked_edges:
            edges = np.array(report.naked_edges)
            # trimesh load_path with (N, 2, 3) arrays extracts discrete segments
            verts = mesh_data.vertices[edges]
            path = trimesh.load_path(verts)
            # Apply color to all line entities inside the path
            for entity in path.entities:
                entity.color = [255, 0, 0, 255]
            scene.add_geometry(path)

        # 2b. Planar Holes (Bright Green Lines)
        if report.planar_holes:
            for loop in report.planar_holes:
                edges = np.array(loop)
                verts = mesh_data.vertices[edges]
                path = trimesh.load_path(verts)
                for entity in path.entities:
                    entity.color = [0, 255, 0, 255]
                scene.add_geometry(path)

        # 2c. Non-Planar Holes (Bright Orange Lines)
        if report.non_planar_holes:
            for loop in report.non_planar_holes:
                edges = np.array(loop)
                verts = mesh_data.vertices[edges]
                path = trimesh.load_path(verts)
                for entity in path.entities:
                    entity.color = [255, 165, 0, 255]
                scene.add_geometry(path)

        # 3. Non-Manifold Edges (Blue Lines)
        if report.non_manifold_edges:
            edges = np.array(report.non_manifold_edges)
            verts = mesh_data.vertices[edges]
            path = trimesh.load_path(verts)
            for entity in path.entities:
                entity.color = [0, 0, 255, 255]
            scene.add_geometry(path)

        # 4. Inverted Normal Faces (Yellow Faces)
        if report.inverted_normal_faces:
            inv_faces = [mesh_data.faces[fi] for fi in report.inverted_normal_faces]
            if inv_faces:
                inv_mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=inv_faces, process=False)
                inv_mesh.visual.face_colors = [255, 255, 0, 255]
                scene.add_geometry(inv_mesh)

        scene.show(viewer='gl')

    except Exception as e:
        print(f"\n  [!] Trimesh viewer failed to launch: {e}")
        import traceback
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys, os

    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(description="Mesh Defect Analyzer")
    parser.add_argument("input", help="Path to input STL file")
    parser.add_argument("-v", "--visualize", action="store_true", help="Open a 3D window to visualize defects")
    
    args = parser.parse_args()
    stl_path = args.input

    if not os.path.exists(stl_path):
        print(f"[ERROR] File not found: {stl_path}")
        sys.exit(1)

    print(f"\n  Loading: {stl_path}\n")
    analyzer = MeshAnalyzer(merge_tolerance=1e-6)
    mesh_data, report = analyzer.analyze(stl_path)

    print(report.summary())

    # ── Detailed defect output ────────────────────────────────────
    if report.naked_edges:
        print(f"\n  Naked Edges (first 10 of {len(report.naked_edges)}):")
        for e in report.naked_edges[:10]:
            v0, v1 = mesh_data.vertices[e[0]], mesh_data.vertices[e[1]]
            print(f"    Edge {e}  |  {v0}  ↔  {v1}")

    if report.non_manifold_edges:
        print(f"\n  Non-Manifold Edges (first 10 of {len(report.non_manifold_edges)}):")
        for e in report.non_manifold_edges[:10]:
            print(f"    Edge {e}")

    if report.planar_holes:
        print(f"\n  Planar Holes ({len(report.planar_holes)} found):")
        for i, loop in enumerate(report.planar_holes):
            print(f"    Hole {i+1}: {len(loop)} boundary edges")

    if report.non_planar_holes:
        print(f"\n  Non-Planar Holes ({len(report.non_planar_holes)} found):")
        for i, loop in enumerate(report.non_planar_holes):
            print(f"    Hole {i+1}: {len(loop)} boundary edges")

    if report.degenerate_faces:
        print(f"\n  Degenerate Faces (first 10 of {len(report.degenerate_faces)}):")
        for fi in report.degenerate_faces[:10]:
            print(f"    Face {fi}: vertices {mesh_data.faces[fi]}")

    if report.duplicate_face_groups:
        print(f"\n  Duplicate Face Groups:")
        for grp in report.duplicate_face_groups[:10]:
            print(f"    {grp}")

    if report.inverted_normal_faces:
        print(f"\n  Inverted Normal Faces (first 10 of {len(report.inverted_normal_faces)}):")
        for fi in report.inverted_normal_faces[:10]:
            print(f"    Face {fi}: stored={mesh_data.normals[fi]}")

    if len(report.disjoint_shells) > 1:
        print(f"\n  Disjoint Shells ({len(report.disjoint_shells)} shells):")
        for i, shell in enumerate(report.disjoint_shells):
            print(f"    Shell {i+1}: {len(shell)} faces")

    if args.visualize:
        visualize_defects(mesh_data, report)
