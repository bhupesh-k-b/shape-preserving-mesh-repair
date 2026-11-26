"""
mesh_pipeline_tests.py
======================
Comprehensive test suite for the Mesh Repair Pipeline.

Run:
    python mesh_pipeline_tests.py          # all tests
    python mesh_pipeline_tests.py -v       # verbose
    python mesh_pipeline_tests.py -k degenerate   # filter by name

Each test function is self-contained — it builds a synthetic mesh in numpy,
runs one pipeline module, and asserts the expected postcondition.
No STL files or trimesh are required.

Test coverage
─────────────
UNIT  1  repair_degenerate_faces  — zero-area removal
UNIT  2  repair_degenerate_faces  — repeated-vertex removal
UNIT  3  repair_duplicate_faces   — canonical dedup
UNIT  4  global_vertex_weld       — merge vertices within tolerance
UNIT  5  global_vertex_weld       — distant vertices untouched
UNIT  6  repair_non_manifold      — 3-face edge → excise worst face
UNIT  7  repair_non_manifold_vertices — pinch-point cracking
UNIT  8  separate_to_manifold     — non-manifold edge cracking
UNIT  9  improved_orientation     — single-shell consistent winding
UNIT 10  improved_orientation     — inside-out shell flipped outward
UNIT 11  check_winding_consistency — detects conflicted edges
UNIT 12  repair_winding_order     — resolves Möbius-like conflict
UNIT 13  crack_mobius_strips      — duplicates vertices on conflict edge
UNIT 14  repair_planar_holes      — fills a square hole (4 boundary edges)
UNIT 15  repair_non_planar_holes  — falls back to centroid fan
UNIT 16  bary_smooth              — vertex positions move toward barycentre
UNIT 17  bary_smooth              — boundary verts stay on boundary
UNIT 18  filter_small_shells      — tiny shell removed by area threshold
UNIT 19  _recompute_normals       — normals are unit vectors
UNIT 20  _signed_volume           — positive volume for outward-facing cube
UNIT 21  _compact_mesh            — unused vertices stripped
UNIT 22  build_edge_map           — correct face count per edge
UNIT 23  check_naked_and_nonmanifold — correct naked / NM classification
UNIT 24  _face_area               — area of right triangle
UNIT 25  _face_normal             — normal of known triangle

INTEGRATION 1  degenerate → duplicate → normals pipeline
INTEGRATION 2  weld + hole fill produces watertight mesh
INTEGRATION 3  non-manifold → normals → weld sequence
INTEGRATION 4  full pipeline on a broken cube (open + NM + inverted faces)
"""

import os
import sys
import math
import pytest
import traceback
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
#  Path Injection: Ensure 'src' package is discoverable when run as a script.
# ─────────────────────────────────────────────────────────────────────────────
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)



# ─────────────────────────────────────────────────────────────────────────────
#  Helpers to build synthetic meshes
# ─────────────────────────────────────────────────────────────────────────────

def _unit_cube_mesh():
    """
    12-triangle unit cube, outward normals, no defects.
    Vertices: 8 corners of [0,1]^3
    """
    v = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
    ], dtype=np.float64)
    f = np.array([
        [0,2,1],[0,3,2],   # -Z
        [4,5,6],[4,6,7],   # +Z
        [0,1,5],[0,5,4],   # -Y
        [2,3,7],[2,7,6],   # +Y
        [0,4,7],[0,7,3],   # -X
        [1,2,6],[1,6,5],   # +X
    ], dtype=np.int32)
    normals = _recompute_normals(v, f)
    return v, f, normals


def _open_cube_minus_cap():
    """Cube missing one face (cap at z=0). 10 triangles, 4 naked edges."""
    v, f, n = _unit_cube_mesh()
    # Drop the two -Z faces (indices 0 and 1)
    keep = np.array([i for i in range(len(f)) if i not in (0, 1)], dtype=int)
    return v, f[keep], n[keep]


def _recompute_normals(v, f):
    if len(f) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    c = np.cross(e1, e2)
    l = np.linalg.norm(c, axis=1, keepdims=True)
    l = np.where(l < 1e-12, 1.0, l)
    return c / l


def _build_edge_map(faces):
    em = defaultdict(list)
    for fi, face in enumerate(faces):
        for j in range(3):
            a, b = int(face[j]), int(face[(j+1)%3])
            em[(min(a,b), max(a,b))].append(fi)
    return em


def _is_watertight(faces):
    em = _build_edge_map(faces)
    return all(len(v) == 2 for v in em.values())


# ─────────────────────────────────────────────────────────────────────────────
#  Test runner
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  Import pipeline modules (lazy — give good error if missing)
# ─────────────────────────────────────────────────────────────────────────────

import importlib, types

def _try_import(module_path):
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        m = types.ModuleType(module_path)
        m._import_error = str(e)
        return m

repair_config              = _try_import("src.core.repair_config")
repair_utils               = _try_import("src.core.repair_utils")
mesh_analyzer              = _try_import("src.core.mesh_analyzer")
repair_degenerates         = _try_import("src.routines.repair_degenerates")
repair_welding             = _try_import("src.routines.repair_welding")
repair_nonmanifold         = _try_import("src.routines.repair_nonmanifold")
repair_normals_mod         = _try_import("src.routines.repair_normals")
repair_winding_mod         = _try_import("src.routines.repair_winding")
repair_mobius_mod          = _try_import("src.routines.repair_mobius")
repair_holes_mod           = _try_import("src.routines.repair_holes")
repair_smooth_mod          = _try_import("src.routines.repair_smooth")
repair_shell_mod           = _try_import("src.routines.repair_shell_union")
repair_topology_mod        = _try_import("src.routines.repair_topology_separation")

def _cfg():
    return repair_config.RepairConfig()

def _log():
    return repair_config.RepairLog()


# ═════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
# ═════════════════════════════════════════════════════════════════════════════

# ── 1 ── repair_degenerate_faces — zero-area ──────────────────────────────────
def test_UNIT_01_degenerate_zero_area():
    v = np.array([[0,0,0],[1,0,0],[2,0,0],[0,1,0],[1,1,0]], dtype=np.float64)
    # face 0: collinear (zero area), face 1: normal triangle
    f = np.array([[0,1,2],[0,3,4]], dtype=np.int32)
    n = _recompute_normals(v, f)
    cfg, log = _cfg(), _log()
    f2, n2 = repair_degenerates.repair_degenerate_faces(v, f, n, cfg, log)
    assert log.degenerate_removed == 1, f"Expected 1 removed, got {log.degenerate_removed}"
    assert len(f2) == 1, f"Expected 1 face remaining, got {len(f2)}"
    assert np.all(f2[0] == f[1]), "Wrong face kept"


# ── 2 ── repair_degenerate_faces — repeated vertex ────────────────────────────
def test_UNIT_02_degenerate_repeated_vertex():
    v = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float64)
    f = np.array([[0,0,2],[0,1,2]], dtype=np.int32)  # face0 has repeated vertex 0
    n = _recompute_normals(v, f)
    cfg, log = _cfg(), _log()
    f2, n2 = repair_degenerates.repair_degenerate_faces(v, f, n, cfg, log)
    assert log.degenerate_removed >= 1
    assert len(f2) == 1


# ── 3 ── repair_duplicate_faces ───────────────────────────────────────────────
def test_UNIT_03_duplicate_faces():
    v = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float64)
    f = np.array([[0,1,2],[0,1,2],[2,0,1]], dtype=np.int32)  # all duplicates
    n = _recompute_normals(v, f)
    log = _log()
    f2, n2 = repair_degenerates.repair_duplicate_faces(f, n, log)
    assert len(f2) == 1, f"Expected 1 face, got {len(f2)}"
    assert log.duplicate_removed == 2, f"Expected 2 removed, got {log.duplicate_removed}"


# ── 4 ── global_vertex_weld — merge within tolerance ─────────────────────────
def test_UNIT_04_weld_vertices_in_tolerance():
    eps = 1e-5
    # Two triangles sharing an edge but with slightly offset vertices
    v = np.array([
        [0,0,0],[1,0,0],[0,1,0],
        [0,0,eps],[1,0,eps],[0,1,0]   # slightly offset copies of first 2
    ], dtype=np.float64)
    f = np.array([[0,1,2],[3,4,5]], dtype=np.int32)
    n = _recompute_normals(v, f)
    cfg = _cfg()
    cfg.merge_tolerance = 1e-4
    log = _log()
    v2, f2, n2 = repair_welding.global_vertex_weld(v, f, n, cfg, log)
    # Some vertices should have been merged
    assert log.vertices_welded > 0, "Expected some vertices to be welded"
    assert len(v2) < len(v), f"Expected fewer unique vertices; got {len(v2)} from {len(v)}"


# ── 5 ── global_vertex_weld — distant vertices untouched ─────────────────────
def test_UNIT_05_weld_distant_vertices_untouched():
    v = np.array([
        [0,0,0],[1,0,0],[0,1,0],
        [5,5,5],[6,5,5],[5,6,5],
    ], dtype=np.float64)
    f = np.array([[0,1,2],[3,4,5]], dtype=np.int32)
    n = _recompute_normals(v, f)
    cfg = _cfg()
    cfg.merge_tolerance = 1e-4
    log = _log()
    v2, f2, n2 = repair_welding.global_vertex_weld(v, f, n, cfg, log)
    assert log.vertices_welded == 0, f"Should not weld distant vertices; got {log.vertices_welded}"


# ── 6 ── repair_non_manifold — removes worst face from 3-face edge ────────────
def test_UNIT_06_non_manifold_edge_excision():
    # Three triangles sharing edge (0,1) — classic non-manifold
    v = np.array([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]], dtype=np.float64)
    f = np.array([
        [0,1,2],  # face 0 shares edge 0-1
        [0,1,3],  # face 1 shares edge 0-1
        [0,1,4],  # face 2 shares edge 0-1  ← 3-face NM edge
    ], dtype=np.int32)
    n = _recompute_normals(v, f)
    cfg, log = _cfg(), _log()
    cfg.max_faces_per_edge = 2
    f2, n2 = repair_nonmanifold.repair_non_manifold(v, f, n, cfg, log)
    assert log.non_manifold_removed >= 1, f"Expected at least 1 removed, got {log.non_manifold_removed}"
    # After repair edge (0,1) should have at most 2 faces
    em = _build_edge_map(f2)
    for edge, flist in em.items():
        assert len(flist) <= 2, f"Edge {edge} still has {len(flist)} faces after repair"


# ── 7 ── repair_topology_separation — pinch-point cracking ───────────────────
def test_UNIT_07_non_manifold_vertex_crack():
    """
    Two fans of triangles sharing only a single vertex (vertex 0).
    After repair they should have separate copies of that vertex.
    """
    v = np.array([
        [0,0,0],       # pinch vertex
        [1,0,0],[0,1,0],   # fan A
        [-1,0,0],[0,-1,0], # fan B
    ], dtype=np.float64)
    # TWO disjoint fans: A={0,1,2}, B={0,3,4}
    f = np.array([
        [0,1,2],
        [0,3,4],
    ], dtype=np.int32)
    log = _log()
    v2, f2 = repair_topology_mod.repair_topology_separation(v, f, log)
    # The cracked vertex count should be updated
    assert log.nm_vertices_cracked >= 1, f"Expected 1 crack, got {log.nm_vertices_cracked}"
    assert len(v2) > len(v), "Vertices should grow"
    assert len(f2) == len(f), "Face count should be preserved"


# ── 8 ── repair_topology_separation — edge cracking ───────────────────────────
def test_UNIT_08_separate_to_manifold():
    v = np.array([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]], dtype=np.float64)
    # Three faces sharing edge (0,1)
    f = np.array([[0,1,2],[0,1,3],[0,1,4]], dtype=np.int32)
    log = _log()
    v2, f2 = repair_topology_mod.repair_topology_separation(v, f, log)
    em = _build_edge_map(f2)
    for edge, flist in em.items():
        assert len(flist) <= 2, f"Edge {edge} still has {len(flist)} faces"
    assert log.nm_vertices_cracked >= 1



# ── 9 ── improved_orientation — consistent winding ───────────────────────────
def test_UNIT_09_orientation_consistent():
    """After repair, no manifold edge should be traversed in the same direction by both faces."""
    v, f, n = _unit_cube_mesh()
    # Flip every other face to break winding
    f_broken = f.copy()
    f_broken[::2, [1,2]] = f_broken[::2, [2,1]]
    log = _log()
    f_fixed = repair_normals_mod.improved_orientation(v, f_broken, log)
    # Check winding consistency
    directed = defaultdict(list)
    for fi, face in enumerate(f_fixed):
        for j in range(3):
            a, b = int(face[j]), int(face[(j+1)%3])
            directed[(a, b)].append(fi)
    em = _build_edge_map(f_fixed)
    conflicts = 0
    for (a,b), flist in em.items():
        if len(flist) == 2:
            fwd = directed.get((a,b),[])
            bwd = directed.get((b,a),[])
            if all(fi in fwd for fi in flist) or all(fi in bwd for fi in flist):
                conflicts += 1
    assert conflicts == 0, f"Found {conflicts} winding conflicts after repair"


# ── 10 ── improved_orientation — inside-out shell flipped outward ─────────────
def test_UNIT_10_inside_out_cube_flipped():
    from src.core.repair_utils import _signed_volume
    v, f, n = _unit_cube_mesh()
    # Flip every face → inside-out
    f_inv = f[:, [0,2,1]]
    assert _signed_volume(v, f_inv) < 0, "Pre-condition: cube should be inside-out"
    log = _log()
    f_fixed = repair_normals_mod.improved_orientation(v, f_inv, log)
    assert _signed_volume(v, f_fixed) > 0, "After orientation repair, signed volume should be positive"


# ── 11 ── check_winding_consistency — detects conflicted edges ────────────────
def test_UNIT_11_winding_audit_detects_conflict():
    v, f, n = _unit_cube_mesh()
    # Artificially flip one face to create a conflict
    f_broken = f.copy()
    f_broken[0, [1,2]] = f_broken[0, [2,1]]
    audit = repair_winding_mod.check_winding_consistency(v, f_broken)
    assert len(audit.conflicted_edges) > 0, "Should detect at least one conflicted edge"
    assert audit.consistency_pct < 100.0


# ── 12 ── repair_winding_order — resolves winding conflicts ──────────────────
def test_UNIT_12_repair_winding_order_resolves():
    v, f, n = _unit_cube_mesh()
    f_broken = f.copy()
    f_broken[0, [1,2]] = f_broken[0, [2,1]]
    cfg, log = _cfg(), _log()
    v2, f2, n2 = repair_winding_mod.repair_winding_order(v, f_broken, n, cfg, log)
    audit = repair_winding_mod.check_winding_consistency(v2, f2)
    assert len(audit.conflicted_edges) == 0, (
        f"After repair, still {len(audit.conflicted_edges)} conflicted edge(s)"
    )


# ── 13 ── crack_mobius_strips — vertex duplication on conflict ────────────────
def test_UNIT_13_mobius_crack_vertex_count():
    """
    If there IS a Möbius-like conflict, crack_mobius_strips should either
    add vertices OR leave the mesh unchanged (if over the crack limit).
    No crash is the minimum bar; if it does crack, vertex count must grow.
    """
    v, f, n = _unit_cube_mesh()
    f_broken = f.copy()
    f_broken[0, [1,2]] = f_broken[0, [2,1]]
    cfg, log = _cfg(), _log()
    cfg.max_mobius_crack = 10  # allow cracking
    v2, f2, n2 = repair_mobius_mod.crack_mobius_strips(v, f_broken, n, cfg, log)
    # Either cracked (vertex count grew) or left unchanged — both are valid outcomes
    assert len(v2) >= len(v), "Vertex count should not shrink"
    assert len(f2) == len(f), "Face count should be unchanged"


# ── 14 ── repair_planar_holes — fills a square hole ──────────────────────────
def test_UNIT_14_planar_hole_filled():
    """Open cube (missing -Z cap) should have its hole filled."""
    v, f, n = _open_cube_minus_cap()
    assert not _is_watertight(f), "Pre-condition: mesh should be open"
    cfg, log = _cfg(), _log()
    v2, f2, n2 = repair_holes_mod.repair_planar_holes(v, f, n, cfg, log)
    assert len(f2) > len(f), "Faces should have been added to fill the hole"
    # The log must reflect the fill
    assert log.planar_holes_filled > 0 or log.non_planar_filled > 0 or log.faces_added > 0, \
        "Log should record a hole fill"


# ── 15 ── repair_non_planar_holes — produces valid output ────────────────────
def test_UNIT_14b_planar_fill_avoids_nonmanifold_diagonal():
    """
    Regression: planar triangulation may choose a diagonal that is already
    saturated (2 incident faces), which would create a non-manifold edge.
    Hole fill should close the cube boundary without reusing either diagonal.
    The added overlapping planar fins are independently invalid solid geometry;
    sealing those into zero-volume shells is not a successful repair.
    """
    v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],   # bottom quad
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],   # top quad (open boundary)
        [0.5, 0.5, 2.0], [0.5, 0.5, 2.2],             # extra verts for diagonal (4,6)
        [0.5, 0.5, 2.4], [0.5, 0.5, 2.6],             # extra verts for diagonal (5,7)
    ], dtype=np.float64)

    # Open cube minus top cap + two already-saturated top diagonals.
    # Any direct planar triangulation of the top quad would reuse one diagonal
    # and create a 3-face edge unless fallback patching is used.
    f = np.array([
        [0, 2, 1], [0, 3, 2],      # bottom
        [0, 1, 5], [0, 5, 4],      # sides
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
        [4, 6, 8], [4, 6, 9],      # diagonal (4,6) has 2 faces
        [5, 7, 10], [5, 7, 11],    # diagonal (5,7) has 2 faces
    ], dtype=np.int32)
    n = _recompute_normals(v, f)

    cfg, log = _cfg(), _log()
    v2, f2, n2 = repair_holes_mod.repair_planar_holes(v, f, n, cfg, log)

    em = _build_edge_map(f2)
    naked = [edge for edge, flist in em.items() if len(flist) == 1]
    non_manifold = [edge for edge, flist in em.items() if len(flist) >= 3]

    assert len(non_manifold) == 0, f"Hole fill introduced non-manifold edges: {non_manifold[:5]}"
    top_boundary={(4,5),(5,6),(6,7),(4,7)}
    assert not (set(naked)&top_boundary), 'Cube cap boundary remains open'
    assert len(naked)==8, 'Preserve the overlapping fins for explicit review'


def test_UNIT_15_non_planar_hole_no_crash():
    """
    repair_non_planar_holes is a passthrough now (all holes handled by
    improved_hole_fill). Verify it runs without error and is idempotent.
    """
    v, f, n = _open_cube_minus_cap()
    cfg, log = _cfg(), _log()
    v2, f2, n2 = repair_holes_mod.repair_non_planar_holes(v, f, n, cfg, log)
    # It should return at least as many faces (idempotent or filling)
    assert len(f2) >= len(f)


# ── 16 ── bary_smooth — vertices move toward barycentre ──────────────────────
@pytest.mark.skip(reason="Smoothing is outside the shape-preserving pipeline; module is not shipped.")
def test_UNIT_16_smooth_moves_vertices():
    v, f, n = _unit_cube_mesh()
    v_orig = v.copy()
    v2 = repair_smooth_mod.bary_smooth(v, f, iterations=3, factor=0.5)
    # Interior vertices should have moved (cube has no interior verts, but
    # boundary smoothing is still applied, so positions should differ)
    delta = np.linalg.norm(v2 - v_orig, axis=1)
    assert np.any(delta > 0), "Smoothing should move at least some vertices"


# ── 17 ── bary_smooth — 0 iterations → no change ─────────────────────────────
@pytest.mark.skip(reason="Smoothing is outside the shape-preserving pipeline; module is not shipped.")
def test_UNIT_17_smooth_zero_iterations_noop():
    v, f, n = _unit_cube_mesh()
    v2 = repair_smooth_mod.bary_smooth(v, f, iterations=0)
    assert np.allclose(v, v2), "Zero iterations should be a no-op"


# ── 18 ── filter_small_shells — removes tiny shell ───────────────────────────
def test_UNIT_18_filter_small_shells():
    """Main cube + tiny 1-triangle shell far away → tiny shell removed."""
    v_cube, f_cube, n_cube = _unit_cube_mesh()
    # Tiny triangle far away (area ≈ 0.005, << 0.2% of cube ~6.0)
    tiny_offset = len(v_cube)
    v_tiny = np.array([[100,100,100],[100.01,100,100],[100,100.01,100]], dtype=np.float64)
    f_tiny = np.array([[tiny_offset, tiny_offset+1, tiny_offset+2]], dtype=np.int32)
    n_tiny = _recompute_normals(np.vstack([v_cube, v_tiny]), f_tiny)

    v_all = np.vstack([v_cube, v_tiny])
    f_all = np.vstack([f_cube, f_tiny])
    n_all = np.vstack([n_cube, n_tiny])

    cfg, log = _cfg(), _log()
    cfg.shell_filter_enabled = True
    cfg.shell_area_threshold = 0.002   # explicitly requested open-shell filtering
    v2, f2, n2 = repair_shell_mod.filter_small_shells(v_all, f_all, n_all, cfg, log)
    assert log.shells_filtered >= 1, f"Expected tiny shell filtered; got {log.shells_filtered}"
    assert len(f2) == len(f_cube), f"Should keep only cube faces ({len(f_cube)}), got {len(f2)}"


# ── 19 ── _recompute_normals — output is unit vectors ─────────────────────────
def test_UNIT_19_recompute_normals_unit():
    from src.core.repair_utils import _recompute_normals as rn
    v, f, _ = _unit_cube_mesh()
    n = rn(v, f)
    lengths = np.linalg.norm(n, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-10), f"Normals not unit: {lengths}"


# ── 20 ── _signed_volume — positive for outward cube ─────────────────────────
def test_UNIT_20_signed_volume_positive():
    from src.core.repair_utils import _signed_volume
    v, f, _ = _unit_cube_mesh()
    vol = _signed_volume(v, f)
    assert vol > 0, f"Outward cube should have positive signed volume; got {vol}"


# ── 21 ── _compact_mesh — unused vertices removed ─────────────────────────────
def test_UNIT_21_compact_mesh():
    from src.core.repair_utils import _compact_mesh
    v = np.array([[0,0,0],[1,0,0],[0,1,0],[99,99,99]], dtype=np.float64)
    f = np.array([[0,1,2]], dtype=np.int32)
    v2, f2 = _compact_mesh(v, f)
    assert len(v2) == 3, f"Expected 3 unique vertices after compaction; got {len(v2)}"
    assert len(f2) == 1


# ── 22 ── build_edge_map — face count per edge ───────────────────────────────
def test_UNIT_22_build_edge_map():
    from src.core.mesh_analyzer import build_edge_map
    v, f, _ = _unit_cube_mesh()
    em = build_edge_map(f)
    for edge, flist in em.items():
        assert len(flist) <= 2, f"Cube should have max 2 faces per edge; {edge}→{len(flist)}"
    # Cube has 18 unique edges
    assert len(em) == 18, f"Expected 18 edges; got {len(em)}"


# ── 23 ── check_naked_and_nonmanifold ────────────────────────────────────────
def test_UNIT_23_naked_and_nonmanifold():
    from src.core.mesh_analyzer import build_edge_map, check_naked_and_nonmanifold
    v, f, _ = _open_cube_minus_cap()
    em = build_edge_map(f)
    naked, nm = check_naked_and_nonmanifold(em)
    assert len(naked) == 4, f"Open cube missing 1 cap should have 4 naked edges; got {len(naked)}"
    assert len(nm) == 0


# ── 24 ── _face_area — right triangle ────────────────────────────────────────
def test_UNIT_24_face_area():
    from src.core.repair_utils import _face_area
    v = np.array([[0,0,0],[2,0,0],[0,2,0]], dtype=np.float64)
    f = np.array([0,1,2], dtype=np.int32)
    area = _face_area(v, f)
    assert abs(area - 2.0) < 1e-10, f"Expected area 2.0; got {area}"


# ── 26 ── repair_topology_separation — complex junction ──────────────────────
def test_UNIT_26_topology_complex_junction():
    """
    Test a highly non-manifold junction:
    - Vertex 0 is shared by 2 separate fans.
    - One of those fans has a non-manifold edge (3 faces sharing edge 0-1).
    """
    v = np.array([
        [0,0,0],         # 0: central vertex
        [1,0,0], [0,1,0], # 1, 2: other vertices for fan A
        [-1,0,0], [0,-1,0], # 3, 4: other vertices for fan B
        [1,1,1]          # 5: extra vertex for non-manifold edge (0,1)
    ], dtype=np.float64)
    
    # Fan A: (0,1,2)
    # Fan B: (0,3,4)
    # Non-manifold addition to edge (0,1): (0,1,5) AND [0,1, something else]
    # To make edge (0,1) non-manifold (3 faces):
    f = np.array([
        [0,1,2], # Face 0
        [0,1,5], # Face 1 -> Edge (0,1) now has 2 faces. Still manifold.
        [0,1,3], # Face 2 -> Edge (0,1) now has 3 faces. NON-MANIFOLD.
        [0,3,4], # Face 3 -> Shares edge (0,3) with Face 2.
    ], dtype=np.int32)
    
    # Adjacency at vertex 0:
    # Edges at 0:
    # (0,1) -> faces 0,1,2 (Count 3) -> NOT connected
    # (0,2) -> face 0
    # (0,5) -> face 1
    # (0,3) -> faces 2,3 (Count 2) -> Connected
    # (0,4) -> face 3
    # Resulting fans:
    # Fan A: {face 0}
    # Fan B: {face 1}
    # Fan C: {face 2, face 3}
    # Total 3 fans -> 2 cracks.
    
    log = _log()
    v2, f2 = repair_topology_mod.repair_topology_separation(v, f, log)
    
    # Check manifoldness
    em = _build_edge_map(f2)
    for edge, flist in em.items():
        assert len(flist) <= 2, f"Edge {edge} still has {len(flist)} faces"
        
    assert log.nm_vertices_cracked >= 2, f"Expected 2 cracks, got {log.nm_vertices_cracked}"


# ═════════════════════════════════════════════════════════════════════════════
#  INTEGRATION TESTS
# ═════════════════════════════════════════════════════════════════════════════

def test_INT_01_degenerate_duplicate_then_normals():
    """Degenerate → duplicate → orientation on a mesh with injected defects."""
    v, f, n = _unit_cube_mesh()
    # Inject a zero-area face and a duplicate
    v_extra = np.vstack([v, v[0]])  # extra vertex same as vertex 0
    degen   = np.array([[0, 0, 1]], dtype=np.int32)   # repeated-vertex degenerate
    dup     = f[0:1].copy()                             # duplicate of face 0
    f_bad   = np.vstack([f, degen, dup])
    n_bad   = _recompute_normals(v_extra, f_bad)

    cfg, log = _cfg(), _log()
    f2, n2   = repair_degenerates.repair_degenerate_faces(v_extra, f_bad, n_bad, cfg, log)
    f3, n3   = repair_degenerates.repair_duplicate_faces(f2, n2, log)
    f4       = repair_normals_mod.improved_orientation(v_extra, f3, log)

    assert log.degenerate_removed >= 1
    assert log.duplicate_removed  >= 1
    # After orientation, winding should be consistent
    directed = defaultdict(list)
    for fi, face in enumerate(f4):
        for j in range(3):
            a, b = int(face[j]), int(face[(j+1)%3])
            directed[(a,b)].append(fi)
    em = _build_edge_map(f4)
    for (a,b), flist in em.items():
        if len(flist) == 2:
            fwd = directed.get((a,b),[])
            bwd = directed.get((b,a),[])
            assert not (all(fi in fwd for fi in flist) or all(fi in bwd for fi in flist)), \
                f"Winding conflict remains at edge ({a},{b})"


def test_INT_02_weld_then_fill_watertight():
    """
    Two offset triangles that after welding form a watertight prism,
    then hole filling should close any remaining gaps.
    """
    v, f, n = _open_cube_minus_cap()
    cfg, log = _cfg(), _log()
    # Welding alone won't change this mesh since vertices are already unique,
    # but hole fill should close the 4-edge boundary loop.
    v2, f2, n2 = repair_holes_mod.repair_planar_holes(v, f, n, cfg, log)
    # Should now be watertight
    assert _is_watertight(f2), "Mesh should be watertight after hole fill"


def test_INT_03_nonmanifold_then_normals_then_weld():
    """Remove NM faces, fix normals, then weld. Verify mesh stays valid."""
    v = np.array([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]], dtype=np.float64)
    # 3-face edge on (0,1)
    f = np.array([[0,1,2],[0,1,3],[0,1,4],[2,3,4]], dtype=np.int32)
    n = _recompute_normals(v, f)
    cfg, log = _cfg(), _log()

    f2, n2 = repair_nonmanifold.repair_non_manifold(v, f, n, cfg, log)
    assert log.non_manifold_removed >= 1

    f3     = repair_normals_mod.improved_orientation(v, f2, log)
    v3, f4, n4 = repair_welding.global_vertex_weld(v, f3, n2, cfg, log)

    # All faces should have valid indices
    assert f4.max() < len(v3), "Face references out-of-bounds vertex"
    assert f4.min() >= 0


def test_INT_04_full_pipeline_broken_cube():
    """
    Integration test: run the major repair stages in pipeline order on a
    cube with injected NM edge, flipped face, and a missing cap.
    Verifies: no crash + mesh is watertight + normals are unit vectors.
    """
    v, f, n = _open_cube_minus_cap()

    # Inject a non-manifold edge by duplicating face 0 with a slight twist
    f_bad = np.vstack([f, np.array([[f[0,0], f[0,1], f[0,2]]], dtype=np.int32)])
    n_bad = _recompute_normals(v, f_bad)

    # Also flip one face
    f_bad[2, [1,2]] = f_bad[2, [2,1]]

    cfg, log = _cfg(), _log()

    # Stage 1 — clean
    f1, n1 = repair_degenerates.repair_degenerate_faces(v, f_bad, n_bad, cfg, log)
    f1, n1 = repair_degenerates.repair_duplicate_faces(f1, n1, log)

    # Stage 2 — non-manifold
    f1, n1 = repair_nonmanifold.repair_non_manifold(v, f1, n1, cfg, log)

    # Stage 3 — normals
    f1     = repair_normals_mod.improved_orientation(v, f1, log)
    n1     = _recompute_normals(v, f1)

    # Stage 4 — weld
    v1, f1, n1 = repair_welding.global_vertex_weld(v, f1, n1, cfg, log)

    # Stage 5 — hole fill
    v1, f1, n1 = repair_holes_mod.repair_planar_holes(v1, f1, n1, cfg, log)

    # Post-conditions
    assert _is_watertight(f1), "Full pipeline should produce a watertight mesh"
    n_final = _recompute_normals(v1, f1)
    lengths = np.linalg.norm(n_final, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-10), "All normals should be unit vectors"


# ═════════════════════════════════════════════════════════════════════════════
#  Report
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
