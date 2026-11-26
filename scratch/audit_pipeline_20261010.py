"""Reproduce pipeline audit findings without changing source meshes or repair code.

Run from the project root: python3 scratch/audit_pipeline_20261010.py
All STL outputs live in a temporary directory that is removed on exit.
"""

from pathlib import Path
import contextlib
import io
import json
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.mesh_analyzer import MeshAnalyzer, MeshData, build_edge_map, check_naked_and_nonmanifold
from src.core.repair_config import RepairConfig, RepairLog
from src.core.repair_utils import _recompute_normals, _signed_volume
from src.pipelines.mesh_repair_initial import InitialCleanupPipeline
from src.pipelines.mesh_repair_pipeline import MeshRepairPipeline, write_stl, _stabilize_float32_vertex_splits
from src.routines.repair_holes import repair_planar_holes
from src.routines.repair_nonmanifold import repair_non_manifold
from src.routines.repair_shell_union import _find_shells, repair_shell_union, filter_small_shells
from src.routines.repair_winding import repair_winding_order
from src.routines.repair_topology_separation import repair_topology_separation
import src.pipelines.mesh_repair_pipeline as pipeline_module


def cube():
    vertices = np.array([
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,1], [1,0,1], [1,1,1], [0,1,1],
    ], dtype=float)
    faces = np.array([
        [0,2,1], [0,3,2], [4,5,6], [4,6,7],
        [0,1,5], [0,5,4], [2,3,7], [2,7,6],
        [0,4,7], [0,7,3], [1,2,6], [1,6,5],
    ], dtype=np.int32)
    return vertices, faces


def defects(faces):
    naked, nm = check_naked_and_nonmanifold(build_edge_map(faces))
    return {"faces": len(faces), "naked_edges": len(naked), "non_manifold_edges": len(nm)}


def run_cli(input_path, output_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), str(input_path), str(output_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    return {
        "returncode": result.returncode,
        "claims_success": "[SUCCESS]" in result.stdout,
        "output_exists": output_path.exists(),
        "stdout_tail": result.stdout.splitlines()[-3:],
        "stderr_tail": result.stderr.splitlines()[-3:],
    }


def main():
    results = {}
    v, f = cube()
    with tempfile.TemporaryDirectory(prefix="mesh-pipeline-audit-") as directory:
        tmp = Path(directory)
        closed_input = tmp / "cube.stl"
        write_stl(str(closed_input), v, f, _recompute_normals(v, f))

        doubled = np.vstack([f, f])
        cfg = RepairConfig(shell_filter_enabled=False)
        filtered_v, filtered_f, _ = filter_small_shells(
            v, doubled, _recompute_normals(v, doubled), cfg, RepairLog(), verbose=False
        )
        results["duplicate_cube_filtering"] = {
            "input_faces": len(doubled),
            "shells_detected": len(_find_shells(doubled, build_edge_map(doubled))),
            "filter_disabled": True,
            "output_faces": len(filtered_f),
        }
        assert len(filtered_f) == 0

        alternate_f = np.array([
            [3,1,0], [3,2,1], [5,7,4], [5,6,7],
            [1,4,0], [1,5,4], [3,6,2], [3,7,6],
            [4,3,0], [4,7,3], [2,5,1], [2,6,5],
        ], dtype=np.int32)
        sheet_f = np.vstack([f, alternate_f])
        split_log = RepairLog()
        _, split_f = repair_topology_separation(v, sheet_f, split_log)
        results["topology_separation_fragmentation"] = {
            "before": defects(sheet_f), "after": defects(split_f),
            "vertices_cracked": split_log.nm_vertices_cracked,
            "output_shell_face_counts": [len(shell) for shell in _find_shells(split_f, build_edge_map(split_f))],
        }
        assert defects(split_f)["naked_edges"] == 48

        doubled_path = tmp / "doubled.stl"
        write_stl(str(doubled_path), v, doubled, _recompute_normals(v, doubled))
        _, report = MeshAnalyzer().analyze(str(doubled_path))
        results["non_manifold_report_watertight"] = {
            "report_is_watertight": report.is_watertight,
            "non_manifold_edges": len(report.non_manifold_edges),
        }
        assert report.is_watertight and report.non_manifold_edges

        nv = np.array([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]], dtype=float)
        nf = np.array([[0,1,2],[0,1,3],[0,1,4]], dtype=np.int32)
        nm_log = RepairLog()
        nf_after, _ = repair_non_manifold(nv, nf, _recompute_normals(nv, nf), RepairConfig(), nm_log)
        results["non_manifold_excision"] = {
            "before": defects(nf), "after": defects(nf_after),
            "reported_faces_removed": nm_log.non_manifold_removed,
        }
        assert defects(nf_after)["non_manifold_edges"] == 1

        open_f = f[2:]
        hole_log = RepairLog()
        _, filled_f, _ = repair_planar_holes(v, open_f, _recompute_normals(v, open_f), RepairConfig(), hole_log)
        results["unit_cube_hole_fill"] = {
            "before": defects(open_f), "after": defects(filled_f),
            "faces_added": hole_log.faces_added,
            "boundary_edge": 1.0, "guard_max_edge": float(0.35 * np.sqrt(3)),
        }
        assert len(filled_f) == len(open_f) and hole_log.faces_added == 0

        open_input = tmp / "open-cube.stl"
        open_output = tmp / "open-repaired.stl"
        write_stl(str(open_input), v, open_f, _recompute_normals(v, open_f))
        repaired, _ = MeshRepairPipeline().repair(str(open_input), str(open_output), verbose=False)
        results["api_returns_unrepaired_mesh"] = {
            "returned_normally": True, "output_exists": open_output.exists(),
            "after": defects(repaired.faces),
        }
        assert defects(repaired.faces)["naked_edges"] == 4

        # Model-scale cleanup behavior on an existing, small sample.
        chair_input = ROOT / "data/test_meshes/Chair.stl"
        chair_output = tmp / "chair-cleanup.stl"
        chair_before = MeshData.from_stl(str(chair_input))
        InitialCleanupPipeline(iterations=3).repair(str(chair_input), str(chair_output), verbose=False)
        chair_after = MeshData.from_stl(str(chair_output))
        d_forward = cKDTree(chair_after.vertices).query(chair_before.vertices)[0]
        d_reverse = cKDTree(chair_before.vertices).query(chair_after.vertices)[0]
        diagonal = np.linalg.norm(np.ptp(chair_before.vertices, axis=0))
        results["chair_cleanup"] = {
            "before": defects(chair_before.faces), "after": defects(chair_after.faces),
            "max_nearest_vertex_distance": float(max(d_forward.max(), d_reverse.max())),
            "bbox_diagonal": float(diagonal),
            "max_nearest_vertex_distance_pct_diagonal": float(100 * max(d_forward.max(), d_reverse.max()) / diagonal),
        }

        plane_input = ROOT / "data/test_meshes/Plane_lower.stl"
        plane_before = MeshData.from_stl(str(plane_input))
        _, plane_filtered, _ = filter_small_shells(
            plane_before.vertices, plane_before.faces, plane_before.normals,
            RepairConfig.from_ini(str(ROOT / "repair_config.ini")), RepairLog(), verbose=False,
        )
        results["plane_lower_filtering"] = {
            "input_faces": len(plane_before.faces),
            "shells_detected": len(_find_shells(plane_before.faces, build_edge_map(plane_before.faces))),
            "output_faces": len(plane_filtered),
        }
        plane_cli = tmp / "plane-repaired.stl"
        results["plane_lower_cli"] = run_cli(plane_input, plane_cli)
        if plane_cli.exists():
            results["plane_lower_cli"]["output_size_bytes"] = plane_cli.stat().st_size
            with plane_cli.open("rb") as handle:
                handle.seek(80)
                results["plane_lower_cli"]["output_triangles"] = struct.unpack("<I", handle.read(4))[0]

        # Uppercase extension aliases the temporary and final paths.
        uppercase_output = tmp / "uppercase-output.STL"
        results["uppercase_output_cli"] = run_cli(closed_input, uppercase_output)
        assert results["uppercase_output_cli"]["claims_success"] and not uppercase_output.exists()

        in_place_input = tmp / "in-place.STL"
        write_stl(str(in_place_input), v, f, _recompute_normals(v, f))
        results["uppercase_in_place_cli"] = run_cli(in_place_input, in_place_input)
        results["uppercase_in_place_cli"]["input_exists_after"] = in_place_input.exists()
        assert not in_place_input.exists()

        # Force the documented fallback to isolate hollow-orientation behavior.
        nested_v = np.vstack([v * 3, v + 1])
        nested_f = np.vstack([f, f + len(v)])
        hollow_cfg = RepairConfig(union_hollow=True, union_force_tier=2, shell_filter_enabled=False)
        hv, hf, hn, union_log = repair_shell_union(
            nested_v, nested_f, _recompute_normals(nested_v, nested_f), hollow_cfg, RepairLog(), verbose=False
        )
        shells = _find_shells(hf, build_edge_map(hf))
        volumes_before = [_signed_volume(hv, hf[np.array(shell)]) for shell in shells]
        hv, hf, _ = repair_winding_order(hv, hf, hn, hollow_cfg, RepairLog(), verbose=False)
        volumes_after = [_signed_volume(hv, hf[np.array(shell)]) for shell in shells]
        results["hollow_void_orientation"] = {
            "union_strategy": union_log.strategy_used,
            "shell_signed_volumes_after_union": volumes_before,
            "shell_signed_volumes_after_winding": volumes_after,
        }
        assert any(value < 0 for value in volumes_before) and all(value > 0 for value in volumes_after)

        # Partially intersecting boxes: Tier 2 never visits them as parents.
        overlap_v = np.vstack([v, v + np.array([0.4,0.3,0.2])])
        overlap_f = np.vstack([f, f + len(v)])
        ov, of, on, overlap_log = repair_shell_union(
            overlap_v, overlap_f, _recompute_normals(overlap_v, overlap_f),
            RepairConfig(union_force_tier=2), RepairLog(), verbose=False
        )
        results["overlapping_boxes_union"] = {
            "strategy": overlap_log.strategy_used, "claims_union_success": overlap_log.success,
            "culled_faces": overlap_log.interior_faces_culled,
            "after": defects(of), "signed_volume": _signed_volume(ov, of),
            "correct_union_volume": 2 - (0.6 * 0.7 * 0.8),
        }
        assert len(of) == 24 and overlap_log.success and overlap_log.interior_faces_culled == 0

        # The internal timeout returns a tuple without writing the requested output.
        clock_value = [0.0]
        def timeout_stage(vertices, faces, normals, *args, **kwargs):
            clock_value[0] = 301.0
            return vertices, faces, normals
        timeout_output = tmp / "timeout.stl"
        with patch.object(pipeline_module.time, "perf_counter", side_effect=lambda: clock_value[0]), patch.object(
            pipeline_module, "filter_small_shells", side_effect=timeout_stage
        ):
            timeout_mesh, timeout_log = MeshRepairPipeline().repair(str(closed_input), str(timeout_output), verbose=False)
        results["internal_timeout"] = {
            "returned_normally": True, "output_exists": timeout_output.exists(),
            "returned_faces": len(timeout_mesh.faces), "reported_time_ms": timeout_log.repair_time_ms,
        }
        assert not timeout_output.exists()

        split_vertices = np.array([[0,0,0], [0,0,0], [1e-6,0,0]], dtype=float)
        stabilized = _stabilize_float32_vertex_splits(split_vertices)
        far_split = np.array([[1e6,1e6,1e6], [1e6,1e6,1e6]], dtype=float)
        far_stabilized = _stabilize_float32_vertex_splits(far_split)
        results["export_vertex_stabilization"] = {
            "before": split_vertices.tolist(), "after": stabilized.tolist(),
            "distinct_float32_positions_after": int(len(np.unique(stabilized.astype(np.float32), axis=0))),
            "duplicate_displacement_at_1e6": float(np.linalg.norm(far_stabilized[1] - far_split[1])),
        }
        assert len(np.unique(stabilized.astype(np.float32), axis=0)) == 2

        # Identical model under two scales; fixed import tolerance collapses the small one.
        micro_path = tmp / "micro-cube.stl"
        write_stl(str(micro_path), v * 1e-7, f, _recompute_normals(v * 1e-7, f))
        micro_mesh = MeshData.from_stl(str(micro_path))
        results["scale_dependent_import"] = {
            "unit_cube_vertices": len(MeshData.from_stl(str(closed_input)).vertices),
            "micro_cube_vertices": len(micro_mesh.vertices),
            "repeated_vertex_faces": int(np.count_nonzero(
                (micro_mesh.faces[:,0] == micro_mesh.faces[:,1]) |
                (micro_mesh.faces[:,1] == micro_mesh.faces[:,2])
            )),
        }
        assert len(micro_mesh.vertices) == 1

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
