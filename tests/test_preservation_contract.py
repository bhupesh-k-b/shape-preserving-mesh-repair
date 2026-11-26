from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from src.core.mesh_analyzer import MeshData, MeshAnalyzer, build_edge_map, check_naked_and_nonmanifold
from src.core.repair_config import RepairConfig, RepairLog
from src.core.repair_utils import _recompute_normals, _signed_volume
from src.pipelines.mesh_repair_pipeline import MeshRepairPipeline, write_stl
from src.routines.repair_holes import repair_planar_holes
from src.routines.repair_nonmanifold import repair_non_manifold
from src.routines.repair_shell_union import _find_shells, filter_small_shells, repair_shell_union
from src.routines.repair_topology_separation import repair_topology_separation
from src.routines.repair_winding import repair_winding_order

ROOT = Path(__file__).resolve().parents[1]


def save(path, mesh):
    write_stl(str(path), mesh.vertices, mesh.faces, mesh.normals)
    return path


def triangle_keys(mesh):
    return {tuple(sorted(tuple(p) for p in mesh.vertices[face])) for face in mesh.faces}


@pytest.mark.parametrize('extension', ['.stl', '.STL', ''])
@pytest.mark.parametrize('in_place', [False, True])
def test_cli_preserves_files_and_reports_verified_success(tmp_path, cube, extension, in_place):
    source = save(tmp_path / ('source'+extension), cube)
    output = source if in_place else tmp_path / ('repaired'+extension)
    run = subprocess.run([sys.executable, str(ROOT/'main.py'), str(source), str(output), '--quiet'],
                         cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout+run.stderr
    assert source.exists() and output.exists()
    assert triangle_keys(MeshData.from_stl(str(output))) == triangle_keys(cube)


def test_disabled_filter_preserves_duplicated_surface(cube):
    f = np.vstack([cube.faces, cube.faces])
    _, result, _ = filter_small_shells(cube.vertices, f, _recompute_normals(cube.vertices,f),
                                     RepairConfig(shell_filter_enabled=False), RepairLog(), verbose=False)
    assert len(result) == 24
    assert len(_find_shells(f,build_edge_map(f))) == 1


def test_analysis_never_calls_nonmanifold_mesh_watertight(tmp_path,cube):
    f=np.vstack([cube.faces,cube.faces])
    path=tmp_path/'doubled.stl'
    write_stl(str(path),cube.vertices,f,_recompute_normals(cube.vertices,f))
    _, report=MeshAnalyzer().analyze(str(path))
    assert not report.is_watertight


def test_excision_removes_actual_candidates_and_logs_truth():
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],dtype=float)
    f=np.array([[0,1,2],[0,1,3],[0,1,4]],dtype=int)
    log=RepairLog()
    out,_=repair_non_manifold(v,f,_recompute_normals(v,f),RepairConfig(),log)
    assert not check_naked_and_nonmanifold(build_edge_map(out))[1]
    assert len(f)-len(out)==log.non_manifold_removed
    assert len(out)>0


def test_coarse_cap_preserves_existing_surface(cube):
    f=cube.faces[2:]
    v,out,_=repair_planar_holes(cube.vertices,f,_recompute_normals(cube.vertices,f),RepairConfig(),RepairLog())
    assert not any(check_naked_and_nonmanifold(build_edge_map(out)))
    assert np.array_equal(v[:len(cube.vertices)],cube.vertices)
    assert set(map(tuple,np.sort(f,axis=1))) <= set(map(tuple,np.sort(out,axis=1)))


def test_separation_never_moves_vertices_or_shreds_sheets(cube):
    other=np.array([[3,1,0],[3,2,1],[5,7,4],[5,6,7],[1,4,0],[1,5,4],
                    [3,6,2],[3,7,6],[4,3,0],[4,7,3],[2,5,1],[2,6,5]],dtype=int)
    f=np.vstack([cube.faces,other])
    v,out=repair_topology_separation(cube.vertices,f,RepairLog())
    assert np.array_equal(v[out],cube.vertices[f])
    assert len(out)==len(f)
    assert not check_naked_and_nonmanifold(build_edge_map(out))[1]
    assert all(len(s)>2 for s in _find_shells(out,build_edge_map(out)))


def test_hollow_orientation_survives_winding(cube):
    v=np.vstack([cube.vertices*3,cube.vertices+1])
    f=np.vstack([cube.faces,cube.faces+8])
    cfg=RepairConfig(union_hollow=True,union_force_tier=2,shell_filter_enabled=False)
    v,f,n,_=repair_shell_union(v,f,_recompute_normals(v,f),cfg,RepairLog(),verbose=False)
    v,f,_=repair_winding_order(v,f,n,cfg,RepairLog())
    volumes=sorted(_signed_volume(v,f[np.array(s)]) for s in _find_shells(f,build_edge_map(f)))
    assert np.allclose(volumes,[-1,27])


def test_real_boolean_for_partial_overlap(cube):
    v=np.vstack([cube.vertices,cube.vertices+[.4,.3,.2]])
    f=np.vstack([cube.faces,cube.faces+8])
    v,f,n,log=repair_shell_union(v,f,_recompute_normals(v,f),RepairConfig(union_force_tier=2),RepairLog(),verbose=False)
    assert not log.success or np.isclose(_signed_volume(v,f),1.664)


@pytest.mark.parametrize('scale,offset',[(1.,0.),(1e-7,0.),(1.,1e6)])
def test_raw_import_retains_scale_and_geometry(tmp_path,cube,scale,offset):
    v=cube.vertices*scale+offset
    f=cube.faces
    path=tmp_path/'scaled.stl'
    write_stl(str(path),v,f,_recompute_normals(v,f))
    raw=MeshData.from_stl(str(path))
    assert len(raw.vertices)==8
    assert len(raw.faces)==12


def test_failed_repair_keeps_existing_destination(tmp_path,cube):
    source=tmp_path/'sheet.stl'
    f=cube.faces[:1]
    write_stl(str(source),cube.vertices,f,_recompute_normals(cube.vertices,f))
    output=tmp_path/'existing.stl'
    output.write_bytes(b'previous good output')
    _,log=MeshRepairPipeline().repair(str(source),str(output),verbose=False)
    assert not log.success
    assert output.read_bytes()==b'previous good output'


def test_micro_cube_repair_keeps_all_geometry(tmp_path,cube):
    v=cube.vertices*1e-7
    source=tmp_path/'micro.stl'
    write_stl(str(source),v,cube.faces,_recompute_normals(v,cube.faces))
    output=tmp_path/'result.stl'
    _,log=MeshRepairPipeline().repair(str(source),str(output),verbose=False)
    assert log.success,log.summary()
    assert triangle_keys(MeshData.from_stl(str(output)))==triangle_keys(MeshData.from_stl(str(source)))


def test_public_pipeline_closes_cube_without_mutating_input(tmp_path,cube):
    original=cube.vertices.copy()
    source=tmp_path/'open.stl'
    write_stl(str(source),cube.vertices,cube.faces[2:],cube.normals[2:])
    output=tmp_path/'repaired.stl'
    _,log=MeshRepairPipeline().repair(str(source),str(output),verbose=False)
    assert log.success,log.summary()
    assert not any(check_naked_and_nonmanifold(build_edge_map(MeshData.from_stl(str(output)).faces)))
    assert np.array_equal(cube.vertices,original)
