import os
import struct
from pathlib import Path
import time
import numpy as np
import pytest
from src.core.mesh_analyzer import MeshData,build_edge_map
from src.core.mesh_io import write_mesh,load_mesh,write_stl
from src.core.repair_config import RepairConfig,RepairLog
from src.core.repair_utils import _recompute_normals,_signed_volume
from src.core.validation import validate,surface_preservation
from src.pipelines.mesh_repair_pipeline import MeshRepairPipeline
import src.pipelines.mesh_repair_pipeline as orchestration


@pytest.mark.parametrize('alias', ['input', 'output', 'symlink', 'hardlink'])
def test_cli_report_cannot_overwrite_mesh(tmp_path, cube, alias):
    from main import main
    source = tmp_path / 'source.stl'
    output = tmp_path / 'output.stl'
    write_stl(source, cube.vertices, cube.faces, cube.normals)
    original = source.read_bytes()
    report = source if alias == 'input' else output
    if alias in {'symlink', 'hardlink'}:
        report = tmp_path / 'alias.json'
        if alias == 'symlink':
            report.symlink_to(source)
        else:
            os.link(source, report)
    with pytest.raises(SystemExit) as exc:
        main([str(source), str(output), '--report', str(report), '--quiet'])
    assert exc.value.code == 2
    assert source.read_bytes() == original
    assert not output.exists()


def test_single_shell_union_requires_independent_geometry_check(monkeypatch, cube):
    from src.core import geometry_backend
    from src.routines.repair_shell_union import repair_shell_union
    monkeypatch.setattr(geometry_backend, 'intersects', lambda *args: True)
    _, _, _, log = repair_shell_union(
        cube.vertices, cube.faces, cube.normals, RepairConfig(), RepairLog(), verbose=False)
    assert not log.success


def test_independent_checker_rejects_intersecting_closed_solids(cube):
    v=np.vstack([cube.vertices,cube.vertices+[.4,.3,.2]])
    f=np.vstack([cube.faces,cube.faces+8])
    report=validate(MeshData(v,f,_recompute_normals(v,f)),RepairConfig())
    assert report.complete and report.self_intersection and not report.valid
    assert report.naked_edges==report.non_manifold_edges==0


def test_surface_gate_rejects_deformation_with_perfect_topology(cube):
    moved=MeshData(cube.vertices+[.001,0,0],cube.faces,cube.normals)
    assert validate(moved,RepairConfig()).valid
    accepted,error,certificate=surface_preservation(cube,moved,RepairConfig())
    assert not accepted and error>.0009
    assert certificate=='uncertified_surface_coverage'


def test_bounded_gate_handles_coplanar_retriangulation(cube):
    other=np.array([[3,1,0],[3,2,1],[5,7,4],[5,6,7],[1,4,0],[1,5,4],
                    [3,6,2],[3,7,6],[4,3,0],[4,7,3],[2,5,1],[2,6,5]],dtype=int)
    retriangulated=MeshData(cube.vertices,other,_recompute_normals(cube.vertices,other))
    accepted,_,_=surface_preservation(cube,retriangulated,RepairConfig(),bidirectional=True)
    assert accepted


def test_indexed_output_preserves_touching_objects_while_stl_is_declined(tmp_path,cube):
    v=np.vstack([cube.vertices,cube.vertices+[1,1,1]])
    f=np.vstack([cube.faces,cube.faces+8])
    indexed=MeshData(v,f,_recompute_normals(v,f))
    source=tmp_path/'touching.3mf'
    write_mesh(source,indexed)
    output=tmp_path/'touching-output.3mf'
    _,log=MeshRepairPipeline().repair(source,output,verbose=False)
    assert log.success,log.summary()
    stl=tmp_path/'touching-output.stl'
    _,log=MeshRepairPipeline().repair(source,stl,verbose=False)
    assert not log.success and not stl.exists()


def test_hollow_3mf_round_trip_preserves_void_in_material_object(tmp_path,cube):
    v=np.vstack([cube.vertices*3,cube.vertices+1])
    f=np.vstack([cube.faces,cube.faces[:,[0,2,1]]+8])
    mesh=MeshData(v,f,_recompute_normals(v,f))
    path=tmp_path/'hollow.3mf'
    write_mesh(path,mesh)
    loaded=load_mesh(path)
    assert validate(loaded,RepairConfig()).valid
    assert np.isclose(_signed_volume(loaded.vertices,loaded.faces),26)
    import zipfile,xml.etree.ElementTree as ET
    with zipfile.ZipFile(path) as archive:
        xml=ET.fromstring(archive.read('3D/3dmodel.model'))
    assert len(xml.find('{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}build'))==1


def test_timeout_preserves_destination_and_bounds_complete_job(tmp_path,cube):
    source=tmp_path/'source.stl'; output=tmp_path/'existing.stl'
    write_stl(source,cube.vertices,cube.faces,cube.normals)
    output.write_bytes(b'previous result')
    started=time.perf_counter()
    _,log=MeshRepairPipeline(RepairConfig(timeout_seconds=.01)).repair(source,output,verbose=False)
    assert log.status=='TIMEOUT' and not log.success
    assert time.perf_counter()-started<3
    assert output.read_bytes()==b'previous result'
    assert not list(tmp_path.glob('.mesh-job-*'))


def test_missing_validator_cannot_report_success(monkeypatch,tmp_path,cube):
    monkeypatch.setenv('MESH_GEOMETRY_WORKER',str(tmp_path/'absent-worker'))
    source=tmp_path/'source.stl'; output=tmp_path/'result.stl'
    write_stl(source,cube.vertices,cube.faces,cube.normals)
    _,log=MeshRepairPipeline(RepairConfig(backend_enabled=False)).repair(source,output,verbose=False)
    assert not log.success and not output.exists()
    assert log.validation.get('complete') is False


def test_rejected_candidate_rolls_back_geometry_and_counters(monkeypatch,tmp_path,cube):
    source=tmp_path/'open.stl'; output=tmp_path/'result.stl'
    write_stl(source,cube.vertices,cube.faces[2:],cube.normals[2:])
    original=load_mesh(source)
    def bad_patch(v,f,n,cfg,log):
        changed=v.copy(); changed[0]+=[10,0,0]
        log.faces_added+=999
        return changed,f,n
    monkeypatch.setattr(orchestration,'repair_planar_holes',bad_patch)
    mesh,log=MeshRepairPipeline(RepairConfig(backend_enabled=False))._repair_in_process(source,output)
    assert not log.success and log.faces_added==0
    assert np.array_equal(mesh.vertices,original.vertices)
    assert not output.exists()


@pytest.mark.parametrize('kind',['empty','truncated','nonfinite'])
def test_malformed_input_is_explicit_without_replacement(tmp_path,cube,kind):
    source=tmp_path/'bad.stl'; output=tmp_path/'result.stl'
    if kind=='empty':
        source.write_bytes(b'\0'*80+struct.pack('<I',0))
    else:
        write_stl(source,cube.vertices,cube.faces,cube.normals)
        if kind=='truncated':
            source.write_bytes(source.read_bytes()[:-2])
        else:
            data=bytearray(source.read_bytes()); data[96:100]=struct.pack('<f',float('nan'))
            source.write_bytes(data)
    _,log=MeshRepairPipeline().repair(source,output,verbose=False)
    assert log.status=='INVALID_INPUT' and not output.exists()


def test_adaptive_configuration_does_not_leak_between_models(tmp_path,cube):
    cfg=RepairConfig(adaptive=True)
    before=cfg.merge_tolerance
    pipeline=MeshRepairPipeline(cfg)
    for factor in [1.,1e-7]:
        source=tmp_path/f'{factor}.stl'; out=tmp_path/f'{factor}-out.stl'
        v=cube.vertices*factor
        write_stl(source,v,cube.faces,_recompute_normals(v,cube.faces))
        _,log=pipeline.repair(source,out,verbose=False)
        assert log.success,log.summary()
    assert cfg.merge_tolerance==before and cfg._deadline is None


def test_explicit_union_uses_material_boundary_and_keeps_voids(tmp_path,cube):
    v=np.vstack([cube.vertices,cube.vertices+[.4,.3,.2]])
    f=np.vstack([cube.faces,cube.faces+8])
    source=tmp_path/'overlap.stl'; output=tmp_path/'united.stl'
    write_stl(source,v,f,_recompute_normals(v,f))
    mesh,log=MeshRepairPipeline(RepairConfig(union_enabled=True)).repair(source,output,verbose=False)
    assert log.success,log.summary()
    assert np.isclose(_signed_volume(mesh.vertices,mesh.faces),1.664,atol=1e-6)


def test_positive_nested_shell_is_not_silently_changed_into_void(tmp_path,cube):
    v=np.vstack([cube.vertices*3,cube.vertices+1])
    f=np.vstack([cube.faces,cube.faces+8])
    source=tmp_path/'nested.stl'; output=tmp_path/'result.stl'
    write_stl(source,v,f,_recompute_normals(v,f))
    _,log=MeshRepairPipeline().repair(source,output,verbose=False)
    assert not log.success and not output.exists()
    # Explicit union defines occupied volume and can remove redundant interiors.
    mesh,log=MeshRepairPipeline(RepairConfig(union_enabled=True)).repair(source,output,verbose=False)
    assert log.success,log.summary()
    assert np.isclose(_signed_volume(mesh.vertices,mesh.faces),27)


def test_known_closed_part_cannot_be_deleted_with_loose_distance_budget(cube):
    from src.core.validation import closed_part_bounds,closed_parts_preserved
    v=np.vstack([cube.vertices,cube.vertices*.01+[2,2,2]])
    f=np.vstack([cube.faces,cube.faces+8])
    source=MeshData(v,f,_recompute_normals(v,f))
    assert not closed_parts_preserved(closed_part_bounds(source),cube,RepairConfig(max_surface_deviation=3))


def test_native_hole_triangulation_keeps_boundary_geometry(cube):
    from src.core.geometry_backend import execute
    source=MeshData(cube.vertices,cube.faces[2:],cube.normals[2:])
    target=execute('fill',source,RepairConfig())
    assert validate(target,RepairConfig()).valid
    accepted,_,_=surface_preservation(cube,target,RepairConfig(),bidirectional=True)
    assert accepted


def test_3mf_units_are_preserved(tmp_path,cube):
    cube.units='inch'
    source=tmp_path/'inches.3mf'; output=tmp_path/'result.3mf'
    write_mesh(source,cube)
    _,log=MeshRepairPipeline().repair(source,output,verbose=False)
    assert log.success
    assert load_mesh(output).units=='inch'


def test_autorefinement_preserves_intersecting_triangle_surface():
    from src.core.geometry_backend import execute
    v=np.array([[0,0,0],[1,0,.1],[0,1,.3],[.2,.1,-1],[.2,.9,1],[.2,-.1,.7]],dtype=float)
    f=np.array([[0,1,2],[3,4,5]],dtype=int)
    original=MeshData(v,f,_recompute_normals(v,f))
    refined=execute('refine',original,RepairConfig())
    accepted,_,_=surface_preservation(original,refined,RepairConfig(),bidirectional=True)
    assert accepted


def test_refined_crossing_objects_cannot_hide_intersections_at_new_edges(cube):
    from src.core.geometry_backend import execute
    from src.core.repair_utils import _compact_mesh
    v=np.vstack([cube.vertices,cube.vertices+[.4,.3,.2]])
    f=np.vstack([cube.faces,cube.faces+8])
    original=MeshData(v,f,_recompute_normals(v,f))
    refined=execute('refine',original,RepairConfig())
    triangles=refined.vertices[refined.faces]
    belongs_a=np.any(np.all(triangles==0,axis=1)|np.all(triangles==1,axis=1),axis=1)
    va,fa=_compact_mesh(refined.vertices,refined.faces[belongs_a])
    vb,fb=_compact_mesh(refined.vertices,refined.faces[~belongs_a])
    v=np.vstack([va,vb]); f=np.vstack([fa,fb+len(va)])
    report=validate(MeshData(v,f,_recompute_normals(v,f)),RepairConfig())
    assert report.naked_edges==report.non_manifold_edges==report.non_manifold_vertices==0
    assert report.self_intersection and not report.valid


def test_void_touching_outer_boundary_is_not_a_valid_sealed_cavity(cube):
    v=np.vstack([cube.vertices*3,cube.vertices+[0,1,1]])
    f=np.vstack([cube.faces,cube.faces[:,[0,2,1]]+8])
    report=validate(MeshData(v,f,_recompute_normals(v,f)),RepairConfig())
    assert report.complete and not report.valid and report.self_intersection


def test_duplicate_indexed_solids_are_not_valid_material_boundary(cube):
    v=np.vstack([cube.vertices,cube.vertices])
    f=np.vstack([cube.faces,cube.faces+8])
    report=validate(MeshData(v,f,_recompute_normals(v,f)),RepairConfig())
    assert report.self_intersection and not report.valid


@pytest.mark.parametrize('permuted',[False,True])
def test_redundant_coplanar_face_can_be_removed_without_shape_loss(tmp_path,cube,permuted):
    v=np.vstack([cube.vertices,[.5,0,.5]])
    f=np.vstack([cube.faces,[[0,1,8]]])
    if permuted:
        f=f[np.random.default_rng(20261010).permutation(len(f))]
    source=tmp_path/'overlaid.npz'; output=tmp_path/'repaired.3mf'
    np.savez(source,vertices=v,faces=f)
    mesh,log=MeshRepairPipeline().repair(source,output,verbose=False)
    assert log.success,log.summary()
    accepted,_,_=surface_preservation(cube,mesh,RepairConfig(),bidirectional=True)
    assert accepted
