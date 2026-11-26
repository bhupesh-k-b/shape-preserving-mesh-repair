import numpy as np
import pytest

from src.core import geometry_backend
from src.core.mesh_analyzer import MeshData, build_edge_map, check_naked_and_nonmanifold
from src.core.mesh_io import write_mesh
from src.core.repair_config import RepairConfig, RepairLog
from src.core.repair_utils import _recompute_normals, _signed_volume
from src.core.repair_backends import candidate
from src.core.validation import surface_preservation, validate
from src.pipelines import mesh_repair_pipeline as orchestration
from src.routines.repair_winding import check_winding_consistency, repair_winding_order


@pytest.mark.parametrize('scale,offset', [(1.,0.), (1e-7,0.), (1.,1e6)])
def test_exact_coplanar_cleanup_removes_covered_triangle_without_moving_surface(cube, scale, offset):
    v = np.vstack([cube.vertices, [[.5,.25,1.]]]) * scale + offset
    f = np.vstack([cube.faces, [[4,5,8]]])
    source = MeshData(v, f, _recompute_normals(v,f))
    cleaned = geometry_backend.execute('clean_coplanar',source,RepairConfig())
    assert len(cleaned.faces) == len(cube.faces)
    assert np.array_equal(cleaned.vertices,source.vertices)
    assert validate(cleaned,RepairConfig().for_mesh(v,f)).valid
    assert surface_preservation(source,cleaned,RepairConfig(),bidirectional=True)[0]


def test_coplanar_cleanup_keeps_thin_parallel_surfaces_and_true_crossings():
    v = np.array([[0,0,0],[1,0,0],[0,1,0],
                  [0,0,1e-9],[1,0,1e-9],[0,1,1e-9],
                  [.2,.1,-1],[.2,.9,1],[.2,-.1,.7]],dtype=float)
    f = np.array([[0,1,2],[3,4,5],[6,7,8]],dtype=int)
    source = MeshData(v,f,_recompute_normals(v,f))
    cleaned = geometry_backend.execute('clean_coplanar',source,RepairConfig())
    assert np.array_equal(cleaned.faces,source.faces)
    assert np.array_equal(cleaned.vertices,source.vertices)


def test_exact_intersection_diagnostics_report_bounded_pairs():
    triangle = np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=float)
    v = np.tile(triangle,(130,1))
    f = np.arange(len(v)).reshape(-1,3)
    source = MeshData(v,f,_recompute_normals(v,f))
    details = geometry_backend.execute('intersections',source,RepairConfig())
    assert len(details['pairs']) == 8192
    assert details['complete'] is False
    assert all(a != b and a < len(f) and b < len(f) for a,b in details['pairs'])


def test_native_fill_receives_best_preserved_candidate_and_matching_counters(tmp_path,cube,monkeypatch):
    v = np.vstack([cube.vertices, [[.5,.25,1.]]])
    f = np.vstack([cube.faces[1:], [[4,5,8]]])
    source = tmp_path/'open-with-overlay.3mf'
    output = tmp_path/'repaired.3mf'
    write_mesh(source,MeshData(v,f,_recompute_normals(v,f)))
    # Keep the first route incomplete so that the excision/native handoff is
    # exercised independently of the local cap and coplanar generators.
    monkeypatch.setattr(orchestration,'repair_topology_separation',lambda v,f,*args:(v,f))
    monkeypatch.setattr(orchestration,'repair_planar_holes',lambda v,f,n,*args:(v,f,n))

    def excise(v,f,n,config,log):
        log.non_manifold_removed += 1
        return f[:-1],n[:-1]

    monkeypatch.setattr(orchestration,'repair_non_manifold',excise)
    execute = geometry_backend.execute
    observed = []

    def capture(action,mesh,*args,**kwargs):
        if action == 'clean_coplanar':
            return mesh
        if action == 'fill':
            _,nm = check_naked_and_nonmanifold(build_edge_map(mesh.faces))
            observed.append(len(nm))
        return execute(action,mesh,*args,**kwargs)

    monkeypatch.setattr(geometry_backend,'execute',capture)
    _,log = orchestration.MeshRepairPipeline()._repair_in_process(source,output)
    assert log.success,log.summary()
    assert observed == [0]
    assert log.strategy == 'CGAL-hole-triangulation'
    assert log.non_manifold_removed == 1


def test_explicit_union_keeps_local_hole_preparation(tmp_path,cube):
    source = tmp_path/'open.3mf'
    output = tmp_path/'closed.3mf'
    write_mesh(source,MeshData(cube.vertices,cube.faces[1:],cube.normals[1:]))
    repaired,log = orchestration.MeshRepairPipeline(RepairConfig(union_enabled=True)).repair(source,output,verbose=False)
    assert log.success,log.summary()
    assert surface_preservation(cube,repaired,RepairConfig(),bidirectional=True)[0]


@pytest.mark.parametrize('seed',range(4))
def test_cached_orientation_audit_and_repair_agree_with_independent_checker(cube,seed):
    import trimesh
    rng=np.random.default_rng(seed)
    f=cube.faces[rng.permutation(len(cube.faces))].copy()
    flip=rng.random(len(f))<.5
    f[flip]=f[flip][:,[0,2,1]]
    independent=trimesh.Trimesh(cube.vertices,f,process=False)
    audit=check_winding_consistency(cube.vertices,f,edge_map=build_edge_map(f),analyze_shells=False)
    assert bool(audit.conflicted_edges)==(not independent.is_winding_consistent)
    v,out,_=repair_winding_order(cube.vertices,f,_recompute_normals(cube.vertices,f),RepairConfig(),RepairLog())
    independent=trimesh.Trimesh(v,out,process=False)
    assert independent.is_winding_consistent and independent.volume>0
    assert np.array_equal(v,cube.vertices)


def test_large_coplanar_overlay_is_removed_without_refining_original_cube(tmp_path,cube):
    n=10
    grid=np.array([[i/n,j/n,1.] for i in range(n+1) for j in range(n+1)])
    triangles=[]
    for i in range(n):
        for j in range(n):
            a=8+i*(n+1)+j
            b=a+n+1
            triangles.extend([[a,b,b+1],[a,b+1,a+1]])
    v=np.vstack([cube.vertices,grid])
    f=np.vstack([cube.faces,triangles])
    source=tmp_path/'dense-overlay.3mf'; output=tmp_path/'cube.3mf'
    write_mesh(source,MeshData(v,f,_recompute_normals(v,f)))
    result,log=orchestration.MeshRepairPipeline().repair(source,output,verbose=False)
    assert log.success,log.summary()
    assert log.strategy=='CGAL-covered-coplanar-cleanup'
    assert log.non_manifold_removed==200
    assert surface_preservation(cube,result,RepairConfig(),bidirectional=True)[0]


def test_explicit_union_cannot_trust_closed_but_self_intersecting_input(cube):
    v=cube.vertices.copy()
    v[6]=[-.3,.2,.4]
    source=MeshData(v,cube.faces,_recompute_normals(v,cube.faces))
    report=validate(source,RepairConfig())
    assert report.self_intersection and report.naked_edges==report.non_manifold_edges==0
    with pytest.raises(RuntimeError,match='non-self-intersecting input sheets'):
        candidate('manifold',source,RepairConfig(union_enabled=True))


def test_explicit_union_preserving_voids_requires_valid_cavity_interfaces(cube):
    v=np.vstack([cube.vertices*3,cube.vertices+[0,1,1]])
    f=np.vstack([cube.faces,cube.faces[:,[0,2,1]]+8])
    source=MeshData(v,f,_recompute_normals(v,f))
    with pytest.raises(RuntimeError,match='valid cavity interfaces'):
        candidate('manifold',source,RepairConfig(union_enabled=True))


@pytest.mark.parametrize('corrupted_shell',['inner','outer'])
def test_inconsistent_nested_winding_cannot_silently_invent_a_cavity(tmp_path,cube,corrupted_shell):
    v=np.vstack([cube.vertices*3,cube.vertices+1])
    f=np.vstack([cube.faces,cube.faces+8])
    if corrupted_shell=='inner':
        f[-1]=f[-1,[0,2,1]]
    else:
        # Cancel the outer's signed-volume cue while preserving its surface.
        f[[2,3,6]]=f[[2,3,6]][:,[0,2,1]]
        assert _signed_volume(v,f[:12])==0
    source=tmp_path/'ambiguous.3mf'; output=tmp_path/'result.3mf'
    write_mesh(source,MeshData(v,f,_recompute_normals(v,f)))
    _,log=orchestration.MeshRepairPipeline().repair(source,output,verbose=False)
    assert not log.success and not output.exists()
    assert log.validation.get('ambiguous_material_roles',0)>0,log.summary()


def test_explicit_cavity_fill_overrides_declared_hollow_input(tmp_path,cube):
    v=np.vstack([cube.vertices*3,cube.vertices+1])
    f=np.vstack([cube.faces,cube.faces[:,[0,2,1]]+8])
    source=tmp_path/'hollow.3mf'; output=tmp_path/'filled.3mf'
    write_mesh(source,MeshData(v,f,_recompute_normals(v,f)))
    cfg=RepairConfig(union_enabled=True,union_hollow=True,preserve_cavities=False)
    repaired,log=orchestration.MeshRepairPipeline(cfg).repair(source,output,verbose=False)
    assert log.success,log.summary()
    assert np.isclose(_signed_volume(repaired.vertices,repaired.faces),27)
