from pathlib import Path
import numpy as np
from src.core.mesh_analyzer import MeshData
from src.core.mesh_io import load_mesh
from src.core.repair_config import RepairConfig
from src.core.validation import face_keys,validate,surface_preservation,numeric_budget
from src.pipelines.mesh_repair_initial import InitialCleanupPipeline
from src.pipelines.mesh_repair_pipeline import MeshRepairPipeline

ROOT=Path(__file__).resolve().parents[1]


def assert_surface_preserved(original,result):
    triangles=original.vertices[original.faces]
    area2=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    active=area2>0
    reference=MeshData(original.vertices,original.faces[active],original.normals[active])
    accepted,_,_=surface_preservation(reference,result,RepairConfig())
    assert accepted
    # Independently sample the facets changed by legitimate T-junction splits.
    missing=~np.isin(face_keys(reference),face_keys(result))
    if np.any(missing):
        import trimesh
        t=reference.vertices[reference.faces[missing]]
        points=np.vstack([t.reshape(-1,3),t.mean(axis=1),((t+np.roll(t,1,axis=1))*.5).reshape(-1,3)])
        _,distance,_=trimesh.proximity.closest_point_naive(trimesh.Trimesh(result.vertices,result.faces,process=False),points)
        assert float(np.max(distance))<=numeric_budget(reference)


def test_chair_exact_cleanup_keeps_all_original_coordinates(tmp_path):
    source=ROOT/'data/test_meshes/Chair.stl'
    output=tmp_path/'chair.stl'
    original=MeshData.from_stl(source)
    InitialCleanupPipeline().repair(str(source),str(output),verbose=False)
    result=MeshData.from_stl(output)
    assert np.array_equal(np.sort(face_keys(original)),np.sort(face_keys(result)))


def test_plane_lower_pipeline_preserves_doubled_surface(tmp_path):
    source=ROOT/'data/test_meshes/Plane_lower.stl'
    output=tmp_path/'plane.stl'
    original=MeshData.from_stl(source)
    mesh,log=MeshRepairPipeline().repair(source,output,verbose=False)
    assert log.success,log.summary()
    assert len(mesh.faces)==742
    assert_surface_preserved(original,mesh)
    assert validate(load_mesh(output),RepairConfig()).valid


def test_chair_full_repair_adds_caps_without_distorting_existing_faces(tmp_path):
    source=ROOT/'data/test_meshes/Chair.stl'
    output=tmp_path/'chair-repaired.stl'
    original=MeshData.from_stl(source)
    mesh,log=MeshRepairPipeline().repair(source,output,verbose=False)
    assert log.success,log.summary()
    assert_surface_preserved(original,mesh)
