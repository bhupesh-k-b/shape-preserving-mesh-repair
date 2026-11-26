"""Deadline-bound, isolated CGAL operations. No silent validator fallback."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import time
import numpy as np
from src.core.mesh_analyzer import MeshData
from src.core.repair_utils import _recompute_normals

ROOT = Path(__file__).resolve().parents[2]


def worker_path():
    return Path(os.environ.get('MESH_GEOMETRY_WORKER', str(ROOT/'.native-build/mesh_geometry_worker')))


def available():
    return worker_path().is_file()


def write_off(path, mesh):
    with open(path, 'w', encoding='ascii') as handle:
        handle.write(f'OFF\n{len(mesh.vertices)} {len(mesh.faces)} 0\n')
        np.savetxt(handle, mesh.vertices, fmt='%.17g')
        np.savetxt(handle, np.column_stack([np.full(len(mesh.faces),3),mesh.faces]), fmt='%d')


def read_off(path):
    with open(path, encoding='ascii') as handle:
        if handle.readline().strip() != 'OFF':
            raise ValueError('Invalid geometry worker output')
        nv,nf,_ = map(int,handle.readline().split())
        vertices=np.array([list(map(float,handle.readline().split())) for _ in range(nv)],dtype=float)
        rows=np.array([list(map(int,handle.readline().split())) for _ in range(nf)],dtype=np.int64)
    if rows.shape != (nf,4) or np.any(rows[:,0]!=3):
        raise ValueError('Invalid geometry worker triangles')
    faces=rows[:,1:]
    return MeshData(vertices,faces,_recompute_normals(vertices,faces))


def execute(action, source, config, target=None, error=None):
    config.check_deadline()
    if not available():
        raise RuntimeError('CGAL validator unavailable: build native/ with CMake or set MESH_GEOMETRY_WORKER')
    with tempfile.TemporaryDirectory(prefix='mesh-geometry-') as directory:
        source_path=Path(directory)/'source.off'
        write_off(source_path,source)
        args=[str(worker_path()),action,str(source_path)]
        if action in {'refine','fill','clean_coplanar'}:
            result_path=Path(directory)/'refined.off'
            args.append(str(result_path))
            if action=='fill':
                args.append(str(config.max_hole_edges))
        elif target is not None:
            target_path=Path(directory)/'target.off'
            write_off(target_path,target)
            args.extend([str(target_path),format(error,'.17g')])
        remaining=(config._deadline-time.perf_counter()) if config._deadline else config.timeout_seconds
        if remaining <= 0:
            raise TimeoutError('Geometry deadline exceeded')
        try:
            run=subprocess.run(args,capture_output=True,text=True,timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError('CGAL operation exceeded the geometry deadline') from exc
        if run.returncode:
            raise RuntimeError(run.stderr.strip() or 'CGAL operation failed')
        config.check_deadline()
        result=json.loads(run.stdout)
        if action in {'refine','fill','clean_coplanar'}:
            return read_off(result_path)
        return result


def intersects(mesh, config):
    from src.core.mesh_analyzer import build_edge_map
    from src.routines.repair_shell_union import _find_shells
    em=build_edge_map(mesh.faces)
    shells=_find_shells(mesh.faces,em)
    raw_intersection=execute('check',mesh,config)['self_intersection']
    if not raw_intersection:
        return False
    if len(shells)<=1 or any(len(ids)!=2 for ids in em.values()):
        return True
    # Never merge coordinate identities merely to make intersections disappear.
    # Separate closed objects may touch, but may not partially overlap in volume.
    if execute('check_components',mesh,config)['self_intersection']:
        return True
    from src.core.repair_backends import candidate
    return candidate('overlap',mesh,config)


def distance_bound(source,target,config,error):
    result=execute('distance',source,config,target,error)
    # Accept using the upper error envelope, not the returned estimate alone.
    return float(result['distance'])+float(result['error_bound'])


def require_boolean_sheets(mesh,config):
    """A declared solid Boolean needs independently valid input sheets.

    Overlap between positive solids is the operation's purpose. An internally
    self-intersecting sheet is not a valid solid merely because its edge counts
    pass or the Boolean library's constructor accepts it.
    """
    from src.core.mesh_analyzer import build_edge_map
    from src.core.validation import nonmanifold_vertices
    from src.routines.repair_winding import check_winding_consistency
    if config.preserve_cavities and not config.union_hollow and config._ambiguous_material_roles:
        raise RuntimeError('Explicit union needs declared material roles for nested source sheets with inconsistent winding')
    em=build_edge_map(mesh.faces)
    if not len(mesh.faces) or any(len(indices)!=2 for indices in em.values()) or nonmanifold_vertices(mesh.faces,em):
        raise RuntimeError('Explicit union requires closed manifold input sheets')
    if check_winding_consistency(mesh.vertices,mesh.faces,edge_map=em,analyze_shells=False).conflicted_edges:
        raise RuntimeError('Explicit union requires coherently oriented input sheets')
    if execute('check_components',mesh,config)['self_intersection']:
        raise RuntimeError('Explicit union requires non-self-intersecting input sheets and valid cavity interfaces')


def covers(source,target,config,budget):
    return execute('covers',source,config,target,budget)['covered']
