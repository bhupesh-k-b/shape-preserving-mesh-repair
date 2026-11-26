"""Topology, geometry, material, and preservation acceptance criteria."""
from dataclasses import dataclass, field, asdict
import hashlib
import numpy as np
from src.core.mesh_analyzer import MeshData, build_edge_map, check_naked_and_nonmanifold, check_duplicate_faces
from src.core.repair_utils import _recompute_normals, _signed_volume
from src.routines.repair_winding import check_winding_consistency
from src.core import geometry_backend


def face_keys(mesh):
    triangles=mesh.vertices[mesh.faces].copy()
    triangles[triangles==0]=0.0  # canonicalize signed zero
    order=np.lexsort((triangles[:,:,2],triangles[:,:,1],triangles[:,:,0]),axis=1)
    ordered=np.take_along_axis(triangles,order[:,:,None],axis=1)
    packed=np.ascontiguousarray(ordered,dtype=np.float64).reshape(-1,9)
    return packed.view(np.dtype((np.void,packed.dtype.itemsize*9))).ravel()


def numeric_budget(mesh):
    points=mesh.vertices[mesh.faces]
    scale=max(float(np.max(np.abs(points))),float(np.linalg.norm(np.ptp(points.reshape(-1,3),axis=0))),np.finfo(float).tiny)
    lengths=np.column_stack([np.linalg.norm(points[:,i]-points[:,(i+1)%3],axis=1) for i in range(3)])
    longest=lengths.max(axis=1)
    areas2=np.linalg.norm(np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]),axis=1)
    altitudes=np.divide(areas2,longest,out=np.zeros_like(areas2),where=longest>0)
    positive=np.concatenate([lengths[lengths>0],altitudes[altitudes>0]])
    cap=float(np.min(positive))*1e-6 if len(positive) else 0.
    return min(128*np.finfo(float).eps*scale,cap)


def closed_part_bounds(mesh):
    from src.routines.repair_shell_union import _find_shells
    em=build_edge_map(mesh.faces)
    parts=[]
    for shell in _find_shells(mesh.faces,em):
        f=mesh.faces[np.array(shell)]
        closed=all(len(em[(min(int(face[k]),int(face[(k+1)%3])),max(int(face[k]),int(face[(k+1)%3])))])==2 for face in f for k in range(3))
        if not closed:
            continue
        v=mesh.vertices[np.unique(f)]
        subset=MeshData(mesh.vertices,f,mesh.normals[np.array(shell)])
        parts.append((v.min(axis=0),v.max(axis=0),numeric_budget(subset)))
    return parts


def closed_parts_preserved(source_parts,target,config):
    if config.union_enabled:
        return True
    targets=closed_part_bounds(target)
    used=set()
    for lower,upper,epsilon in source_parts:
        budget=config.max_surface_deviation+epsilon
        match=next((i for i,(a,b,_) in enumerate(targets) if i not in used and
                    np.all(np.abs(lower-a)<=budget) and np.all(np.abs(upper-b)<=budget)),None)
        if match is None:
            return False
        used.add(match)
    return True


def surface_preservation(source,target,config,bidirectional=False):
    """Exact facet retention certificate, otherwise a bounded CGAL distance."""
    source_keys=face_keys(source)
    target_keys=face_keys(target)
    retained=np.isin(source_keys,target_keys)
    reverse=np.isin(target_keys,source_keys) if bidirectional else None
    if np.all(retained) and (not bidirectional or np.all(reverse)):
        return True,0.0,'exact_facets'
    budget=config.max_surface_deviation+numeric_budget(source)
    error=max(budget/16,np.finfo(float).tiny)
    missing=source.faces[~retained]
    upper=0.0
    if len(missing):
        subset=MeshData(source.vertices,missing,_recompute_normals(source.vertices,missing))
        if geometry_backend.covers(subset,target,config,budget):
            upper=budget
        else:
            # An unresolved coverage certificate is not accepted. Generic
            # Hausdorff subdivision at machine precision can be unboundedly
            # expensive, so only use it for explicitly nonzero user budgets.
            if config.max_surface_deviation==0:
                return False,float('inf'),'uncertified_surface_coverage'
            upper=geometry_backend.distance_bound(subset,target,config,error)
    if bidirectional and np.any(~reverse):
        f=target.faces[~reverse]
        subset=MeshData(target.vertices,f,_recompute_normals(target.vertices,f))
        if geometry_backend.covers(subset,source,config,budget):
            upper=max(upper,budget)
        elif config.max_surface_deviation==0:
            return False,float('inf'),'uncertified_surface_coverage'
        else:
            upper=max(upper,geometry_backend.distance_bound(subset,source,config,error))
    return upper<=budget,upper,'certified_surface_coverage'


def nonmanifold_vertices(faces, edge_map=None):
    """A manifold vertex has one connected incident fan, including boundaries."""
    edge_map=edge_map or build_edge_map(faces)
    incident={}
    for fi,face in enumerate(faces):
        for vi in face:
            incident.setdefault(int(vi),set()).add(fi)
    links={vi:{} for vi in incident}
    for (a,b),flist in edge_map.items():
        if len(flist)!=2:
            continue
        x,y=flist
        for vi in (a,b):
            links[vi].setdefault(x,set()).add(y)
            links[vi].setdefault(y,set()).add(x)
    bad=[]
    for vi,fan in incident.items():
        visited=set(); todo=[next(iter(fan))]
        while todo:
            fi=todo.pop()
            if fi in visited:
                continue
            visited.add(fi)
            todo.extend(links[vi].get(fi,()))
        if visited!=fan:
            bad.append(vi)
    return bad


@dataclass
class ValidationReport:
    complete: bool=False
    valid: bool=False
    face_count: int=0
    naked_edges: int=0
    non_manifold_edges: int=0
    non_manifold_vertices: int=0
    degenerate_faces: int=0
    duplicate_faces: int=0
    winding_conflicts: int=0
    self_intersection: bool=False
    cavity_orientation_errors: int=0
    ambiguous_material_roles: int=0
    zero_volume_shells: int=0
    reasons: list=field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def validate(mesh,config,geometry_cache=None):
    config.check_deadline()
    report=ValidationReport(face_count=len(mesh.faces))
    v,f=mesh.vertices,mesh.faces
    if not len(v) or not len(f):
        report.reasons=['Empty mesh']; return report
    if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3:
        report.reasons=['Invalid array shapes']; return report
    if not np.all(np.isfinite(v)) or np.min(f)<0 or np.max(f)>=len(v):
        report.reasons=['Nonfinite coordinates or invalid face indices']; return report
    if config.preserve_cavities and not config.union_hollow:
        # Ambiguity belongs to the source contract. Moving a candidate's bounds
        # cannot make an unknown source material role disappear.
        report.ambiguous_material_roles=len(config._ambiguous_material_roles)
    edge_map=build_edge_map(f)
    naked,nm=check_naked_and_nonmanifold(edge_map)
    report.naked_edges=len(naked); report.non_manifold_edges=len(nm)
    report.non_manifold_vertices=len(nonmanifold_vertices(f,edge_map))
    triangles=v[f]
    areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    report.degenerate_faces=int(np.count_nonzero(areas<=config.area_tolerance*2))
    report.duplicate_faces=sum(len(g)-1 for g in check_duplicate_faces(f))
    audit=check_winding_consistency(v,f,edge_map=edge_map,analyze_shells=False)
    report.winding_conflicts=len(audit.conflicted_edges)
    config.check_deadline()
    # Independently check geometry even if incidence alone would pass.
    digest=hashlib.sha256(v.tobytes()+f.tobytes()).digest()
    try:
        if geometry_cache is not None and digest in geometry_cache:
            report.self_intersection=geometry_cache[digest]
        else:
            report.self_intersection=geometry_backend.intersects(mesh,config)
            if geometry_cache is not None:
                geometry_cache[digest]=report.self_intersection
        report.complete=True
    except RuntimeError as exc:
        report.reasons.append(str(exc))
    # Only closed, consistent shells have meaningful signed material volume.
    if not naked and not nm and not report.winding_conflicts:
        from src.routines.repair_shell_union import _find_shells, _build_shell_infos, classify_shells
        infos=_build_shell_infos(v,f,_find_shells(f,edge_map,manifold_only=True),edge_map)
        infos=classify_shells(v,f,infos)
        for info in infos:
            volume=_signed_volume(v,f[np.array(info.face_indices)])
            if volume==0:
                report.zero_volume_shells+=1
            expected_negative=(config.preserve_cavities or config.union_hollow) and info.depth%2==1
            if volume!=0 and (volume<0)!=expected_negative:
                report.cavity_orientation_errors+=1
    for name in ['naked_edges','non_manifold_edges','non_manifold_vertices','degenerate_faces',
                 'duplicate_faces','winding_conflicts','cavity_orientation_errors','ambiguous_material_roles','zero_volume_shells']:
        if getattr(report,name):
            report.reasons.append(f'{name}: {getattr(report,name)}')
    if report.self_intersection:
        report.reasons.append('Self-intersecting surface')
    report.valid=report.complete and not report.reasons
    return report
