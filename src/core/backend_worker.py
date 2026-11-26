"""Native geometry candidate generation in an expendable subprocess."""
import json
import sys
import numpy as np
from src.core.mesh_analyzer import MeshData, build_edge_map
from src.core.repair_utils import _signed_volume, _compact_mesh, _recompute_normals
from src.routines.repair_shell_union import _find_shells, _build_shell_infos, classify_shells


def manifold_candidate(mesh,preserve_cavities,hollow=False):
    import manifold3d as m
    em=build_edge_map(mesh.faces)
    shells=_find_shells(mesh.faces,em,manifold_only=True)
    infos=classify_shells(mesh.vertices,mesh.faces,_build_shell_infos(mesh.vertices,mesh.faces,shells,em))
    solids=[]
    roles=[]
    for info in infos:
        if info.is_open:
            raise ValueError('Manifold Boolean requires closed input sheets')
        f=mesh.faces[np.array(info.face_indices)]
        if _signed_volume(mesh.vertices,f)<0:
            f=f[:,[0,2,1]]
        v,f=_compact_mesh(mesh.vertices,f)
        solid=m.Manifold(m.Mesh64(np.ascontiguousarray(v),np.ascontiguousarray(f,dtype=np.uint64),tolerance=0))
        if solid.status()!=m.Error.NoError or solid.is_empty():
            raise ValueError(f'Invalid Manifold input: {solid.status()}')
        solids.append(solid)
        root=info.index
        while infos[root].parent_index is not None:
            root=infos[root].parent_index
        roles.append(bool(info.depth%2) if hollow else info.depth>0 and info.volume*infos[root].volume<0)
    children={info.index:[] for info in infos}
    roots=[]
    for info in infos:
        if preserve_cavities and info.parent_index is not None:
            children[info.parent_index].append(info.index)
        else:
            roots.append(info.index)
    def material(index):
        shape=solids[index]
        for child in children[index]:
            region=material(child)
            shape=shape-region if roles[child]!=roles[index] else shape+region
        return shape
    result=m.Manifold.batch_boolean([material(i) for i in roots],m.OpType.Add)
    if result.status()!=m.Error.NoError or result.is_empty():
        raise ValueError(f'Manifold Boolean failed: {result.status()}')
    raw=result.to_mesh64()
    v=np.array(raw.vert_properties[:,:3],dtype=float)
    f=np.array(raw.tri_verts,dtype=np.int64)
    remap=np.arange(len(v))
    for a,b in zip(raw.merge_from_vert,raw.merge_to_vert):
        remap[int(a)]=int(b)
    for _ in range(8):
        new=remap[remap]
        if np.array_equal(new,remap):
            break
        remap=new
    v,f=_compact_mesh(v,remap[f])
    return MeshData(v,f,_recompute_normals(v,f))


def partial_volume_overlap(mesh):
    """Exact-contact objects are allowed; positive-volume partial overlap is not."""
    import manifold3d as m
    em=build_edge_map(mesh.faces)
    shells=_find_shells(mesh.faces,em,manifold_only=True)
    bodies=[]; bounds=[]
    for shell in shells:
        f=mesh.faces[np.array(shell)]
        if _signed_volume(mesh.vertices,f)<0:
            f=f[:,[0,2,1]]
        v,f=_compact_mesh(mesh.vertices,f)
        body=m.Manifold(m.Mesh64(np.ascontiguousarray(v),np.ascontiguousarray(f,dtype=np.uint64),tolerance=0))
        if body.status()!=m.Error.NoError or body.is_empty():
            raise ValueError('Unable to independently validate component volumes')
        bodies.append(body); bounds.append((v.min(axis=0),v.max(axis=0)))
    for i,a in enumerate(bodies):
        for j in range(i+1,len(bodies)):
            if np.any(bounds[i][1]<=bounds[j][0]) or np.any(bounds[j][1]<=bounds[i][0]):
                continue
            b=bodies[j]
            intersection=a^b
            if intersection.status()!=m.Error.NoError:
                raise ValueError('Component intersection verification failed')
            if intersection.is_empty():
                continue
            remainder_a=a-b; remainder_b=b-a
            if remainder_a.status()!=m.Error.NoError or remainder_b.status()!=m.Error.NoError:
                raise ValueError('Component containment verification failed')
            if remainder_a.is_empty() and remainder_b.is_empty():
                return True  # coincident solids are unresolved duplicate boundaries
            if not remainder_a.is_empty() and not remainder_b.is_empty():
                return True
    return False


def main():
    action,source,destination,options=sys.argv[1:]
    config=json.loads(options)
    with np.load(source,allow_pickle=False) as data:
        mesh=MeshData(data['vertices'],data['faces'],data['normals'])
    if action=='overlap':
        np.savez(destination,interior_overlap=partial_volume_overlap(mesh))
        return
    if action=='manifold':
        result=manifold_candidate(mesh,config['preserve_cavities'],config.get('hollow',False))
    elif action=='meshfix':
        import pymeshfix
        fix=pymeshfix.MeshFix(mesh.vertices,mesh.faces)
        fix.repair(joincomp=False,remove_smallest_components=False)
        v,f=np.asarray(fix.points),np.asarray(fix.faces)
        result=MeshData(v,f,_recompute_normals(v,f))
    else:
        raise ValueError('Unknown native candidate backend')
    np.savez(destination,vertices=result.vertices,faces=result.faces,normals=result.normals)


if __name__=='__main__':
    main()
