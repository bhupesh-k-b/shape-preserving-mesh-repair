"""Precision-aware STL and indexed multi-object 3MF I/O."""
from pathlib import Path
import struct
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from src.core.mesh_analyzer import MeshData, build_edge_map
from src.core.repair_utils import _recompute_normals, _compact_mesh

NS='http://schemas.microsoft.com/3dmanufacturing/core/2015/02'


def write_stl(path,vertices,faces,normals=None):
    with np.errstate(over='ignore',invalid='ignore'):
        triangles=np.asarray(vertices)[faces].astype(np.float32)
    if not np.all(np.isfinite(triangles)):
        raise ValueError('Geometry cannot be represented as finite STL float32 coordinates')
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
    records=np.zeros(len(faces),dtype=dtype)
    records['normal']=_recompute_normals(vertices,faces)
    records['vertices']=triangles
    with open(path,'wb') as handle:
        handle.write(b'Shape-preserving MeshRepairPipeline'.ljust(80,b'\0'))
        handle.write(struct.pack('<I',len(faces)))
        handle.write(records.tobytes())


def write_3mf(path,mesh,units=None):
    from src.routines.repair_shell_union import _find_shells, _build_shell_infos, classify_shells
    ET.register_namespace('',NS)
    model=ET.Element(f'{{{NS}}}model',unit=units or mesh.units or 'millimeter')
    resources=ET.SubElement(model,f'{{{NS}}}resources'); build=ET.SubElement(model,f'{{{NS}}}build')
    em=build_edge_map(mesh.faces)
    shells=_find_shells(mesh.faces,em,manifold_only=True)
    infos=classify_shells(mesh.vertices,mesh.faces,_build_shell_infos(mesh.vertices,mesh.faces,shells,em))
    groups={}
    for info in infos:
        root=info.index
        while infos[root].parent_index is not None:
            root=infos[root].parent_index
        groups.setdefault(root,[]).extend(info.face_indices)
    # Cavities belong to their material object, never to independent positive
    # build objects that a slicer might interpret as filled solids.
    for index,shell in enumerate(groups.values(),1):
        obj=ET.SubElement(resources,f'{{{NS}}}object',id=str(index),type='model')
        geometry=ET.SubElement(obj,f'{{{NS}}}mesh')
        v,f=_compact_mesh(mesh.vertices,mesh.faces[np.array(shell)])
        vertices=ET.SubElement(geometry,f'{{{NS}}}vertices')
        for p in v:
            ET.SubElement(vertices,f'{{{NS}}}vertex',x=format(p[0],'.17g'),y=format(p[1],'.17g'),z=format(p[2],'.17g'))
        triangles=ET.SubElement(geometry,f'{{{NS}}}triangles')
        for face in f:
            ET.SubElement(triangles,f'{{{NS}}}triangle',v1=str(face[0]),v2=str(face[1]),v3=str(face[2]))
        ET.SubElement(build,f'{{{NS}}}item',objectid=str(index))
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('3D/3dmodel.model',ET.tostring(model,encoding='utf-8',xml_declaration=True))
        archive.writestr('[Content_Types].xml','<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        archive.writestr('_rels/.rels','<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')


def load_3mf(path):
    with zipfile.ZipFile(path) as archive:
        root=ET.fromstring(archive.read('3D/3dmodel.model'))
    units=root.attrib.get('unit','millimeter')
    if units not in {'micron','millimeter','centimeter','inch','foot','meter'}:
        raise ValueError('Unsupported 3MF unit')
    objects={obj.attrib['id']:obj for obj in root.findall(f'{{{NS}}}resources/{{{NS}}}object')}
    vertices=[]; faces=[]; count=0
    build=root.find(f'{{{NS}}}build')
    if build is None:
        raise ValueError('3MF build is missing')
    for item in build:
        if 'transform' in item.attrib:
            raise ValueError('Transformed 3MF assemblies require a dedicated importer')
        obj=objects[item.attrib['objectid']]
        geometry=obj.find(f'{{{NS}}}mesh')
        if geometry is None:
            raise ValueError('3MF component assemblies require a dedicated importer')
        v=np.array([[float(vertex.attrib[axis]) for axis in ('x','y','z')]
                    for vertex in geometry.find(f'{{{NS}}}vertices')],dtype=float)
        f=np.array([[int(triangle.attrib[axis]) for axis in ('v1','v2','v3')]
                    for triangle in geometry.find(f'{{{NS}}}triangles')],dtype=np.int64)
        vertices.append(v); faces.append(f+count); count+=len(v)
    if not faces:
        raise ValueError('Empty 3MF')
    v=np.vstack(vertices); f=np.vstack(faces)
    return MeshData(v,f,_recompute_normals(v,f),units=root.attrib.get('unit','millimeter'))


def load_mesh(path):
    if Path(path).suffix.lower()=='.3mf':
        mesh=load_3mf(path)
    elif Path(path).suffix.lower()=='.npz':
        with np.load(path,allow_pickle=False) as data:
            v=np.asarray(data['vertices'],dtype=float)
            f=np.asarray(data['faces'] if 'faces' in data else data['facets'])
        if not np.all(np.isfinite(f)) or np.any(f!=np.floor(f)):
            raise ValueError('Nonintegral triangle indices')
        if f.size and (np.min(f)<0 or np.max(f)>=len(v)):
            raise ValueError('Invalid triangle indices')
        mesh=MeshData(v,f,_recompute_normals(v,f.astype(np.int64)))
    else:
        mesh=MeshData.from_stl(str(path))
    if not len(mesh.faces) or not len(mesh.vertices):
        raise ValueError('Empty input mesh')
    if not np.all(np.isfinite(mesh.vertices)) or np.min(mesh.faces)<0 or np.max(mesh.faces)>=len(mesh.vertices):
        raise ValueError('Nonfinite coordinates or invalid triangle indices')
    return mesh


def write_mesh(path,mesh,units=None):
    if Path(path).suffix.lower()=='.3mf':
        write_3mf(path,mesh,units)
    else:
        write_stl(path,mesh.vertices,mesh.faces,mesh.normals)
