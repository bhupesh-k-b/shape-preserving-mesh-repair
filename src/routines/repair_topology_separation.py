"""
Module for robust topological separation of non-manifold junctions.
Resolves both non-manifold edges and non-manifold vertices by duplicating vertices.
"""

import math

import numpy as np
from collections import defaultdict, deque
from typing import Tuple, List, Dict, Set

from src.core.repair_config import RepairLog


def _edge_traversal_dir(face: np.ndarray, a: int, b: int) -> int:
    """+1 if `face` walks the edge a->b, -1 if b->a, 0 if it does not use it."""
    for k in range(3):
        u, v = int(face[k]), int(face[(k + 1) % 3])
        if u == a and v == b:
            return 1
        if u == b and v == a:
            return -1
    return 0


def _pair_faces_around_edge(vertices: np.ndarray,
                            faces: np.ndarray,
                            a: int,
                            b: int,
                            flist: List[int]) -> Set[Tuple[int, int]]:
    """Pair the faces meeting on non-manifold edge (a, b) into manifold sheets.

    The faces incident to an over-used edge are sorted by their radial angle
    about the edge axis, then greedily paired with the angularly-nearest face
    that traverses the edge in the opposite direction (a consistently oriented
    manifold edge has exactly one face each way).  Each resulting pair is one
    surface sheet passing through the edge.

    Without this, a non-manifold edge is treated as severing every face that
    touches it.  On a surface whose edges are *mostly* non-manifold -- two
    sheets sharing one vertex/edge skeleton -- that shatters the surface into
    individual triangles instead of separating it into two coherent sheets.
    """
    axis = vertices[b] - vertices[a]
    axis_len = float(np.linalg.norm(axis))
    if axis_len < 1e-30:
        return set()
    axis = axis / axis_len

    # Orthonormal frame perpendicular to the edge.
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(axis[0])) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u_len = float(np.linalg.norm(u))
    if u_len < 1e-30:
        return set()
    u = u / u_len
    w = np.cross(axis, u)

    entries: List[Tuple[float, int, int]] = []
    for fi in flist:
        face = faces[fi]
        direction = _edge_traversal_dir(face, a, b)
        if direction == 0:
            continue
        third = [int(x) for x in face if int(x) != a and int(x) != b]
        if not third:
            continue
        vec = vertices[third[0]] - vertices[a]
        vec = vec - float(np.dot(vec, axis)) * axis
        vec_len = float(np.linalg.norm(vec))
        if vec_len < 1e-30:
            continue                      # third vertex lies on the edge line
        vec = vec / vec_len
        angle = math.atan2(float(np.dot(vec, w)), float(np.dot(vec, u)))
        entries.append((angle, fi, direction))

    if len(entries) < 2:
        return set()

    areas={fi:float(np.linalg.norm(np.cross(vertices[faces[fi,1]]-vertices[faces[fi,0]],
                                          vertices[faces[fi,2]]-vertices[faces[fi,0]]))) for _,fi,_ in entries}
    entries.sort(key=lambda e: (e[0],-areas[e[1]],e[1]))
    n = len(entries)
    used: Set[int] = set()
    pairs: Set[Tuple[int, int]] = set()

    for i in range(n):
        _, fi, di = entries[i]
        if fi in used:
            continue
        candidates=sorted((entry for entry in entries if entry[1] not in used and entry[2]==-di),
                          key=lambda entry: (abs(math.atan2(math.sin(entry[0]-entries[i][0]),math.cos(entry[0]-entries[i][0]))),-areas[entry[1]],entry[1]))
        for _, fj, dj in candidates:
            if fj in used or fj == fi:
                continue
            if dj == -di:
                pairs.add((fi, fj) if fi < fj else (fj, fi))
                used.add(fi)
                used.add(fj)
                break

    return pairs


def repair_topology_separation(vertices, faces, log, config=None):
    """Split coherent incident fans by identity, with zero coordinate movement."""
    if not len(faces):
        return vertices, faces
    from src.core.mesh_analyzer import build_edge_map
    incident=defaultdict(list)
    for fi,face in enumerate(faces):
        for vi in face:
            incident[int(vi)].append(fi)
    links=defaultdict(list)
    split_edges=0
    for (a,b),flist in build_edge_map(faces).items():
        if config is not None:
            config.check_deadline()
        if len(flist)==2:
            pairs=[tuple(flist)]
        elif len(flist)>2:
            pairs=_pair_faces_around_edge(vertices,faces,a,b,flist)
            split_edges+=1
        else:
            continue
        for pair in pairs:
            links[a].append(pair); links[b].append(pair)
    result=faces.copy()
    added=[]
    for vi,fan in incident.items():
        if config is not None:
            config.check_deadline()
        adjacency=defaultdict(list)
        for a,b in links[vi]:
            adjacency[a].append(b); adjacency[b].append(a)
        seen=set(); fans=[]
        for start in fan:
            if start in seen:
                continue
            component=[]; todo=[start]
            while todo:
                fi=todo.pop()
                if fi in seen:
                    continue
                seen.add(fi); component.append(fi)
                todo.extend(adjacency[fi])
            fans.append(component)
        for component in fans[1:]:
            new_index=len(vertices)+len(added)
            added.append(vertices[vi].copy())
            for fi in component:
                result[fi,result[fi]==vi]=new_index
    log.nm_vertices_cracked+=len(added)
    log.nm_edges_cracked+=split_edges
    return (np.vstack([vertices,np.array(added)]) if added else vertices.copy()),result
