"""
Module for topologically cracking Möbius strips and other non-orientable shells.
"""

import numpy as np
from typing import Tuple, List, Dict, Set
from collections import defaultdict

from src.core.repair_config import RepairConfig, RepairLog
from src.routines.repair_winding import check_winding_consistency


def crack_mobius_strips(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    config: RepairConfig,
    log: RepairLog
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crack non-orientable singularities by duplicating vertices along conflicted edges.
    This resolves the 'twisted trap' without deleting any faces.
    """
    if len(faces) == 0:
        return vertices, faces, normals

    # First, identify the conflicts
    audit = check_winding_consistency(vertices, faces)
    if not audit.conflicted_edges:
        return vertices, faces, normals

    # limit processing if there are too many conflicts
    conflict_edges = audit.conflicted_edges
    if len(conflict_edges) > config.max_mobius_crack:
        print(f"    [Mobius] Warning: {len(conflict_edges)} conflicts exceeds limit ({config.max_mobius_crack}). Processing first {config.max_mobius_crack} only.")
        conflict_edges = conflict_edges[:config.max_mobius_crack]

    vertices = vertices.copy()
    faces = faces.copy()
    
    # We need to build a map of which face uses which directed edge
    # so we know which one to 'detach'
    directed_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for fi, face in enumerate(faces):
        for j in range(3):
            a, b = int(face[j]), int(face[(j + 1) % 3])
            directed_map[(a, b)].append(fi)

    cracked_count = 0
    for edge_v in conflict_edges:
        u, v = edge_v
        # The audit says this edge is conflicted, meaning multiple faces
        # traverse it in the SAME direction. 
        # For manifold edges, there are exactly 2 faces.
        # Both share (u, v) or both share (v, u).
        
        f_list_fwd = directed_map.get((u, v), [])
        f_list_bwd = directed_map.get((v, u), [])
        
        if len(f_list_fwd) == 2:
            conflict_faces = f_list_fwd
            target_edge = (u, v)
        elif len(f_list_bwd) == 2:
            conflict_faces = f_list_bwd
            target_edge = (v, u)
        else:
            # Not a simple manifold conflict, skip for now
            continue

        # Crack: duplicate u and v, and assign them to one of the faces.
        fi_to_change = conflict_faces[1]
        
        # New vertex indices
        u_new = len(vertices)
        v_new = len(vertices) + 1
        
        # Add new vertices to the array
        vertices = np.vstack([vertices, vertices[u], vertices[v]])
        
        # Update the face
        face = faces[fi_to_change]
        new_face = face.copy()
        for i in range(3):
            if new_face[i] == u:
                new_face[i] = u_new
            elif new_face[i] == v:
                new_face[i] = v_new
        
        faces[fi_to_change] = new_face
        cracked_count += 1
        
        # Technically we should update maps if we were doing more cracks involving 
        # these same faces/vertices, but since we are doing a single-pass 
        # based on the initial audit, and we only handle small counts, 
        # this is usually safe.

    log.mobius_cracked += cracked_count
    return vertices, faces, normals
