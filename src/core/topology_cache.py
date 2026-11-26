import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

class TopologyCache:
    """
    Provides lazy evaluation and memoization of mesh topology structures.
    Invalidate this cache whenever the mesh 'faces' array is modified.
    """
    def __init__(self, faces: np.ndarray):
        self._faces = faces
        self._edge_map = None
        self._directed_edge_map = None
        self._vertex_to_faces = None
        self._face_adjacency = None
        self._naked_edges = None
        self._nonmanifold_edges = None

    def invalidate(self):
        """Clear all cached structures."""
        self._edge_map = None
        self._directed_edge_map = None
        self._vertex_to_faces = None
        self._face_adjacency = None
        self._naked_edges = None
        self._nonmanifold_edges = None

    @property
    def edge_map(self) -> Dict[Tuple[int, int], List[int]]:
        """Undirected edge map: (v_min, v_max) -> [face_indices]."""
        if self._edge_map is None:
            from src.core.mesh_analyzer import build_edge_map
            self._edge_map = build_edge_map(self._faces)
        return self._edge_map

    @property
    def directed_edge_map(self) -> Dict[Tuple[int, int], List[int]]:
        """Directed edge map: (v0, v1) -> [face_indices]."""
        if self._directed_edge_map is None:
            d_map = defaultdict(list)
            for fi, face in enumerate(self._faces):
                for i in range(3):
                    edge = (int(face[i]), int(face[(i + 1) % 3]))
                    d_map[edge].append(fi)
            self._directed_edge_map = dict(d_map)
        return self._directed_edge_map

    @property
    def vertex_to_faces(self) -> List[Set[int]]:
        """Adjacency: vertex_index -> set(face_indices)."""
        if self._vertex_to_faces is None:
            n_verts = int(np.max(self._faces) + 1) if self._faces.size > 0 else 0
            adj = [set() for _ in range(n_verts)]
            for fi, face in enumerate(self._faces):
                for vi in face:
                    adj[vi].add(fi)
            self._vertex_to_faces = adj
        return self._vertex_to_faces

    @property
    def defects(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Returns (naked_edges, non_manifold_edges)."""
        if self._naked_edges is None or self._nonmanifold_edges is None:
            from src.core.mesh_analyzer import check_naked_and_nonmanifold
            self._naked_edges, self._nonmanifold_edges = check_naked_and_nonmanifold(self.edge_map)
        return self._naked_edges, self._nonmanifold_edges

    @property
    def naked_edges(self) -> List[Tuple[int, int]]:
        return self.defects[0]

    @property
    def nonmanifold_edges(self) -> List[Tuple[int, int]]:
        return self.defects[1]

    @property
    def face_adjacency(self) -> List[Set[int]]:
        """Adjacency: face_index -> set(neighbor_face_indices)."""
        if self._face_adjacency is None:
            n_faces = len(self._faces)
            adj = [set() for _ in range(n_faces)]
            edge_map = self.edge_map
            for edge, faces in edge_map.items():
                if len(faces) > 1:
                    for i in range(len(faces)):
                        for j in range(i + 1, len(faces)):
                            fi, fj = faces[i], faces[j]
                            adj[fi].add(fj)
                            adj[fj].add(fi)
            self._face_adjacency = adj
        return self._face_adjacency
