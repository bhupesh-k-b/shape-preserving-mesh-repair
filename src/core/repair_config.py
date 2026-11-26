"""
Configuration and logging definition for the Mesh Repair Pipeline.
"""

from typing import Optional
from dataclasses import dataclass, field, replace
import time
import numpy as np
@dataclass
class RepairConfig:
    area_tolerance:       float = 1e-12   # below this → degenerate face
    merge_tolerance:      float = 1e-6    # vertex merge distance
    planar_tolerance:     float = 1e-4    # SVD flatness ratio
    normal_angle_tol_deg: float = 90.0    # stored-vs-computed flip threshold
    # Shell filtering/union
    shell_filter_enabled: bool  = False
    shell_area_threshold: float = 0.002   # 0.2% of total area; opt-in open noise removal
    # Non-manifold: how many faces to keep per edge (usually 2)
    max_faces_per_edge:   int   = 2
    # Boundary sliver welding: merge naked-edge vertices closer than this
    boundary_weld_tolerance: float = 0.001
    # Möbius cracking: max number of edges to split to resolve non-orientability
    max_mobius_crack:     int   = 1000

    # Union
    union_hollow:         bool  = False
    union_force_tier:     Optional[int] = None
    union_sample_ratio:   float = 0.2
    
    # Adaptive tolerances
    adaptive:             bool  = False
    # Intentional changes to existing geometry are disabled by default.
    max_surface_deviation: float = 0.0
    union_enabled:        bool = False
    preserve_cavities:    bool = True
    backend_enabled:     bool = True
    timeout_seconds:     float = 120.0
    max_hole_edges:      int = 2048
    _deadline: Optional[float] = field(default=None, repr=False, compare=False)
    _material_roles: dict = field(default_factory=dict, repr=False, compare=False)
    _ambiguous_material_roles: dict = field(default_factory=dict, repr=False, compare=False)
    _roles_initialized: bool = field(default=False, repr=False, compare=False)

    def check_deadline(self):
        if self._deadline is not None and time.perf_counter() >= self._deadline:
            raise TimeoutError("Mesh repair deadline exceeded")

    def for_mesh(self, vertices, faces):
        """Derive a per-call policy without mutating reusable configuration."""
        cfg = replace(self)
        cfg._material_roles={}
        cfg._ambiguous_material_roles={}
        cfg._roles_initialized=False
        if not np.isfinite(cfg.max_surface_deviation) or not np.isfinite(cfg.timeout_seconds) or cfg.max_surface_deviation < 0 or cfg.timeout_seconds <= 0:
            raise ValueError("Deviation must be nonnegative and timeout must be positive")
        if cfg.max_hole_edges<3:
            raise ValueError('max_hole_edges must be at least 3')
        if cfg.adaptive:
            cfg.compute_adaptive_tolerances(vertices, faces)
        if cfg.max_surface_deviation == 0:
            cfg.area_tolerance = 0.0  # only exact degenerates may be discarded
            cfg.merge_tolerance=0.0
            cfg.boundary_weld_tolerance=0.0
        else:
            cfg.merge_tolerance = min(cfg.merge_tolerance, cfg.max_surface_deviation)
            cfg.boundary_weld_tolerance = min(cfg.boundary_weld_tolerance, cfg.max_surface_deviation)
        return cfg
    
    def compute_adaptive_tolerances(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        """
        Derive local tolerances, clamped to the declared surface-distance budget.
        Strict preservation disables coordinate welding entirely.
        """
        import numpy as np
        from src.core.repair_utils import compute_edge_lengths
        
        if len(faces) == 0:
            return
            
        edges = compute_edge_lengths(vertices, faces)
        if len(edges) == 0:
            return
            
        edges = edges[edges > 0]
        if len(edges) == 0:
            return
        p1 = float(np.percentile(edges, 1))
        p5 = float(np.percentile(edges, 5))
        
        diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        cap = min(1e-4 * diagonal, self.max_surface_deviation)
        self.merge_tolerance = min(0.01 * p1, cap)
        self.boundary_weld_tolerance = min(0.01 * p5, cap)
        # area_tolerance is area, so we square the linear tolerance
        self.area_tolerance = min(self.area_tolerance, (0.001 * p1) ** 2)
        
        # We also set planar_tolerance based on p5 if it's currently at default
        # (Though planar_tolerance is a ratio, a safe bet is to scale it by model scale if needed, 
        # but here we'll stick to the user's p1/p5 logic for merge/weld/degen).
    
    @classmethod
    def from_ini(cls, path: str) -> "RepairConfig":
        import configparser
        import os
        
        cfg = cls()
        if not os.path.exists(path):
            return cfg
            
        parser = configparser.ConfigParser()
        parser.read(path)
        
        if parser.has_section('Tolerances'):
            cfg.area_tolerance = parser.getfloat('Tolerances', 'area_tolerance', fallback=cfg.area_tolerance)
            cfg.merge_tolerance = parser.getfloat('Tolerances', 'merge_tolerance', fallback=cfg.merge_tolerance)
            cfg.planar_tolerance = parser.getfloat('Tolerances', 'planar_tolerance', fallback=cfg.planar_tolerance)
            cfg.normal_angle_tol_deg = parser.getfloat('Tolerances', 'normal_angle_tol_deg', fallback=cfg.normal_angle_tol_deg)
            cfg.adaptive = parser.getboolean('Tolerances', 'adaptive', fallback=cfg.adaptive)
            
        if parser.has_section('Shells'):
            cfg.shell_filter_enabled = parser.getboolean('Shells', 'filter_shells', fallback=cfg.shell_filter_enabled)
            cfg.shell_area_threshold = parser.getfloat('Shells', 'shell_area_threshold', fallback=cfg.shell_area_threshold)
            
        if parser.has_section('NonManifold'):
            cfg.max_faces_per_edge = parser.getint('NonManifold', 'max_faces_per_edge', fallback=cfg.max_faces_per_edge)
            
        if parser.has_section('Welding'):
            cfg.boundary_weld_tolerance = parser.getfloat('Welding', 'boundary_weld_tolerance', fallback=cfg.boundary_weld_tolerance)
            
        if parser.has_section('Mobius'):
            cfg.max_mobius_crack = parser.getint('Mobius', 'max_mobius_crack', fallback=cfg.max_mobius_crack)
            

            
        if parser.has_section('Union'):
            cfg.union_enabled = parser.getboolean('Union', 'enabled', fallback=cfg.union_enabled)
            cfg.union_hollow = parser.getboolean('Union', 'hollow', fallback=cfg.union_hollow)
            if parser.has_option('Union', 'force_tier'):
                val = parser.get('Union', 'force_tier').strip()
                cfg.union_force_tier = int(val) if val else None
            cfg.union_sample_ratio = parser.getfloat('Union', 'sample_ratio', fallback=cfg.union_sample_ratio)

        if parser.has_section('Preservation'):
            cfg.max_surface_deviation = parser.getfloat('Preservation', 'max_surface_deviation', fallback=cfg.max_surface_deviation)
            cfg.preserve_cavities = parser.getboolean('Preservation', 'preserve_cavities', fallback=cfg.preserve_cavities)
        if parser.has_section('Execution'):
            cfg.timeout_seconds = parser.getfloat('Execution', 'timeout_seconds', fallback=cfg.timeout_seconds)
            cfg.backend_enabled = parser.getboolean('Execution', 'backend_enabled', fallback=cfg.backend_enabled)
            
        return cfg


@dataclass
class RepairLog:
    status: str = "NEEDS_REVIEW"
    strategy: str = "none"
    verification_basis: str = "original_surface"
    reasons: list = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    attempts: list = field(default_factory=list)
    reconstructed_area: float = 0.0
    source_surface_error: float = 0.0
    distance_error_bound: float = 0.0
    provenance: dict = field(default_factory=dict)
    degenerate_removed:   int = 0
    duplicate_removed:    int = 0
    shells_filtered:      int = 0
    non_manifold_removed: int = 0
    faces_reoriented:     int = 0
    vertices_welded:      int = 0
    planar_holes_filled:  int = 0
    non_planar_filled:    int = 0
    mobius_cracked:       int = 0
    nm_vertices_cracked:  int = 0
    nm_edges_cracked:     int = 0
    faces_added:          int = 0
    repair_time_ms:       float = 0.0

    @property
    def success(self):
        return self.status in {"VERIFIED_UNCHANGED", "VERIFIED_REPAIRED"}

    def summary(self) -> str:
        lines = [
            "━" * 60,
            "  REPAIR LOG",
            f"  Status / strategy          : {self.status} / {self.strategy}",
            f"  Preservation error bound   : {self.source_surface_error:.6g}",
            f"  New/retriangulated area    : {self.reconstructed_area:.6g}",
            f"  Reasons                    : {'; '.join(self.reasons) or 'none'}",
            "━" * 60,
            f"  Degenerate faces removed   : {self.degenerate_removed}",
            f"  Duplicate faces removed    : {self.duplicate_removed}",
            f"  Insignificant shells culled: {self.shells_filtered}",
            f"  Non-manifold faces removed : {self.non_manifold_removed}",
            f"  Faces reoriented (normals) : {self.faces_reoriented}",
            f"  Boundary vertices welded   : {self.vertices_welded}",
            f"  Planar holes filled        : {self.planar_holes_filled}",
            f"  Non-planar holes filled    : {self.non_planar_filled}",
            f"  Möbius strips cracked      : {self.mobius_cracked}",
            f"  NM-Vertices cracked        : {self.nm_vertices_cracked}",
            f"  NM-Edges cracked           : {self.nm_edges_cracked}",
            f"  New faces added (patches)  : {self.faces_added}",
            f"  Repair time                : {self.repair_time_ms:.1f} ms",
            "━" * 60,
        ]
        return "\n".join(lines)
