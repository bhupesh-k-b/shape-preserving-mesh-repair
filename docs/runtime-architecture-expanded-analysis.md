# Expanded runtime architecture

[Open the expanded diagram](runtime-architecture-expanded.html).

The diagram expands the repair stages into 12 components while retaining one primary path: input STL → `main.py` → `InitialCleanupPipeline` → `MeshRepairPipeline` → output STL and logs. Edges below that path represent internal execution within their parent stage. The grouped repair components are logical views of functions, rather than independent services. Stage 1 hands off a temporary binary STL to stage 2 through the CLI.

| Parent | Components shown beneath it | Verified source |
| --- | --- | --- |
| Initial Cleanup | Edge Cleanup; Patch + Normals | `src/core/initial_cleanup_engine.py:764–816`; `src/pipelines/mesh_repair_initial.py:20–47` |
| Advanced Repair | Mesh Analyzer; Repair Routines | `src/pipelines/mesh_repair_pipeline.py:130–480`; `src/core/mesh_analyzer.py:375–426` |
| Repair Routines | Topology + Winding; Hole Patching | Advanced pipeline imports and calls; `src/routines/repair_shell_union.py`, `repair_degenerates.py`, `repair_nonmanifold.py`, `repair_topology_separation.py`, `repair_winding.py`, `repair_mobius.py`, `repair_welding.py`, `repair_holes.py` |

Edge Cleanup groups exact hash matching, nearby edge matching and vertex snapping. Patch + Normals groups disconnected-facet removal, hole filling, normal direction/value correction and signed-volume correction. `main.py` requests three initial iterations; the cleanup engine may exit its nearby matching loop earlier.

Topology + Winding groups shell union/filtering, degenerate and duplicate cleanup, non-manifold excision and vertex separation, orientation audits, Möbius cracking and welding. Hole Patching groups planar and non-planar patch routines. Their detailed call ordering, repeat passes and rollback remain in the Advanced Repair pipeline; the two branches are decomposition, not a claim of parallel execution. The supporting card describes convergence, final winding audit, float32 stabilization and export/reload analysis.

Input, application code, output filesystem/console and third-party code trust boundaries are drawn explicitly. Numerical and geometry dependencies are represented once beside Repair Routines; they also support loading and cleanup. All imported libraries share process privileges. Optional trimesh Boolean backends, operator configuration, HTTPS Thingi10K ingestion and evaluator subprocess behavior are summarized in cards to keep the expanded internal architecture readable.

The analysis describes the current working tree. Full runtime observations and alternate evaluation entrypoints remain in [the repository analysis](runtime-architecture-analysis.md). The editable Archify source is `runtime-architecture-expanded.archify.json`; deterministic delivery and browser/perceptual evidence are recorded in `runtime-architecture-expanded-receipt.json` and `runtime-architecture-expanded.visual-check.json`.
