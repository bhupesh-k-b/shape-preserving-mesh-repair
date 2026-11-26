# Runtime architecture evidence

The [interactive Archify diagram](runtime-architecture.html) contains 11 components, one emphasized STL repair path, three detail cards, and explicit input, network, application, output, and third-party code boundaries.

## Runtime and primary path

The application executes locally as Python code. `main.py` coordinates both repair stages in the same process. The diagram's stage-to-stage arrow represents a file handoff coordinated by the CLI: initial cleanup writes a temporary STL and the advanced pipeline reads it. The CLI removes that temporary file after advanced repair returns.

| Component | Verified working-tree evidence |
| --- | --- |
| Input STL | `main.py:19–25`; `src/core/initial_cleanup_engine.py:42–124` |
| Repair CLI | `main.py:14–61` |
| Initial Cleanup | `src/pipelines/mesh_repair_initial.py:20–47`; `src/core/initial_cleanup_engine.py:764–816` |
| Advanced Repair | `src/pipelines/mesh_repair_pipeline.py:86–480` |
| Output STL + Logs | `src/pipelines/mesh_repair_pipeline.py:26–40,456–480`; `src/core/repair_config.py:106–145` |
| Thingi10K | `src/scripts/evaluate_thingi10k_hf.py:18–38`; `src/scripts/evaluate_thingi10k.py:17–22` |
| Batch Evaluators | `src/scripts/evaluate_dataset.py:31–118`; `src/scripts/evaluate_thingi10k_hf.py:42–126` |
| RepairConfig | `main.py:43–51`; `src/core/repair_config.py:9–103`; `repair_config.ini` |
| Mesh Analyzer | `src/core/mesh_analyzer.py:94–137,375–426`; advanced pipeline load and post-export analysis |
| Repair Routines | Advanced pipeline imports at lines 15–23 and calls at lines 172–463; implementations in `src/routines/` |
| Geometry Libraries | `requirements.txt`; analyzer imports `numpy-stl`; welding imports `scipy.spatial.cKDTree`; holes use optional `Delaunay`; shell union attempts `trimesh.boolean` |

Advanced repair operates on NumPy vertex, face and normal arrays. It filters and unions shells, removes degenerates and duplicates, excises non-manifold faces, separates topology, corrects winding, welds vertices, resolves non-orientable strips, and patches planar/non-planar holes. Its convergence loop defaults to three passes and rolls back a pass if structural defects increase. The final path audits winding, stabilizes split vertices for float32 export, writes binary STL, and reloads it for analysis when verbose output is enabled.

## Boundaries and external dependencies

Input STL bytes enter parsers with the permissions of the Python process. The application boundary groups repository-owned repair code; it is a code ownership and trust distinction, not an operating-system isolation mechanism. Third-party numerical and geometry libraries also execute with the process's privileges. NumPy is central to the pipeline, `numpy-stl` loads the advanced mesh representation, and SciPy supplies spatial/triangulation paths. The optional trimesh Boolean union can require a compatible manifold or Blender backend and falls back to other union strategies on failure. One dependency edge represents these shared imports; the diagram deliberately avoids repeating a library edge from every consumer.

INI configuration and input/output paths are operator controlled. Temporary STL and final STL files cross the filesystem boundary; local OS permissions govern access. Console reports are emitted to stdout. Network access occurs in optional dataset evaluation, where Hugging Face supplies metadata and geometry over HTTPS or through the `datasets` streaming client.

The optional batch evaluator is drawn outside the repair process. `evaluate_dataset.py` launches `main.py` in a subprocess with a default 120-second hard timeout, independently verifies the exported file, and writes a CSV report. The direct HTTPS/NPZ evaluator also uses a subprocess, but supplies no subprocess timeout. `evaluate_thingi10k.py` instead calls `MeshRepairPipeline` directly and bypasses initial cleanup; this alternate path is explained in a card rather than drawn as another path. Evaluator CSV output and output revalidation are similarly summarized in cards.

## Operational observations

The advanced pipeline's 300-second timer is cooperative, checked between operations, and may return early before writing an output file. The CLI's completion message does not enforce a zero-defect postcondition. In the local dataset evaluator, `Status = Success` records a zero subprocess return code; `Watertight` is a separate field and can be false. The diagram therefore labels artifacts as output rather than guaranteed repaired geometry.

## Evidence scope and artifact receipts

This analysis describes the current working tree, including modified and untracked files. Its base HEAD is `6738cb7f9da3286ea792d1e4ac0d8a9c93872cb7`, but the diagram does not claim that HEAD contains the analyzed runtime. Working-tree file hashes are recorded in `runtime-architecture-evidence.json`. Revision-pinned source hyperlinks are omitted because they would point to different code.

The editable source is `runtime-architecture.archify.json`. Deterministic delivery and browser evidence are separate checks; their receipts and perceptual review status are recorded in `runtime-architecture-receipt.json`. The diagram opens with Archify's default static viewer and supports search, focus, themes and exports.
