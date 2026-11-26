# Advanced Mesh Repairing — audit, 10 October 2026

The current pipeline has reproducible geometry-loss, repair, validation, and file-handling defects. I would not treat its present output as production-ready without independent validation. The strongest evidence is a CLI run that deletes its requested output while reporting success, complete removal of `Plane_lower.stl`, and five failing geometry tests in the project's own suite.

Scope: `/Users/bhupesh/Advanced_Mesh_Repairing`, including the existing uncommitted changes. Git HEAD was `6738cb7`; this is an audit of the working files, not just that commit. `Advanced_Mesh_Repairing_refined` and `Version2` are excluded from the conclusions. Repair source files and original STL assets were not changed. All reproduction STL files were created in temporary directories.

Validation used Python 3.14, NumPy 2.4.4, SciPy 1.17.1, and trimesh 4.12.2. A manifold Boolean backend is available in this environment. Explicit Tier 2 checks exercise the supported fallback independently of that backend.

Evidence and reproduction:

- [Reproduction script](../scratch/audit_pipeline_20261010.py): 16 recorded cases; assertions reproduce the current faults. These assertions document existing behavior and will need updating when the faults are fixed.
- [Controlled-case results](pipeline-audit-2026-10-10-results.json).
- [Four additional CLI sample results](pipeline-audit-2026-10-10-samples.json).
- `python3 src/test_mesh_repair_pipeline.py -v`: **23 passed, 7 failed**. Five failures concern non-manifold repair or hole filling; two reference a missing smoothing module.
- `python3 -m pytest src/test_mesh_repair_pipeline.py -q`: **one collection/setup error**, because pytest collects the decorator named `test` and cannot find its `fn` fixture.

P1 denotes a defect that should be fixed before trusting or releasing the pipeline. P2 denotes a reliability or engineering issue that should follow. The findings are ordered approximately by remediation priority, with file loss first.

1. **P1 — uppercase output extensions cause deletion of the final output, and in-place use can delete the input.**

   Source: [main.py:31](../main.py:31), [main.py:52](../main.py:52).

   The temporary path is constructed with `output_path.replace(".stl", "_initial_cleanup_temp.stl")`. For a path ending in `.STL`, the temporary path equals the final path. Both stages run, then cleanup removes that file. The CLI exits **0** and prints `[SUCCESS]`, although the output no longer exists. The same bug applies to extensionless output paths when their full path contains no lowercase `.stl`.

   Two runs reproduced this: a separate `uppercase-output.STL` vanished, and a temporary input repaired in place as `in-place.STL` was deleted. No user asset was involved.

   Fix: allocate a unique temporary file independently of the requested extension, protect input/output aliases, and publish the completed result with an atomic replacement after verification. Cleanup should remove only the independently allocated temporary path.

2. **P1 — shell detection and filtering can erase a complete model before deduplication.**

   Source: [repair_shell_union.py:98](../src/routines/repair_shell_union.py:98), [repair_shell_union.py:185](../src/routines/repair_shell_union.py:185), [mesh_repair_pipeline.py:172](../src/pipelines/mesh_repair_pipeline.py:172).

   `_find_shells` connects faces only across edges incident to exactly two faces. Duplicate or non-manifold faces therefore fracture a genuinely connected surface into artificial components. `filter_small_shells` unconditionally deletes every component with at most two faces, even when `shell_filter_enabled=False`. Stage 0 runs this filter before duplicate removal.

   A duplicated cube becomes **24 one-face shells and zero output faces**, with filtering disabled. `Plane_lower.stl` becomes **1,484 one-face shells and zero output faces**. Running its complete CLI pipeline writes an **84-byte, zero-triangle STL**, then crashes during verbose post-analysis with an empty-array reduction. That run exits 1; it does not count as a successful CLI repair.

   Fix: remove duplicates/degenerates before destructive filtering, distinguish connected components from manifold sheets, honor the disabling option, and refuse to replace a nonempty model with an empty result. Make removal of valid small closed parts an explicit policy rather than a watertightness optimization.

3. **P1 — initial cleanup welds at legitimate feature scale and changes geometry substantially.**

   Source: [initial_cleanup_engine.py:776](../src/core/initial_cleanup_engine.py:776).

   Nearby matching starts at `stl.stats.shortest_edge` and increases that distance on subsequent passes. A shortest edge can represent a real feature rather than a crack. This stage runs before the advanced configuration and before `--adaptive` has any effect.

   Initial cleanup of `Chair.stl` changes the vertex geometry by a maximum bidirectional nearest-vertex distance of **2.0 units**, **7.94%** of the original 25.19-unit bounding diagonal. Face count stays at 66 and naked edges fall from 6 to 0. This is a concrete example of an improved edge metric accompanied by geometry distortion. The distance is a vertex metric, not a surface Hausdorff measurement.

   Fix: bound welding by both model scale and local feature size, limit accumulated displacement, and preserve the original mesh when a repair exceeds the allowed deviation. Apply these limits in initial cleanup as well as later welding.

4. **P1 — non-manifold excision calculates faces to remove but never removes them.**

   Source: [repair_nonmanifold.py:47](../src/routines/repair_nonmanifold.py:47), [repair_nonmanifold.py:61](../src/routines/repair_nonmanifold.py:61).

   `faces_to_remove` is populated by the ranking pass and never applied to `faces` or `normals`. Only the later leaf-pruning pass can change the arrays. In the three-triangle shared-edge fixture, the routine leaves the non-manifold edge and all three faces untouched, while reporting **three faces removed**. The latter happens because pruning increments the log before `if not keep: break` prevents the removal.

   This is also the cause of `UNIT_06_non_manifold_edge_excision` failing. The ranking includes a face's own normal among its neighbors, which should be excluded when the excision pass is repaired.

   Fix: apply the removal mask consistently, update logs only for committed changes, and attempt topology-preserving separation before destructive excision. Retain a nonempty-result guard.

5. **P1 — the hole distortion guard rejects ordinary holes on coarse meshes.**

   Source: [repair_holes.py:1384](../src/routines/repair_holes.py:1384), [repair_holes.py:1394](../src/routines/repair_holes.py:1394).

   Every edge of every candidate patch must be shorter than `0.35 * model_diagonal`. This includes edges already present on the hole boundary. A unit cube missing one cap has one-unit boundary edges, but the threshold is **0.6062**. A valid cap therefore fails regardless of triangulation, and the fallback cannot rescue it.

   The routine adds **zero faces**, leaving all **four naked edges**. The public advanced repair API also returns normally with that hole still present. Four suite failures concern planar fill or integrations relying on it. Separately, the condition `post_nm_count > 0` rejects a locally safe fill if any unrelated non-manifold edge remains; the comment describes preventing new non-manifold edges, but the implementation tests an absolute count.

   Fix: evaluate newly introduced geometry against the local hole and neighboring surface, account for existing boundary edges, and compare pre/post non-manifold defects. Report why a patch is rejected rather than silently continuing.

6. **P1 — topology separation can turn coherent sheets into fragments subsequently culled as noise.**

   Source: [repair_topology_separation.py:53](../src/routines/repair_topology_separation.py:53), [repair_topology_separation.py:109](../src/routines/repair_topology_separation.py:109).

   Overused edges disconnect every incident face in the local adjacency graph. There is no pairing of faces belonging to the same sheet. Duplicated vertices receive a fixed `[1.1e-4, 1.2e-4, 1.3e-4] * rank` displacement, regardless of scale; this is also much larger than the default global merge tolerance.

   A cube represented by two coincident sheets with different face diagonals has 24 faces and 12 non-manifold edges. Separation removes the non-manifold edge count but creates **48 naked edges**, cracks **40 vertices**, and produces **12 two-face components**. The subsequent mandatory culling policy would delete every such component. This fixture has no duplicate vertex-set triangles for the duplicate remover to eliminate.

   Fix: pair compatible faces around non-manifold edges into coherent sheets, preserve triangle geometry during index separation, and validate resulting sheet continuity before any culling. Handle STL's inability to store distinct coincident vertex identities explicitly at export.

7. **P1 — winding repair destroys the cavity orientation requested by hollow mode.**

   Source: [repair_shell_union.py:595](../src/routines/repair_shell_union.py:595), [repair_winding.py:522](../src/routines/repair_winding.py:522), [mesh_repair_pipeline.py:226](../src/pipelines/mesh_repair_pipeline.py:226).

   Tier 2 hollow union correctly orients an inner void inward, but subsequent winding repair makes every closed shell's signed volume positive. The two stages have conflicting definitions of correct orientation.

   Nested cubes reproduce shell volumes **`[27, -1]` after hollow union**, then **`[27, +1]` after winding repair**. The inner shell ceases to encode a void. Tier 1 also calls ordinary Boolean union without receiving the hollow option, so it cannot honor that option's cavity semantics.

   Fix: propagate containment depth and intended material/void roles through orientation and final verification. Hollow Boolean operations must retain or subtract cavities as requested.

8. **P1 — Tier 2 reports successful union without resolving partially overlapping solids.**

   Source: [repair_shell_union.py:366](../src/routines/repair_shell_union.py:366), [repair_shell_union.py:575](../src/routines/repair_shell_union.py:575), [repair_shell_union.py:786](../src/routines/repair_shell_union.py:786).

   Classification assigns parents only when a parent bounding box fully contains the child. Tier 2 only examines shells with parents, then attempts to detect overlap using a condition that requires the parent not to contain the child. Ordinary partial overlaps never enter this branch. A nonempty result is enough to set `ulog.success=True`.

   Two unit cubes offset by `[0.4, 0.3, 0.2]` retain all **24 faces**, cull **zero faces**, and report successful Tier 2 union. Both edge defect counts are zero, but the closed shells still intersect. Their combined signed volume is **2.0**, versus a geometric union volume of **1.664**.

   Fix: classify partial overlaps independently of containment and use an actual triangle-splitting Boolean operation. If that operation is unavailable or fails, return a clearly incomplete result instead of declaring union success. A naive concatenation should likewise identify unresolved overlaps.

9. **P1 — success reporting is disconnected from output validity and fidelity.**

   Source: [mesh_repair_pipeline.py:469](../src/pipelines/mesh_repair_pipeline.py:469), [main.py:57](../main.py:57), [evaluate_dataset.py:81](../src/scripts/evaluate_dataset.py:81), [mesh_analyzer.py:44](../src/core/mesh_analyzer.py:44).

   Post-export analysis runs only under `verbose=True` and its result does not gate the return. The API has no explicit succeeded/partial/failed status. `main.py` prints success whenever calls return. The evaluator sets `Status = "Success"` for exit code 0 even when its independently computed `Watertight` value is false. `AnalysisResult.is_watertight` checks only naked edges, so the doubled cube reports `True` despite **18 non-manifold edges**.

   Edge incidence alone also accepts the intersecting-box fixture and cannot detect erased modeled parts, displaced vertices, incorrect cavity orientation, or winding inconsistency. Because export recomputes stored normals, comparing stored normals to triangle normals does not independently validate material orientation.

   Fix: return a structured result, always verify the serialized/reloaded output, require a nonempty finite mesh with valid faces and appropriate topology/orientation, and make CLI/evaluation status follow that result. Add component retention and geometric deviation measurements with thresholds appropriate to the repair mode. Check self-intersections and intended cavities when claiming printable solid output.

10. **P2 — export stabilization can create coordinate collisions and substantial displacement.**

    Source: [mesh_repair_pipeline.py:55](../src/pipelines/mesh_repair_pipeline.py:55), [mesh_repair_pipeline.py:74](../src/pipelines/mesh_repair_pipeline.py:74), [mesh_repair_pipeline.py:461](../src/pipelines/mesh_repair_pipeline.py:461).

    Duplicate float32 positions are moved using offsets derived from the maximum absolute coordinate, but proposed positions are not checked against other vertices. Input positions `[(0,0,0), (0,0,0), (1e-6,0,0)]` become `[(0,0,0), (1e-6,0,0), (1e-6,0,0)]`: the duplicate is moved directly onto an existing vertex. At coordinates near `1e6`, the displacement for a duplicate is **0.4768 units**. It depends on distance from the origin, not just mesh size.

    This mutation occurs after convergence and final winding repair, with no subsequent repair or acceptance gate.

    Fix: search for collision-free representable positions, bound the allowed displacement, validate affected triangle geometry, and reload/verify the exported mesh. If float32 cannot represent the required separation within the allowed deviation, report that limitation rather than silently distorting the model.

11. **P2 — the internal timeout is neither a hard time limit nor an explicit failure.**

    Source: [mesh_repair_pipeline.py:139](../src/pipelines/mesh_repair_pipeline.py:139), [mesh_repair_pipeline.py:178](../src/pipelines/mesh_repair_pipeline.py:178).

    The 300-second checks run between selected stages. Loading, a single expensive routine, initial cleanup, or final processing can overrun without interruption. Early timeout paths return a mesh/log tuple normally before writing an output and leave `repair_time_ms=0`; later timeout paths can break convergence and proceed to write a partial result. A mocked elapsed-time check reproduced a normal return with **12 faces, no output file, and zero reported repair time**. This is a control-flow reproduction, not a performance measurement.

    The evaluator's subprocess timeout does enforce a hard limit for its child; that protection does not cover ordinary CLI or library calls.

    Fix: expose a consistent deadline/result contract, poll inside expensive routines, and use process isolation where interruption is required. Timeout must be explicit and must not be followed by a successful status or reuse of a stale output.

12. **P2 — absolute tolerances and late adaptive setup make behavior depend on scale and entry point.**

    Source: [mesh_analyzer.py:117](../src/core/mesh_analyzer.py:117), [mesh_repair_pipeline.py:149](../src/pipelines/mesh_repair_pipeline.py:149), [repair_config.py:16](../src/core/repair_config.py:16), [main.py:42](../main.py:42).

    Loading already merges vertices with an absolute `1e-6` grid before adaptive tolerances are computed. A unit cube loads with **8 vertices**; the same cube scaled to `1e-7` loads with **1 vertex and 12 repeated-vertex faces**. Enabling adaptation later cannot recover this lost topology. The config object is also mutated by adaptive computation, so reusing a pipeline can carry the previous model's merge tolerance into the next import.

    Configuration lookup depends on the working directory. Dataclass defaults differ from the INI: shell threshold is `0.0002` versus `0.002`, boundary welding is `0.0001` versus `0.001`, and planar tolerance is `0.0001` versus `0.001`. The dataclass comment incorrectly labels `0.0002` as 0.2%; it is 0.02%.

    Fix: load raw coordinates without destructive approximate merging, derive per-model tolerances before repair, avoid mutating reusable configuration, and resolve a declared configuration path consistently. Define supported units/scales and unify defaults.

13. **P2 — the test harness cannot provide ordinary pytest coverage of the actual pipeline contract.**

    Source: [test_mesh_repair_pipeline.py:137](../src/test_mesh_repair_pipeline.py:137), [test_mesh_repair_pipeline.py:706](../src/test_mesh_repair_pipeline.py:706).

    The decorator executes tests at import time and catches their exceptions. Pytest collects the decorator itself, causing the missing `fn` fixture error. The script runner correctly exits nonzero when its results contain failures, but the suite's named full-pipeline integration test manually calls a subset of routines; it does not exercise `main.py`, initial cleanup, shell union/filtering, export/reload, or the public pipeline's result contract. `repair_smooth` is absent even though two tests require it. The `-k` option is parsed after decorated tests have already executed, so it does not actually filter execution.

    Fix: make these discoverable pytest tests, remove or explicitly mark unsupported-module cases, and add end-to-end cases covering output extension handling, duplicate sheets, geometric fidelity, hollow semantics, overlapping solids, export/reload, scale, and timeout behavior. Keep routine tests alongside those contract tests.

Four additional complete CLI repairs (`Chair`, `100030`, `Non_Planar_Test`, and `b1`) exited 0 and had zero naked, non-manifold, and degenerate faces on reload. This establishes that the pipeline succeeds on some inputs; it does not resolve the documented geometry and contract defects. `Plane_lower` failed as described above. This audit did not repeat a 100-model Thingi10K benchmark, so it neither verifies nor replaces the README's 93% rate.

One earlier diagnosis was deliberately not repeated as a finding: the pipeline passes a pre-repair winding audit to its Möbius decision, but `crack_mobius_strips` performs a fresh audit internally and returns immediately when conflicts are gone. A stale caller snapshot by itself is therefore not evidence of destructive cracking of already-fixed conflicts in this version.

Recommended sequence: fix temporary-file handling and establish an output-validity gate; remove destructive filtering of artificial shells; repair excision and the hole guard; preserve coherent sheets and cavity roles; then harden overlap handling, scale, export, deadlines, and end-to-end tests. Track geometry retention alongside watertightness before claiming an improved success rate.
