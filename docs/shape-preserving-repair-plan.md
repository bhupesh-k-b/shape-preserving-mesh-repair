# Plan: preserve the original shape and maximize verified repair coverage

The priority is to preserve the original geometry, parts, thin features, and cavities while repairing as many meshes as possible. Remeshing is permitted when it changes the tessellation without exceeding the geometry contract. Increasing watertightness by moving features, deleting parts, or filling intended cavities does not count as a successful repair.

Keep the existing modular routines and improve them incrementally. Replace the orchestration and acceptance rules first, then add stronger geometry backends for cases the local routines cannot solve. This plan builds on the [current-code audit](pipeline-audit-2026-10-10.md); it does not apply the fixes yet.

**The limit on 100%.** An arbitrary damaged STL does not uniquely specify its intended repair. A missing surface can have several plausible shapes; a zero-thickness sheet cannot become a solid without adding thickness. Two solids touching at one edge cannot always be represented as a single manifold surface while preserving their exact geometry. These inputs need an explicit interpretation or a different output representation. The engineering target is 100% passing supported regressions, a measured 99%+ correct-repair target on a predefined supported benchmark, and no knowingly invalid outputs reported as successful. Those are targets to demonstrate, not a promised rate.

## 1. Define preservation and success before changing algorithms

Use a shape-preserving default policy:

- Preserve the coordinates of existing vertices and the existing surface outside identified defect regions. Start with zero intentional displacement of unaffected geometry.
- Allow exact duplicate removal, edge splitting, and retriangulation of an existing planar surface when its geometric surface remains the same.
- Permit new patch geometry only at classified defects. Record exactly which regions were reconstructed. A hole boundary constrains a patch but does not prove the intended missing shape.
- Preserve modeled components and material/void roles. Component counts alone are insufficient: legitimate union can merge components, while splitting can create artificial fragments.
- Require an explicit distance budget before approximate welding or remeshing can move existing surface geometry. Units and a minimum feature width must be declared or calibrated; a fixed percentage of the global bounding box is not a universal preservation limit.
- Maintain both the raw original and the last accepted candidate. A failed operation must not become the input to the next repair strategy.

Track separate measures:

| Measure | Definition | Release target |
|---|---|---|
| Correct repair rate | Outputs judged both geometrically faithful and valid / all benchmark inputs | Improve toward 99%+ on the predefined supported cohort |
| Overall automatic coverage | Verified automatic outputs / every attempted input | Report separately, including unsupported inputs and timeouts in the denominator |
| Incorrect acceptance rate | Outputs reported as successful that fail independent verification or the reference shape contract / accepted outputs | Zero observed in the release benchmark; report statistical uncertainty |
| Part and cavity retention | Intended components and voids preserved under the declared operation | All labeled regression cases pass |
| Preservation error | Surface deviation on retained original geometry, plus displacement of retained original vertices | Zero intentional movement by default; declared numerical/export budget only |
| Reconstruction quality | Agreement with clean reference surfaces where available; patch validity and plausibility otherwise | Report separately from preservation |

Do not treat a sampled distance maximum as a certified Hausdorff bound. Use dense adaptive surface sampling during development and a bounded-error distance method for release-critical cases where feasible. CGAL provides both approximate and bounded-error Hausdorff distance functions. [CGAL distance documentation](https://doc.cgal.org/6.1.1/Polygon_mesh_processing/group__PMP__distance__grp.html).

Measure both directions of surface deviation. Original-to-output catches deleted features; output-to-original catches extra geometry. Evaluate reconstructed patches separately: a valid cap added to a cube necessarily introduces surface that is absent from the damaged input. Compare the complete repaired mesh with the known clean mesh on synthetic/reference cases.

Area and volume are diagnostic signals, not sufficient acceptance tests. Duplicate removal can reduce area without changing the surface, and volume of an open or self-intersecting input is not a trustworthy reference.

## 2. Handle the current repository as a sequence of reviewable changes

Before implementation, snapshot the current working files and regression assets, including uncommitted changes. Use `Advanced_Mesh_Repairing` as the canonical repository. Compare useful fixes from the refined copy individually; port them only with a reproducing test and before/after fidelity measurements.

| Change | Files primarily affected | Work | Completion gate |
|---|---|---|---|
| PR 1: usable tests and baseline | `src/test_mesh_repair_pipeline.py`, new `tests/`, evaluation scripts | Convert import-time decorators to pytest; retain meaningful routine tests; add the audit reproductions as expected-correctness tests; classify unsupported smoothing tests | Tests collect normally; every known bug has a regression; baseline metrics and failures are recorded |
| PR 2: trustworthy output contract | `main.py`, `mesh_repair_pipeline.py`, `mesh_analyzer.py`, `repair_config.py` | Unique temporary files, alias-safe cleanup, atomic publication, consistent statuses, unconditional post-export verification | Uppercase and in-place cases preserve files; invalid or missing output never receives success |
| PR 3: eliminate destructive cleanup | `initial_cleanup_engine.py`, `repair_shell_union.py`, STL loading | Exact cleanup before filtering; separate connectivity from manifold-sheet detection; preserve parts; raw loading before tolerance merging | Duplicate cube and `Plane_lower` retain their surface; chair preservation test passes |
| PR 4: correct local repair algorithms | `repair_nonmanifold.py`, `repair_topology_separation.py`, `repair_holes.py`, `repair_winding.py` | Apply actual removal masks; coherent sheet pairing; valid hole patches; containment-aware winding | Supported cube, sheet, and nested-cavity regressions pass without distortion |
| PR 5: transactional orchestration | `mesh_repair_pipeline.py`, new policy/result/validation modules | Branch candidates from accepted snapshots; enforce geometry budgets; commit arrays and logs together; route by defect class | Failed candidates cannot damage subsequent attempts; quiet and verbose runs have identical acceptance |
| PR 6: stronger backends and bounded local remeshing | New backend adapters plus shell/hole routines | Integrate supported geometry operations; repair only affected regions; compare multiple candidates | Held-out coverage increases without preservation regressions |
| PR 7: release benchmark and runtime controls | Evaluation scripts, worker execution, dependency configuration | Fixed datasets/seeds, independent checks, hard deadlines, reproducible versions, performance optimization | Release targets are measured on untouched test data; runtime failures are explicit |

The order is deliberate: establish trustworthy tests and success reporting before optimizing repair percentages.

## 3. Specific repairs to existing code

**`main.py` and output handling.** Replace extension-based temporary naming with a unique allocation. Publish only a verified serialized candidate. Keep any existing destination intact on failure. In-place operation should replace the input only after verification; cleanup must never target the input or final destination. Fix configuration lookup so behavior does not depend on the launch directory.

**`mesh_analyzer.py`.** Separate raw import, topology analysis, and approximate welding. Validate finite coordinates and face indices. Analyze non-manifold vertices as well as edges, winding consistency, self-intersections, and material containment. Preserve distinct vertex identities internally instead of using approximate coordinate rounding to decide topology during loading.

**`initial_cleanup_engine.py`.** Retain useful exact matching and orientation logic, but remove shortest-edge-based nearby welding from the default path. Stop automatic deletion of unconnected geometry. Replace broad hole filling and approximate welds with policy-controlled local operations. Initial cleanup must use the same preservation contract as advanced repair. Preserve original vertex coordinates for output rather than restoring unchanged vertices through a numerical normalization round trip.

**`repair_shell_union.py`.** Clean duplicate/degenerate data before shell filtering. Use all shared edges for surface connectivity and a separate representation for manifold sheets. Honor `shell_filter_enabled=False`; remove mandatory micro-shell deletion. Preserve valid small closed components. Classify disjoint, nested, and partially overlapping components independently. Union is a declared operation, not the default response to every multi-part model. Replace the false-success interior-culling fallback for overlaps with a real Boolean operation or an explicit incomplete result.

**`repair_nonmanifold.py`.** Apply `faces_to_remove` and the matching normals mask. Exclude the face itself from neighbor scoring. Update counters only after actual removal. Make deletion a last local option with recorded provenance; coherent separation should run first. A reduction in edge defects does not justify losing visible surface.

**`repair_topology_separation.py`.** Pair compatible incident faces into complete sheets rather than severing all faces at an overused edge. Use geometry, traversal direction, and connectivity together; ambiguous pairing should produce alternative candidates or review status. Duplicate indices without the hardcoded coordinate offsets. Confirm that the resulting sheets remain continuous and that they can be represented in the chosen output format.

**`repair_holes.py`.** Replace the global `0.35 * diagonal` guard with local patch tests. Existing boundary edges must not disqualify a valid patch simply because they are long. Test triangle quality, patch/boundary orientation, new self-intersections, and local surface continuity. Compare newly introduced non-manifold defects with the baseline. Generate several bounded candidates for non-planar holes, preserving their boundary coordinates. Large or ambiguous missing surfaces should not be silently capped solely to improve the edge count.

**`repair_winding.py`.** Separate within-sheet winding consistency from material orientation. Orient outer boundaries and inner cavities according to containment depth and declared solid semantics. Preserve cavity roles through later repair and export; do not independently make every closed shell's volume positive.

**Export stabilization.** Replace arbitrary vertex nudging with an export feasibility check. Never move original coordinates just to make index-separated topology survive STL. Where exact coincident sheets need distinct object identity, prefer a suitable multi-object/indexed representation or report the STL limitation. Manifold's documentation explicitly notes that STL loses topology and recommends 3MF for solid meshes. [Manifold format guidance](https://github.com/elalish/manifold#3d-formats).

## 4. Build an orchestration layer that maximizes coverage without accumulated damage

Keep a canonical in-memory mesh with float64 coordinates, indexed faces, per-face provenance, component/void labels, and an immutable per-model configuration. Derive any tolerances before approximate processing. Use centering/scaling for numerical calculations while preserving the original coordinates of untouched vertices.

Pass this mesh directly between initial cleanup and advanced repair. Remove the intermediate STL round trip from the final architecture so float32 quantization and coordinate-based topology reconstruction cannot silently alter the mesh between stages.

Use this processing order:

1. Strict raw import and initial diagnosis.
2. Exact duplicate/degenerate cleanup under the preservation contract.
3. Component, sheet, containment, and feature classification.
4. Local topology repair and within-sheet orientation.
5. Exact or budgeted local stitching and hole reconstruction.
6. Self-intersection resolution and declared Boolean operations where needed.
7. Material/void orientation and complete validation.
8. Export to a temporary destination, reload, verify, and publish atomically.

This is a dependency order, not a requirement to run every operation on every mesh. Already-valid inputs should take a verified unchanged path. Newly exposed defects can trigger another targeted operation within a finite deadline.

For difficult defects, build a small portfolio of candidates from the same accepted snapshot: alternative sheet pairings, patch triangulations, and backend operations. A candidate must pass hard preservation and validity constraints before it is ranked. Among passing candidates, prefer the smallest affected region and least surface deviation, then better triangle quality and lower runtime. Do not rank a damaged candidate as best merely because it has fewer naked edges.

Each stage should return candidate geometry, changed-region provenance, committed counters, validation results, and failure reasons. Roll back counters with geometry. Cache topology only while its face array remains unchanged.

Suggested public statuses: `VERIFIED_UNCHANGED`, `VERIFIED_REPAIRED`, `NEEDS_REVIEW`, `INVALID_INPUT`, `TIMEOUT`, and `FAILED`. Include the strategy, component effects, reconstructed area, fidelity measurements, validator completeness, and exported-output verification. CLI and evaluation status must use this same result. A timed-out or incomplete validation cannot count as success.

## 5. Add stronger algorithms where they actually fit

**Primary route: local, surface-preserving repair.** Keep corrected Python routines for straightforward defects. Use local edge splitting, constrained triangulation, and bounded remeshing for difficult patches. Preserve existing surface outside the patch. New triangles are acceptable; unintended geometric movement is not.

**Robust geometry route: CGAL adapter.** Evaluate a narrow worker for intersection detection/resolution, difficult hole triangulation, and Boolean operations. CGAL 6.1.1 supports triangle-soup autorefinement and optional iterative snap rounding. Snap rounding may move coordinates and can fail its iteration limit, so enforce the preservation budget, check its return value, and validate the result. Resolving intersections does not by itself guarantee a closed manifold solid. [CGAL autorefinement documentation](https://doc.cgal.org/6.1.1/Polygon_mesh_processing/group__PMP__corefinement__grp.html), [CGAL hole filling](https://doc.cgal.org/6.1.1/Polygon_mesh_processing/group__PMP__hole__filling__grp.html).

**Boolean route: Manifold.** Use `manifold3d` for appropriate already-manifold solids and inspect import/operation status. Raw open or non-manifold meshes cannot simply be handed to it as a universal repair operation; its documented input requirements still apply. [Manifold input requirements](https://github.com/elalish/manifold#manifold-library).

**Optional candidate generator: PyMeshFix.** Benchmark it behind the same geometry gate. Explicitly set `remove_smallest_components=False` and avoid automatic component joining unless that matches the intended operation; the documented repair default discards all but the largest component. Passing the gate, rather than the library name, determines acceptance. [PyMeshFix repair options](https://pymeshfix.pyvista.org/_autosummary/_autosummary/pymeshfix.MeshFix.repair.html).

**Volumetric reconstruction is not the default under the clarified requirement.** SDF/voxel reconstruction can increase closure coverage but can also round edges, erase thin features, or alter cavities. Treat it as an experimental candidate only when a nonzero reconstruction budget is explicitly part of the shape contract and independent measurements demonstrate compliance. Generalized winding numbers can help define inside/outside for imperfect meshes, but do not recover missing intent. [libigl signed-distance documentation](https://github.com/libigl/libigl.github.io/blob/docs/docs/tutorial.md).

Start with one new backend and measure its incremental benefit. Add another only for a documented remaining failure class. Pin the tested dependency versions and run native operations in isolated workers so failures and hard deadlines remain controllable.

## 6. Build evidence for the claimed repair rate

Use four complementary test sets:

| Set | Purpose | Required checks |
|---|---|---|
| Existing regressions | Protect known failures | Chair; `Plane_lower`; duplicate and alternate-triangulation sheets; cube caps; nested cavities; overlaps; uppercase/in-place paths; export collisions; timeouts |
| Clean reference meshes with injected defects | Measure actual recovery accuracy | Start from known geometry; inject gaps, duplicate faces, winding errors, non-manifold junctions, and intersections; compare repaired surfaces/parts/voids to the reference |
| Real-world meshes with reviewed intended repair | Test realistic fidelity | Preserve eyes, teeth, walls, small parts, and cavities; label ambiguous cases explicitly |
| Stress and invariance cases | Test robustness | Scale, translation, face-order permutations, near-coplanar faces, huge coordinates, malformed files, mesh size, deadlines, and absent backends |

Preserve the defect-free identity case: clean valid input must remain geometrically unchanged. Include combined defects, not just one isolated fault at a time. Check both STL and any supported indexed/multi-object export formats.

Fix the dataset manifest and random seeds. Split development and held-out sets by model family/near-duplicate groups before tuning. Declare supported cohorts before seeing test outcomes. On each change, run the small regression suite and its affected cohort; run the full benchmark at milestone gates. Store input hashes, configuration, dependency versions, candidate strategy, failure class, geometry measurements, runtime, and memory.

Use an independent geometry checker and reference meshes so success is not defined solely by the routines being repaired. On real damaged meshes without a reference or reviewed intended outcome, describe the result as verified against stated geometric/topological constraints rather than claiming exact recovery of the unknown original shape.

A 100-model test is too small to establish near-perfect reliability. With zero observed incorrect acceptances in 100 independent representative accepted outputs, the one-sided 95% binomial upper bound is about 2.95%. With 3,000 it is about 0.10%. These statements require reliable labels and independence; thousands of variants of one model do not provide the same evidence.

Investigate residual failures by class. Add a targeted capability only when it improves the held-out correct-repair rate without violating preservation. Improve runtime after correctness gates are working: reuse acceleration structures, vectorize scoring, batch intersection/distance queries, and avoid rebuilding the complete topology for each hole.

## First implementation milestone

Complete PRs 1–4 and run all supplied meshes with the new reporting contract. Required outcomes: no deleted files, no empty replacement of a nonempty mesh, no chair distortion, valid cube caps, coherent separated sheets, retained small modeled components, and correctly oriented cavities. Difficult inputs may still be reported as unresolved at this milestone.

Then complete candidate orchestration and add the strongest backend for the largest remaining failure category. Raise the verified repair rate through better local reconstruction and topology handling, while preserving the geometry contract. Release a percentage claim only after the held-out benchmark supports it.
