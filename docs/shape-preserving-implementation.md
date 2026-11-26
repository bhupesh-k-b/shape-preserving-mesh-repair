# Shape-preserving repair implementation — 2026-10-10

The implementation now rejects repairs that fail the original-surface contract,
even when their edge counts improve. It preserves the existing tuple API while
adding explicit statuses and reports. The near-100% real-world repair target has
not been demonstrated. Reference recovery and unlabeled real-mesh coverage are
reported separately below.

## Changes completed

1. Converted the legacy test harness to pytest and added audit reproductions,
   real Chair/Plane regressions, export checks, and runtime/failure tests.
2. Removed default nearby welding, tiny-part deletion, intermediate STL round
   trips, and export vertex nudges. Raw STL import merges exact coordinate matches
   only; indexed 3MF retains separate identities and source units.
3. Corrected non-manifold deletion masks, coherent sheet separation, local cap
   guards, and cavity winding. Redundant coplanar sheets can be removed only when
   a surface-coverage certificate proves that no original surface is lost.
4. Added per-mesh policy copies, candidate rollback, retained closed-part checks,
   and complete validation of the serialized artifact before atomic publication.
5. Added isolated CGAL checking, hole triangulation, triangle-soup autorefinement,
   surface coverage, and bounded-distance operations. Added float64 Manifold
   material Booleans and proper-overlap checks. Missing verifiers never pass.
6. Added complete-job POSIX process deadlines, descendant termination, explicit
   timeout status, and preservation of existing outputs on unsuccessful jobs.
7. Added reproducible evaluation, source/input/output hashes, reference checks,
   dependency pins, and JSON reports. CLI report aliases cannot overwrite meshes.

The C++ worker uses exact predicates and constructions for geometry operations.
Its output coordinates are serialized doubles and are checked again downstream.
Surface-preservation certificates either retain exact facets or prove coverage
within a numerical/declared distance envelope. The numerical allowance is capped
by local feature dimensions. Strict mode permits no intentional deformation.
For explicit nonzero budgets, bounded Hausdorff queries use their upper error
envelope. Sampled development distances are not presented as certified bounds.

Explicit `--union` authorizes removal of internal material surfaces. Reports name
the resulting Boolean boundary as the verification reference rather than claiming
that all original internal surfaces survived. Cavity elimination is a separate
explicit option. Default repair retains represented parts and cavity roles.

## Verification

- Full suite after the continuation: **92 passed, 2 skipped**, 24.21 seconds.
- Both skipped tests require an unshipped smoothing module. Smoothing is outside
  the zero-displacement objective; the skips are explicit rather than hidden.
- The legacy cube-with-fins test now requires the valid cube cap to close while
  leaving the coplanar fins unresolved. Closing those fins would reward invalid
  zero-volume shells; the revised test checks no new non-manifold edges.
- Python compilation and native build were checked. Repository whitespace checks
  and input-hash verification are part of the final check.
- Before implementation, the legacy suite had 23 passing and 7 failing tests.
  After conversion and added reproductions, the baseline had 23 failures,
  24 passes, and 2 skips. See the baseline logs in this directory.

The final benchmark summaries are
[reference cases](implementation-reference-benchmark.json),
[real meshes / 3MF](implementation-dataset-3mf.json), and
[real meshes / STL](implementation-dataset-stl.json). Per-case CSVs and per-output
JSON logs retain failure reasons and attempted strategies. Final measurements
use seed `20261010`, zero intentional surface deviation, and 45-second job limits.
The STL and 3MF cohorts ran concurrently, so timings include machine contention.

| Cohort | Verified | Clean-reference matches | Review | Timeout |
|---|---:|---:|---:|---:|
| Restricted synthetic references / 3MF | 72/72 (100%) | 72/72 | 0 | 0 |
| Dense coplanar-overlay references / 3MF | 18/18 (100%) | 18/18 | 0 | 0 |
| Real meshes / 3MF | 5/29 (17.2%) | No labels | 20 | 4 |
| Real meshes / STL | 5/29 (17.2%) | No labels | 20 | 4 |

The five verified real cases are `b1`, `test10`, `Non_Planar_Test` (unchanged),
`Plane_lower`, and `Chair` (repaired). `90276`, accepted by the earlier intermediate
checker, is now rejected: the stronger indexed component checks detect a remaining
intersection. Its Boolean candidate also fails the strict source-surface certificate.
It is not counted as a successful repair.

[Final check evidence](implementation-final-checks.json) confirms that benchmark
source and native-worker hashes match the final implementation, all 29 mesh assets
match the backup, and only verified outputs were published in the fresh real-mesh
output directory. [Residual analysis](implementation-residual-analysis.json) retains
the full rejected-candidate records. Among the 20 review cases, every model had an
intersection in at least one candidate, 19 had boundary/incidence defects in at least
one candidate, and 18 had a candidate fail source preservation. These categories
overlap and describe attempted repairs rather than mutually exclusive input classes.

The [continuation report](shape-preserving-iteration2.md) documents the later
candidate-routing fix, exact coplanar cleanup, winding/boundary optimization,
intersection diagnostics, and independent input checks for explicit unions.
Default real-mesh coverage remains 5/29; the continuation improves correctness
and specific repair routes without claiming that the broad repair-rate target
has been achieved.

### Benchmark scope

The 72 labeled cases are three primitive families, three scales (`1e-7`, `1`,
`1000`), two translations (`0`, `1e6`), and four states: identity, winding errors,
one missing triangle, and combined winding/duplicate/zero-area defects. All outputs
are compared bidirectionally with clean references. The 48 sphere/cylinder variants
are marked as held-out families in the manifest; they are correlated synthetic
variants, not an independent population study.

The 29 real meshes have no reviewed clean target labels. A verified result means
the existing surface and stated topology/material constraints passed. It does
not establish the intended shape of newly reconstructed surfaces. Duplicate or
closely related models in this small cohort also limit generalization.

## Remaining work toward the repair-rate target

The acceptance foundation and corrected local/native routes are implemented.
The plan's broad release target and reviewed realistic reference set remain open.
The next implementation work should be driven by the residual reports:

1. Label intended material and patch geometry for ambiguous damaged models before
   measuring recovery accuracy. Preserve a family-separated test set during tuning.
2. Improve local intersection routing and alternative sheet/patch candidates.
   Exact autorefinement alone does not turn every intersecting soup into a valid
   solid; eliminating occupied overlap may require an explicit material union.
3. Optimize expensive boundary, containment, and refinement cases within the hard
   deadline. Raising the timeout changes coverage and should be measured separately.
4. Add representative thin-feature, cavity, multipart, and combined-defect reference
   cases. The current 72 primitive variants do not establish 99% on realistic data.

Global voxel/SDF reconstruction and unrestricted smoothing are not enabled. They
need an explicit nonzero geometry budget and reference evidence before inclusion.
An optional MeshFix worker path exists but was not installed, evaluated, or enabled
in the main candidate portfolio. A full feature-width policy, peak-memory telemetry,
alternate pairing search, and reviewed patch plausibility are not implemented.

Only flat mesh-object 3MF input is supported; transformed assemblies are rejected.
STL can lose topology or exceed the strict float64-to-float32 geometry envelope.
Windows hard-deadline process management remains unimplemented.

## Handling the current code

Keep this repository as the canonical version and review the changes by layer:
tests/output contract, conservative cleanup, local routines, validation/native
workers, then evaluation. Do not replace it wholesale with the refined sibling.
Use 3MF for the strict default and inspect `NEEDS_REVIEW`/`TIMEOUT` reports before
changing geometry budgets or material semantics.

The pre-change snapshot includes the dirty working tree and all 29 existing mesh
assets. It is stored in `.repair-backups/` and has a SHA256 manifest. The prior user
deletion of `v29d_engraved.stl` was left intact. No commit, push, or deployment was
performed. Historical audit source locations refer to that pre-change state.
