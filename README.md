# Advanced Mesh Repairing

**Live prototype:** [bhupeshbamboria.com/#mesh](https://www.bhupeshbamboria.com/#mesh) — upload an STL and download the repaired file.

**Write-up:** [docs/WRITEUP.pdf](docs/WRITEUP.pdf) — the problem, the approach, and what I learned.

Mesh repair with zero intentional movement of the original surface by default.
The pipeline tries local repairs and isolated CGAL/Manifold candidates, then
checks topology, intersections, material orientation, retained parts, surface
preservation, and the exported file before publishing a result.

A closed mesh alone is insufficient evidence of a correct repair. Damaged inputs
can have several plausible missing surfaces, and some exact contacts cannot be
represented as manifold STL. Those cases may require review. The earlier 93% and
universal 100% claims do not describe this stricter implementation.

## Setup

The tested environment is macOS, Python 3.14, and CGAL 6.1.1. Python dependencies
are pinned in `requirements.txt`. The CGAL worker is required for complete
verification, including when repair backends are disabled.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
brew install cmake cgal
cmake -S native -B .native-build -DCMAKE_BUILD_TYPE=Release
cmake --build .native-build --parallel 2
python3 -m pytest -q
```

For another installation layout, set `MESH_GEOMETRY_WORKER` to the executable.
The native worker requires CGAL 6.1 or later. Windows process termination has
not been implemented or validated; the current hard-deadline mechanism uses
POSIX process groups.

## Repair a mesh

```bash
python3 main.py input.stl repaired.3mf --report repaired.json
python3 main.py input.stl repaired.stl --timeout 120 --report repaired.json
```

Input supports STL and flat indexed 3MF mesh objects. 3MF assembly transforms
and component instances are rejected. Internal evaluation also supports NPZ
meshes. Prefer 3MF when separate sheets or touching parts need distinct vertex
identities; STL stores float32 triangle coordinates and loses those identities.
The pipeline reloads either output format and rejects it if serialization breaks
the preservation or validity checks. It never nudges vertices to force STL to pass.

Default behavior preserves existing surface geometry and represented cavity
roles. Exact duplicate and zero-area triangle cleanup is allowed; local patches
can add missing surfaces. A patch's validity does not prove that it matches the
unknown intended shape. The JSON report records attempts, acceptance reasons,
surface certificates, and new or retriangulated faces.

Optional controls:

- `--max-deviation 0.001`: allow that much existing-surface deviation in input
  coordinate units. Default `0` allows only a feature-capped numerical allowance.
- `--adaptive`: derive per-mesh tolerances within the declared distance budget.
- `--union`: explicitly combine represented solids after independent input-sheet checks. This changes the preservation
  reference to the declared material Boolean boundary and can remove interior faces.
- `--fill-cavities`: with union enabled, intentionally remove enclosed voids.
- `--no-backends`: use local repair candidates; independent validation still runs.
- `--config path.ini`: override the root-relative configuration file.

Exit codes are `0` for `VERIFIED_UNCHANGED` or `VERIFIED_REPAIRED`, `124` for a hard
timeout, and `2` for other unsuccessful results. Existing destinations are
preserved on rejection or timeout. Successful in-place repair is supported.
The report must be a different file from either mesh.

## Evidence and evaluation

The regression suite passes **92 tests**, with **2 explicit smoothing skips**.
The 72-case reference benchmark checks cube, sphere, and cylinder variants with
known recoverable defects, scale changes, and translations. It measures recovery
on that restricted cohort, not universal accuracy. The real-mesh cohort contains
29 inputs without reviewed clean references; its result measures automatic
coverage under the preservation contract.

Final results: **72/72 reference matches**, and **5/29 real meshes verified
(17.2%)** in both STL and 3MF, with 20 review results and 4 timeouts at 45 seconds.
The broad near-100% repair-rate target remains unmet.

An additional **18/18 dense coplanar-overlay cases** match their cube references.
The exact cleanup removes redundant overlays without retriangulating the original
cube. See [the implementation report](docs/shape-preserving-implementation.md) for the
routing fixes and runtime work.

See [the implementation report](docs/shape-preserving-implementation.md),
[reference results](docs/implementation-reference-benchmark.json), and
[real-mesh results](docs/implementation-dataset-3mf.json). Unsupported inputs and
timeouts remain in the denominator.

```bash
python3 src/scripts/evaluate_dataset.py --dataset data/test_meshes --samples 100 --seed 20261010 --timeout 45 --format 3mf --output-dir .benchmarks/eval-3mf --csv docs/eval-3mf.csv
python3 src/scripts/generate_repair_benchmark.py --help
python3 src/scripts/evaluate_dataset.py --manifest docs/reference-benchmark-manifest.json --format 3mf --output-dir .benchmarks/reference-outputs --csv docs/reference-results.csv
```

Evaluation stores input/output hashes, failures, timings, package versions,
source hashes, and native-worker hashes. `requirements-evaluation.txt` adds the
optional remote-dataset dependencies. Historical audit files describe the
pre-fix snapshot and are retained for comparison.

## Code and recovery

- `src/pipelines/mesh_repair_pipeline.py`: candidate orchestration and atomic output.
- `src/core/validation.py`: topology, material, part, and geometry acceptance checks.
- `native/geometry_worker.cpp`: exact-predicate geometry operations and coverage checks.
- `src/core/backend_worker.py`: isolated Manifold64 Boolean operations.
- `src/core/mesh_io.py`: raw import and STL/3MF serialization.
- `tests/`: preservation, runtime, export, and real-mesh regressions.

For development diagnostics, `scratch/diagnose_residual_candidates.py MODEL_ID`
saves attempted candidate meshes and up to 8,192 exact intersection pairs under
`.benchmarks/residual-diagnostics/`. Truncated diagnostics are explicitly marked;
they are not acceptance certificates or published repaired outputs.
