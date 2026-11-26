"""Save rejected candidates and exact intersection pairs for development."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core import geometry_backend
from src.core.mesh_io import load_mesh
from src.core.repair_config import RepairConfig
from src.pipelines import mesh_repair_pipeline as pipeline


def diagnose(name):
    destination = ROOT / '.benchmarks/residual-diagnostics' / name
    destination.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/test_meshes' / (name + '.stl')
    original_validate = pipeline.validate
    records = []

    def capture(mesh, config, cache=None):
        config.check_deadline()
        number = len(records)
        np.savez(destination / f'candidate-{number}.npz', vertices=mesh.vertices,
                 faces=mesh.faces, normals=mesh.normals)
        report = original_validate(mesh, config, cache)
        pairs = geometry_backend.execute('intersections', mesh, config) if report.self_intersection else {}
        records.append({'candidate': number, 'validation': report.to_dict(), 'intersections': pairs})
        (destination / 'candidates.json').write_text(json.dumps(records, indent=2) + '\n')
        return report

    pipeline.validate = capture
    started = time.perf_counter()
    try:
        _, log = pipeline.MeshRepairPipeline(RepairConfig(timeout_seconds=90))._repair_in_process(
            source, destination / 'result.3mf')
        (destination / 'result.json').write_text(json.dumps(log.__dict__, indent=2) + '\n')
        print(name, log.status, round(time.perf_counter() - started, 2), flush=True)
    finally:
        pipeline.validate = original_validate


if __name__ == '__main__':
    for name in sys.argv[1:]:
        diagnose(name)
