import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
from src.core.mesh_analyzer import MeshData
from src.core.geometry_backend import ROOT, require_boolean_sheets


def candidate(action,mesh,config):
    config.check_deadline()
    if action=='manifold' and config.union_enabled:
        require_boolean_sheets(mesh,config)
    with tempfile.TemporaryDirectory(prefix='mesh-candidate-') as directory:
        source=Path(directory)/'source.npz'; target=Path(directory)/'candidate.npz'
        np.savez(source,vertices=mesh.vertices,faces=mesh.faces,normals=mesh.normals)
        remaining=config._deadline-time.perf_counter() if config._deadline else config.timeout_seconds
        if remaining<=0:
            raise TimeoutError('Candidate deadline exceeded')
        try:
            run=subprocess.run([sys.executable,'-m','src.core.backend_worker',action,str(source),str(target),
                                json.dumps({'preserve_cavities':config.preserve_cavities,'hollow':config.union_hollow})],
                               cwd=ROOT,capture_output=True,text=True,timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError('Native candidate exceeded its deadline') from exc
        if run.returncode:
            raise RuntimeError(run.stderr.strip().splitlines()[-1] if run.stderr else 'Candidate backend failed')
        with np.load(target,allow_pickle=False) as data:
            if action=='overlap':
                return bool(data['interior_overlap'].item())
            result=MeshData(data['vertices'],data['faces'],data['normals'])
        config.check_deadline()
        return result
