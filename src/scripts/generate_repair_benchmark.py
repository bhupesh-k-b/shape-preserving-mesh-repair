"""Generate deterministic, labeled geometry-recovery and invariance cases."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.core.mesh_analyzer import MeshData
from src.core.mesh_io import write_mesh
from src.core.repair_utils import _recompute_normals


def generate(directory,manifest,seed=20261010):
    directory=Path(directory).absolute(); directory.mkdir(parents=True,exist_ok=True)
    manifest=Path(manifest).absolute(); manifest.parent.mkdir(parents=True,exist_ok=True)
    models={'cube':trimesh.creation.box(),
            'icosphere':trimesh.creation.icosphere(subdivisions=1),
            'cylinder':trimesh.creation.cylinder(radius=.7,height=2.2,sections=12)}
    rows=[]
    for family,model in models.items():
        rng=np.random.default_rng(seed)
        missing=int(rng.integers(0,len(model.faces)))
        for scale_index,scale in enumerate([1e-7,1.,1000.]):
            for offset_index,offset in enumerate([0.,1e6]):
                vertices=np.asarray(model.vertices)*scale+offset
                faces=np.array(model.faces,dtype=np.int64)
                reference=directory/f'{family}-{scale_index}-{offset_index}-reference.3mf'
                write_mesh(reference,MeshData(vertices,faces,_recompute_normals(vertices,faces)))
                for defect in ['identity','winding','hole','combined']:
                    damaged=faces.copy()
                    if defect in ['hole','combined']:
                        damaged=np.delete(damaged,missing,axis=0)
                    if defect in ['winding','combined']:
                        damaged[::3]=damaged[::3][:,[0,2,1]]
                    if defect=='combined':
                        damaged=np.vstack([damaged,damaged[:3],[[0,0,1]]])
                    source=directory/f'{family}-{scale_index}-{offset_index}-{defect}.npz'
                    np.savez(source,vertices=vertices,faces=damaged)
                    rows.append({'input':os.path.relpath(source,manifest.parent),
                                 'reference':os.path.relpath(reference,manifest.parent),
                                 'family':family,'defect':defect,
                                 'split':'regression' if family=='cube' else 'held_out_family'})
    manifest.write_text(json.dumps(rows,indent=2)+'\n')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'.benchmarks/reference-inputs')
    parser.add_argument('--manifest',type=Path,default=ROOT/'docs/reference-benchmark-manifest.json')
    args=parser.parse_args()
    rows=generate(args.directory,args.manifest)
    print(f'Generated {len(rows)} labeled cases across three model families; 48 are held-out-family variants.')
