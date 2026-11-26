"""Fixed large coplanar-overlay recovery cohort; all cases have cube references."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.core.mesh_analyzer import MeshData
from src.core.mesh_io import write_mesh
from src.core.repair_utils import _recompute_normals


def main():
    destination=ROOT/'.benchmarks/iteration2/overlay-inputs'
    destination.mkdir(parents=True,exist_ok=True)
    cube=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                   [0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=float)
    faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],
                    [0,1,5],[0,5,4],[2,3,7],[2,7,6],
                    [0,4,7],[0,7,3],[1,2,6],[1,6,5]],dtype=int)
    entries=[]
    for n in [8,10,16]:
        grid=np.array([[i/n,j/n,1.] for i in range(n+1) for j in range(n+1)])
        overlay=[]
        for i in range(n):
            for j in range(n):
                a=8+i*(n+1)+j; b=a+n+1
                overlay.extend([[a,b,b+1],[a,b+1,a+1]])
        for scale in [1e-7,1.,1000.]:
            for offset in [0.,1e6]:
                name=f'cube-overlay-{n}-{scale:g}-{offset:g}'
                source=destination/(name+'.npz')
                reference=destination/(name+'-reference.3mf')
                v=np.vstack([cube,grid])*scale+offset
                f=np.vstack([faces,overlay])
                np.savez(source,vertices=v,faces=f,normals=_recompute_normals(v,f))
                clean_vertices=cube*scale+offset
                write_mesh(reference,MeshData(clean_vertices,faces,_recompute_normals(clean_vertices,faces)))
                entries.append({'input':'../'+str(source.relative_to(ROOT)),
                                'reference':'../'+str(reference.relative_to(ROOT)),
                                'family':'cube','defect':'coplanar_overlay',
                                'overlay_faces':len(overlay),'scale':scale,'translation':offset,
                                'split':'targeted_regression'})
    (ROOT/'docs/coplanar-overlay-benchmark-manifest.json').write_text(json.dumps(entries,indent=2)+'\n')
    print(f'Generated {len(entries)} fixed cases')


if __name__=='__main__':
    main()
