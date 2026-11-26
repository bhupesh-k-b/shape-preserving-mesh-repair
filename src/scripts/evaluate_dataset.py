"""Reproducible repair evaluation: coverage, preservation, and reference fidelity."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.core.mesh_io import load_mesh
from src.core.repair_config import RepairConfig
from src.core.validation import surface_preservation
from src.pipelines.mesh_repair_pipeline import MeshRepairPipeline


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_watertight(path):
    try:
        import trimesh
        mesh=load_mesh(path)
        independent=trimesh.Trimesh(mesh.vertices,mesh.faces,process=False)
        return bool(len(mesh.faces) and independent.is_watertight and independent.is_winding_consistent)
    except (ValueError,OSError):
        return False


def evaluate_one(source,output,config=None,reference=None):
    config=config or RepairConfig()
    source,output=Path(source),Path(output)
    original_hash=sha256(source)
    started=time.perf_counter()
    mesh,log=MeshRepairPipeline(config).repair(source,output,verbose=False)
    verified=bool(log.success and output.exists() and log.validation.get('complete') and log.validation.get('valid'))
    topology=is_watertight(output) if verified else False
    if verified and not topology:
        verified=False
        log.status='INDEPENDENT_CHECK_FAILED'
        log.reasons.append('Independent trimesh topology check failed')
    reference_ok=None; reference_error=None
    if reference is not None and verified:
        reference_ok,reference_error,_=surface_preservation(load_mesh(reference),mesh,config,bidirectional=True)
        if not np.isfinite(reference_error):
            reference_error=None
        if not reference_ok:
            log.status='REFERENCE_MISMATCH'
            verified=False
            log.reasons.append('Repaired mesh does not meet the clean-reference surface contract')
    report_path=output.with_suffix(output.suffix+'.json')
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(asdict(log),indent=2,allow_nan=False)+'\n')
    if sha256(source)!=original_hash:
        raise RuntimeError('Benchmark input was modified')
    return {
        'Model ID':source.stem,'Status':log.status,'Verified':verified,'Watertight':topology,
        'Reference Match':reference_ok,'Reference Error Bound':reference_error,
        'Time Taken (s)':round(time.perf_counter()-started,3),
        'Source SHA256':original_hash,'Output SHA256':sha256(output) if log.success and output.exists() else '',
        'Strategy':log.strategy,'Surface Error Bound':log.source_surface_error if verified else None,
        'New or Retriangulated Area':log.reconstructed_area if verified else None,
        'Naked Edges':log.validation.get('naked_edges'),'Non-Manifold Edges':log.validation.get('non_manifold_edges'),
        'Non-Manifold Vertices':log.validation.get('non_manifold_vertices'),
        'Reasons':'; '.join(log.reasons)
    }


def write_results(rows,path,manifest):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else ['Model ID','Status','Verified'])
        writer.writeheader(); writer.writerows(rows)
    known=[row for row in rows if row['Reference Match'] is not None]
    summary={'attempted':len(rows),'verified':sum(row['Verified'] for row in rows),
             'automatic_coverage':sum(row['Verified'] for row in rows)/len(rows) if rows else 0,
             'reference_labeled_outputs':len(known),'reference_correct':sum(row['Reference Match'] for row in known),
             'statuses':{status:sum(row['Status']==status for row in rows) for status in sorted({row['Status'] for row in rows})},
             'manifest':manifest,
             'runtime':runtime_description(),
             'interpretation':'Coverage verifies explicit constraints. It is not recovery accuracy without a labeled intended reference.'}
    path.with_suffix('.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


_RUNTIME=None
def runtime_description():
    global _RUNTIME
    if _RUNTIME is None:
        from src.core.geometry_backend import worker_path
        files=sorted(ROOT.glob('src/**/*.py'))+[ROOT/'main.py',ROOT/'native/geometry_worker.cpp']
        _RUNTIME={'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in
                  ['numpy','numpy-stl','scipy','trimesh','manifold3d']},
                  'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in files},
                  'geometry_worker_sha256':sha256(worker_path()) if worker_path().exists() else None}
    return _RUNTIME


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/test_meshes')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'data/eval_outputs')
    parser.add_argument('--csv',type=Path,default=ROOT/'data/evaluation_results.csv')
    parser.add_argument('--samples',type=int,default=100)
    parser.add_argument('--seed',type=int,default=20261010)
    parser.add_argument('--timeout',type=float,default=120)
    parser.add_argument('--format',choices=['stl','3mf'],default='stl')
    parser.add_argument('--manifest',type=Path,help='Fixed JSON list of input paths, or rows containing input and reference paths')
    parser.add_argument('--max-deviation',type=float,default=0)
    parser.add_argument('--union',action='store_true')
    args=parser.parse_args(argv)
    if args.manifest:
        entries=json.loads(args.manifest.read_text())
        entries=[{'input':entry} if isinstance(entry,str) else entry for entry in entries]
        base=args.manifest.parent
        selected=[(Path(e['input']) if Path(e['input']).is_absolute() else base/e['input'],
                   (Path(e['reference']) if Path(e['reference']).is_absolute() else base/e['reference']) if e.get('reference') else None) for e in entries]
    else:
        candidates=sorted(p for p in args.dataset.iterdir() if p.suffix.lower() in ['.stl','.3mf'])
        if not candidates:
            parser.error('No supported mesh files in dataset')
        selected=[(p,None) for p in random.Random(args.seed).sample(candidates,min(args.samples,len(candidates)))]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest=[{'input':str(p.absolute()),'sha256':sha256(p),'reference':str(ref.absolute()) if ref else None} for p,ref in selected]
    cfg=RepairConfig(timeout_seconds=args.timeout,max_surface_deviation=args.max_deviation,union_enabled=args.union)
    rows=[]
    for index,(source,reference) in enumerate(selected,1):
        output=args.output_dir/f'{index:04d}-{source.stem}-repaired.{args.format}'
        row=evaluate_one(source,output,cfg,reference)
        rows.append(row)
        print(f"[{index}/{len(selected)}] {source.name}: {row['Status']} ({row['Time Taken (s)']}s)",flush=True)
        write_results(rows,args.csv,manifest)
    summary=write_results(rows,args.csv,manifest)
    print(f"Verified coverage: {summary['verified']}/{summary['attempted']} ({100*summary['automatic_coverage']:.1f}%)")
    return 0


if __name__=='__main__':
    raise SystemExit(main())
