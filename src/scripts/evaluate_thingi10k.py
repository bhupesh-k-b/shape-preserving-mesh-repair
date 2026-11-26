"""Evaluate an explicitly declared boundary-defect cohort from HuggingFace."""
from pathlib import Path
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.scripts.evaluate_dataset import evaluate_one,write_results,sha256
from src.core.repair_config import RepairConfig


def evaluate_thingi10k(limit=10,output_dir='data/eval_outputs_thingi10k',csv_path='data/thingi10k_results.csv',timeout=120):
    from datasets import load_dataset
    dataset=load_dataset('Thingi10K/Thingi10K',split='train',streaming=True)
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]; manifest=[]
    with tempfile.TemporaryDirectory(prefix='thingi-benchmark-') as directory:
        for entry in dataset:
            if len(rows)>=limit:
                break
            if entry.get('num_boundary_edges',0)==0:
                continue
            index=len(rows)+1
            source=Path(directory)/f'{index:04d}.npz'
            np.savez(source,vertices=np.asarray(entry['vertices']),faces=np.asarray(entry['faces']))
            output=output_dir/f'{index:04d}-repaired.3mf'
            row=evaluate_one(source,output,RepairConfig(timeout_seconds=timeout))
            row['Model ID']=str(entry.get('file_id',index))
            rows.append(row)
            manifest.append({'model_id':row['Model ID'],'sha256':sha256(source),'cohort':'boundary-defect'})
            write_results(rows,csv_path,manifest)
            print(f"{row['Model ID']}: {row['Status']}",flush=True)
    return write_results(rows,csv_path,manifest)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--timeout',type=float,default=120)
    args=parser.parse_args()
    evaluate_thingi10k(limit=args.limit,timeout=args.timeout)
