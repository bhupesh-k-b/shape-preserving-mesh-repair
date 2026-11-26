"""Download a boundary-defect cohort and use the common verified evaluator."""
import argparse
import csv
import io
from pathlib import Path
import sys
import tempfile
import urllib.request
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.scripts.evaluate_dataset import evaluate_one,write_results,sha256
from src.core.repair_config import RepairConfig

THINGI10K_BASE_URL='https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main'
THINGI10K_METADATA_URL=f'{THINGI10K_BASE_URL}/metadata'
THINGI10K_NPZ_URL=f'{THINGI10K_BASE_URL}/npz'


def fetch_csv(url):
    with urllib.request.urlopen(url,timeout=30) as response:
        return list(csv.DictReader(response.read().decode('utf-8').splitlines()))


def load_npz_from_url(url):
    with urllib.request.urlopen(url,timeout=30) as response:
        content=response.read()
    with np.load(io.BytesIO(content),allow_pickle=False) as arrays:
        return arrays['vertices'],arrays['facets']


def evaluate_hf(limit=10,output_dir='data/eval_outputs_hf',csv_path='data/hf_results.csv',timeout=120):
    records=fetch_csv(f'{THINGI10K_METADATA_URL}/geometry_data.csv')
    selected=[row for row in records if int(row['num_boundary_edges'])>0][:limit]
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]; manifest=[]
    with tempfile.TemporaryDirectory(prefix='thingi-hf-') as directory:
        for index,record in enumerate(selected,1):
            file_id=record['file_id']
            if not str(file_id).isdigit():
                raise ValueError('Unexpected dataset model identifier')
            v,f=load_npz_from_url(f'{THINGI10K_NPZ_URL}/{file_id}.npz')
            source=Path(directory)/f'{index:04d}.npz'
            np.savez(source,vertices=v,faces=f)
            row=evaluate_one(source,output_dir/f'{file_id}-repaired.3mf',RepairConfig(timeout_seconds=timeout))
            row['Model ID']=file_id; rows.append(row)
            manifest.append({'model_id':file_id,'sha256':sha256(source),'cohort':'boundary-defect'})
            write_results(rows,csv_path,manifest)
            print(f"{file_id}: {row['Status']}",flush=True)
    return write_results(rows,csv_path,manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--timeout',type=float,default=120)
    args=parser.parse_args()
    evaluate_hf(limit=args.limit,timeout=args.timeout)
