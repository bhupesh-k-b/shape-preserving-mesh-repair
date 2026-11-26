from dataclasses import asdict
from pathlib import Path
import json
import sys
import numpy as np
from src.core.repair_config import RepairConfig
from src.pipelines.mesh_repair_pipeline import MeshRepairPipeline


def main():
    source,output,arrays,report,config,iterations=sys.argv[1:]
    mesh,log=MeshRepairPipeline(RepairConfig(**json.loads(config)))._repair_in_process(source,output,int(iterations))
    np.savez(arrays,vertices=mesh.vertices,faces=mesh.faces,normals=mesh.normals,units=mesh.units or '')
    Path(report).write_text(json.dumps(asdict(log),indent=2)+'\n')


if __name__=='__main__':
    main()
