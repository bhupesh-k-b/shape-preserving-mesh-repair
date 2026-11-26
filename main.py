"""CLI for validated, shape-preserving mesh repair."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from src.core.repair_config import RepairConfig
from src.pipelines.mesh_repair_pipeline import MeshRepairPipeline


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('output',help='STL or indexed multi-object .3mf output')
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().with_name('repair_config.ini'))
    parser.add_argument('--adaptive',action='store_true')
    parser.add_argument('--max-deviation',type=float,default=None,help='Allowed existing surface deviation in input coordinate units; default 0')
    parser.add_argument('--timeout',type=float,default=None)
    parser.add_argument('--union',action='store_true',help='Explicitly combine solid components, preserving cavities')
    parser.add_argument('--fill-cavities',action='store_true',help='With --union, intentionally eliminate enclosed voids')
    parser.add_argument('--no-backends',action='store_true',help='Use only local candidates; independent validation is still required')
    parser.add_argument('--quiet',action='store_true')
    parser.add_argument('--report',type=Path,help='Write the complete structured result as JSON')
    args=parser.parse_args(argv)
    if args.report:
        for mesh_path in (Path(args.input), Path(args.output)):
            aliases_mesh = args.report.resolve() == mesh_path.resolve()
            if args.report.exists() and mesh_path.exists():
                aliases_mesh = aliases_mesh or args.report.samefile(mesh_path)
            if aliases_mesh:
                parser.error('--report must use a different file from the input and output mesh')
    cfg=RepairConfig.from_ini(str(args.config))
    cfg.adaptive=args.adaptive or cfg.adaptive
    cfg.union_enabled=args.union or cfg.union_enabled
    if args.fill_cavities:
        if not cfg.union_enabled:
            parser.error('--fill-cavities requires --union')
        cfg.preserve_cavities=False
    if args.max_deviation is not None:
        cfg.max_surface_deviation=args.max_deviation
    if args.timeout is not None:
        cfg.timeout_seconds=args.timeout
    cfg.backend_enabled=cfg.backend_enabled and not args.no_backends
    try:
        _,log=MeshRepairPipeline(cfg).repair(args.input,args.output,verbose=not args.quiet)
    except ValueError as exc:
        parser.error(str(exc))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(asdict(log),indent=2,allow_nan=False)+'\n')
    if log.success:
        if not args.quiet:
            print(f'[SUCCESS] {log.status}: {Path(args.output).absolute()}')
        return 0
    print(f'[{log.status}] '+ '; '.join(log.reasons))
    return 124 if log.status=='TIMEOUT' else 2


if __name__=='__main__':
    raise SystemExit(main())
