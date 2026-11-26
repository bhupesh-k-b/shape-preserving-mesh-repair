"""
Initial cleanup mesh repair pipeline.
Fast, single-pass repair using hash-based edge matching.
"""
import os
import sys
import time
import tempfile
from pathlib import Path

# Allow running directly
if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from src.core.initial_cleanup_engine import StlFile, repair, stats_summary


class InitialCleanupPipeline:
    """Drop-in alternative to MeshRepairPipeline using initial cleanup algorithms."""

    def __init__(self, iterations=3):
        self.iterations = iterations

    def repair(self, input_path: str, output_path: str, verbose: bool = True):
        t0 = time.perf_counter()
        if verbose:
            print(f"\n  Loading  : {input_path}")

        stl = StlFile.from_file(input_path)
        from src.core.mesh_analyzer import MeshData
        MeshData.from_stl(input_path)  # reject truncated/nonfinite input first
        if verbose:
            print(f"  Facets   : {stl.stats.number_of_facets}")
            print(f"  BBox     : {stl.stats.min} -> {stl.stats.max}")
            print(f"\n  Repairing (Initial Cleanup)...\n")

        repair(stl, iterations=self.iterations, verbose=verbose)

        if stl.stats.number_of_facets==0:
            raise ValueError('Exact cleanup cannot replace a model with an empty result')
        output=Path(output_path).absolute()
        output.parent.mkdir(parents=True,exist_ok=True)
        descriptor,temp_path=tempfile.mkstemp(prefix='.exact-cleanup-',suffix='.stl',dir=output.parent)
        os.close(descriptor)
        try:
            stl.write_binary(temp_path)
            MeshData.from_stl(temp_path)
            os.replace(temp_path,output)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        elapsed = (time.perf_counter() - t0) * 1000

        if verbose:
            print(f"\n  Written  : {output_path}")
            print(f"  Time     : {elapsed:.1f} ms")
            print(stats_summary(stl))

        return stl


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Hardcoded file names - you can change these here
    inp = r"Thingi10k/Thingi10k/raw_meshes/101952.stl"
    out = r"101952_repaired.stl"

    # Ensure the path is correct relative to this script or provide absolute path
    if not os.path.exists(inp):
        # Try to find it in the root if relative path fails
        root_inp = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "100104.stl"))
        if os.path.exists(root_inp):
            inp = root_inp
        else:
            print(f"[ERROR] File not found: {inp}")
            sys.exit(1)

    pipeline = InitialCleanupPipeline()
    pipeline.repair(inp, out, verbose=True)

