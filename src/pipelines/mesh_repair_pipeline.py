"""Shape-preserving repair with isolated candidates and verified publication."""
from dataclasses import asdict
from pathlib import Path
import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import numpy as np

if __package__ in {None,''}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from src.core.mesh_analyzer import MeshData, build_edge_map, check_naked_and_nonmanifold
from src.core.mesh_io import write_stl, write_mesh, load_mesh
from src.core.repair_config import RepairConfig, RepairLog
from src.core.repair_utils import _recompute_normals
from src.core.validation import validate, surface_preservation, face_keys, nonmanifold_vertices, closed_part_bounds, closed_parts_preserved
from src.core import geometry_backend
from src.core.repair_backends import candidate as backend_candidate
from src.routines.repair_degenerates import repair_degenerate_faces, repair_duplicate_faces
from src.routines.repair_topology_separation import repair_topology_separation
from src.routines.repair_winding import repair_winding_order, check_winding_consistency
from src.routines.repair_holes import repair_planar_holes
from src.routines.repair_welding import global_vertex_weld
from src.routines.repair_shell_union import filter_small_shells
from src.routines.repair_nonmanifold import repair_non_manifold
from src.routines.repair_shell_union import _find_shells


ROOT=Path(__file__).resolve().parents[2]


def _stabilize_float32_vertex_splits(vertices):
    """Compatibility helper: topology identity must never nudge original geometry."""
    return vertices.copy()


def empty_mesh():
    return MeshData(np.zeros((0,3)),np.zeros((0,3),dtype=int),np.zeros((0,3)))


class MeshRepairPipeline:
    def __init__(self,config=None):
        self.config=config or RepairConfig()

    def _count_defects(self,vertices,faces):
        naked,nm=check_naked_and_nonmanifold(build_edge_map(faces))
        if not len(faces):
            return len(naked),len(nm),0,0
        tri=vertices[faces]
        degen=int(np.count_nonzero(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)==0))
        dup=len(faces)-len(np.unique(np.sort(faces,axis=1),axis=0))
        return len(naked),len(nm),degen,dup

    def _count_winding_conflicts(self,vertices,faces):
        return len(check_winding_consistency(vertices,faces).conflicted_edges)

    def repair(self,input_path,output_path,verbose=True,max_iterations=3):
        """Hard deadline for the complete job, including native descendants.

        Keep the existing tuple API; RepairLog now carries the explicit status.
        The destination is replaced only after a fully verified child result.
        """
        started=time.perf_counter()
        output=Path(output_path).absolute()
        output.parent.mkdir(parents=True,exist_ok=True)
        timeout=float(self.config.timeout_seconds)
        if not np.isfinite(timeout) or timeout<=0:
            raise ValueError('timeout_seconds must be positive and finite')
        with tempfile.TemporaryDirectory(prefix='.mesh-job-',dir=output.parent) as directory:
            job=Path(directory)
            artifact=job/('verified'+(output.suffix or '.stl'))
            cfg=asdict(self.config); cfg.pop('_deadline',None)
            args=[sys.executable,'-m','src.core.pipeline_worker',str(Path(input_path).absolute()),str(artifact),
                  str(job/'result.npz'),str(job/'result.json'),json.dumps(cfg),str(max_iterations)]
            run=subprocess.Popen(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
            try:
                stdout,stderr=run.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(run.pid,signal.SIGKILL)
                run.communicate()
                return empty_mesh(),RepairLog(status='TIMEOUT',reasons=['Hard repair deadline exceeded; destination preserved'],
                                               repair_time_ms=(time.perf_counter()-started)*1000)
            except BaseException:
                if run.poll() is None:
                    os.killpg(run.pid,signal.SIGKILL)
                run.communicate()
                raise
            if run.returncode or not (job/'result.json').exists():
                return empty_mesh(),RepairLog(status='FAILED',reasons=[stderr.strip() or 'Repair worker failed'],
                                               repair_time_ms=(time.perf_counter()-started)*1000)
            log=RepairLog(**json.loads((job/'result.json').read_text()))
            with np.load(job/'result.npz',allow_pickle=False) as data:
                mesh=MeshData(data['vertices'],data['faces'],data['normals'],units=str(data['units'].item()) or None)
            if log.success:
                if time.perf_counter()-started>=timeout:
                    log.status='TIMEOUT'; log.reasons=['Publication deadline exceeded; destination preserved']
                elif not artifact.exists():
                    log.status='FAILED'; log.reasons=['Verified output artifact missing']
                else:
                    os.replace(artifact,output)
            log.repair_time_ms=(time.perf_counter()-started)*1000
            if verbose:
                print(log.summary())
            return mesh,log

    def _repair_in_process(self,input_path,output_path,max_iterations=3):
        started=time.perf_counter(); mesh=empty_mesh(); log=RepairLog()
        try:
            mesh=load_mesh(input_path)
            cfg=self.config.for_mesh(mesh.vertices,mesh.faces)
            cfg._deadline=started+cfg.timeout_seconds
            cfg.check_deadline()
            base_log=RepairLog()
            f,n=repair_degenerate_faces(mesh.vertices,mesh.faces,mesh.normals,cfg,base_log)
            if not len(f):
                log.status='INVALID_INPUT'; log.reasons=['No nondegenerate surface remains']
                return mesh,log
            # Ignore only mathematically zero-area input, never actual thin features.
            reference=MeshData(mesh.vertices,f.copy(),n.copy())
            original_reference=reference
            f,n=repair_duplicate_faces(f,n,base_log)
            base=MeshData(mesh.vertices.copy(),f,_recompute_normals(mesh.vertices,f))
            source_parts=closed_part_bounds(base)
            geometry_cache={}; attempts=[]
            best=base; best_report=None; best_score=None; best_log=copy.deepcopy(base_log)

            def local(source,source_log):
                v=source.vertices.copy(); f=source.faces.copy(); n=source.normals.copy()
                active=copy.deepcopy(source_log)
                f,n=repair_degenerate_faces(v,f,n,cfg,active)
                f,n=repair_duplicate_faces(f,n,active)
                if cfg.shell_filter_enabled:
                    v,f,n=filter_small_shells(v,f,n,cfg,active,verbose=False)
                v,f,n=repair_winding_order(v,f,n,cfg,active,verbose=False)
                em=build_edge_map(f)
                if check_naked_and_nonmanifold(em)[1] or nonmanifold_vertices(f,em):
                    v,f=repair_topology_separation(v,f,active,cfg)
                    n=_recompute_normals(v,f)
                    v,f,n=repair_winding_order(v,f,n,cfg,active,verbose=False)
                if cfg.max_surface_deviation>0:
                    v,f,n=global_vertex_weld(v,f,n,cfg,active)
                for _ in range(max(0,max_iterations)):
                    cfg.check_deadline()
                    naked,_=check_naked_and_nonmanifold(build_edge_map(f))
                    if not naked:
                        break
                    if len(naked)>cfg.max_hole_edges*2:
                        break  # route dense boundary work to isolated native triangulation
                    previous=face_keys(MeshData(v,f,n))
                    v,f,n=repair_planar_holes(v,f,n,cfg,active)
                    if np.array_equal(previous,face_keys(MeshData(v,f,n))):
                        break
                    v,f,n=repair_winding_order(v,f,n,cfg,active,verbose=False)
                return MeshData(v,f,n),active

            def evaluate(name,candidate,active):
                nonlocal best,best_report,best_score,best_log,log
                cfg.check_deadline()
                report=validate(candidate,cfg,geometry_cache)
                parts_retained=closed_parts_preserved(source_parts,candidate,cfg)
                if parts_retained:
                    accepted,error,certificate=surface_preservation(reference,candidate,cfg)
                else:
                    accepted,error,certificate=False,float('inf'),'closed_part_retention_failed'
                entry={'strategy':name,'preserves_source':bool(accepted),'source_surface_error_bound':float(error) if np.isfinite(error) else None,
                       'certificate':certificate,'closed_parts_preserved':parts_retained,'validation':report.to_dict()}
                attempts.append(entry)
                # Route subsequent attempts from the best preservation-certified
                # geometry and its matching counters, never the last attempt.
                score=(not report.complete,
                       report.naked_edges+report.non_manifold_edges*2+report.non_manifold_vertices,
                       report.winding_conflicts+report.degenerate_faces+report.duplicate_faces,
                       report.cavity_orientation_errors+report.ambiguous_material_roles+report.zero_volume_shells,
                       report.self_intersection)
                if accepted and (best_score is None or score<best_score):
                    best=candidate; best_report=report; best_score=score
                    best_log=copy.deepcopy(active)
                if not accepted or not report.valid:
                    return False
                # The exported mesh is the acceptance target, not float64 arrays.
                try:
                    write_mesh(output_path,candidate,units=mesh.units)
                    serialized=load_mesh(output_path)
                    post=validate(serialized,cfg,geometry_cache)
                    preserved,output_error,out_certificate=surface_preservation(reference,serialized,cfg)
                    preserved=preserved and closed_parts_preserved(source_parts,serialized,cfg)
                    export_ok,quantization_error,_=surface_preservation(candidate,serialized,cfg,bidirectional=True)
                except (ValueError,RuntimeError) as exc:
                    entry['export_rejection']=str(exc)
                    return False
                if not post.valid or not preserved or not export_ok:
                    entry['export_validation']=post.to_dict()
                    export_error=max(output_error,quantization_error)
                    entry['export_preservation_error_bound']=float(export_error) if np.isfinite(export_error) else None
                    if not post.valid and len(candidate.vertices)!=len(serialized.vertices):
                        entry['format_note']='STL loses separate vertex identities; use .3mf for coincident manifold sheets'
                    return False
                log=active; best=serialized
                raw_equal=(len(mesh.faces)==len(candidate.faces) and
                           np.array_equal(mesh.vertices[mesh.faces],candidate.vertices[candidate.faces]))
                log.status='VERIFIED_UNCHANGED' if raw_equal else 'VERIFIED_REPAIRED'
                log.strategy=name; log.validation=post.to_dict(); log.attempts=attempts
                log.verification_basis='declared_material_boolean_boundary' if cfg.union_enabled else 'original_surface'
                log.source_surface_error=max(error,output_error)
                log.distance_error_bound=0.0 if out_certificate=='exact_facets' else float(cfg.max_surface_deviation)
                original_keys=face_keys(original_reference); result_keys=face_keys(serialized)
                new=~np.isin(result_keys,original_keys)
                triangles=serialized.vertices[serialized.faces[new]]
                log.reconstructed_area=float(np.sum(np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)*.5))
                log.provenance={'source_face_count':len(original_reference.faces),'exact_source_faces_retained':int(np.count_nonzero(np.isin(original_keys,result_keys))),
                                'new_or_retriangulated_face_indices':np.flatnonzero(new).tolist(),
                                'surface_certificate':out_certificate,'intentional_surface_budget':cfg.max_surface_deviation,
                                'original_closed_parts':len(source_parts),'closed_parts_preserved':True,
                                'component_policy':'merge authorized by explicit union' if cfg.union_enabled else 'retain distinct known closed components',
                                'reconstruction_assumption':'Closed boundary patches; missing target surface is not provided'}
                log.reasons=[]
                return True

            primary,primary_log=local(base,base_log)
            if not cfg.union_enabled and evaluate('local-preserving',primary,primary_log):
                return best,log
            if cfg.union_enabled:
                # Union still needs coherent input sheets; certify the local
                # preparation before it becomes a Boolean input.
                kept,_,_=surface_preservation(reference,primary,cfg)
                if kept:
                    best=primary; best_log=copy.deepcopy(primary_log)
            # Remove only representationally redundant sheets, proving complete
            # source coverage for every deletion. A small component is never
            # sufficient evidence of noise.
            try:
                v,f,n=repair_winding_order(base.vertices,base.faces,base.normals,cfg,RepairLog())
                trim_log=copy.deepcopy(base_log)
                v,f=repair_topology_separation(v,f,trim_log,cfg)
                trimmed=MeshData(v,f,_recompute_normals(v,f))
                removed=0
                for _ in range(2):
                    groups=_find_shells(trimmed.faces,build_edge_map(trimmed.faces))
                    changed=False
                    for group in sorted(groups,key=len):
                        cfg.check_deadline()
                        if len(group)>64 or len(group)==len(trimmed.faces):
                            continue
                        mask=np.ones(len(trimmed.faces),dtype=bool); mask[group]=False
                        trial=MeshData(trimmed.vertices,trimmed.faces[mask],trimmed.normals[mask])
                        if not closed_parts_preserved(source_parts,trial,cfg):
                            continue
                        kept,_,_=surface_preservation(trimmed,trial,cfg)
                        if kept:
                            removed+=len(group); trimmed=trial; changed=True
                            break
                    if not changed:
                        break
                if removed:
                    trim_log.non_manifold_removed+=removed
                    trimmed,trim_log=local(trimmed,trim_log)
                    if not cfg.union_enabled and evaluate('covered-sheet-cleanup',trimmed,trim_log):
                        return best,log
            except RuntimeError as exc:
                attempts.append({'strategy':'covered-sheet-cleanup','rejection':str(exc)})
            # A bounded excision candidate is accepted only if the geometry
            # proof demonstrates no loss of original surface or known parts.
            if check_naked_and_nonmanifold(build_edge_map(base.faces))[1]:
                excision_log=copy.deepcopy(base_log)
                f,n=repair_non_manifold(base.vertices,base.faces,base.normals,cfg,excision_log)
                if len(f)<len(base.faces):
                    excised,excision_log=local(MeshData(base.vertices,f,n),excision_log)
                    if not cfg.union_enabled and evaluate('preservation-checked-excision',excised,excision_log):
                        return best,log
            if cfg.backend_enabled:
                if not cfg.union_enabled and best_report is not None and best_report.self_intersection:
                    try:
                        cleaned=geometry_backend.execute('clean_coplanar',base,cfg)
                        if len(cleaned.faces)<len(base.faces):
                            cleaned_log=copy.deepcopy(base_log)
                            cleaned_log.non_manifold_removed+=len(base.faces)-len(cleaned.faces)
                            cleaned,cleaned_log=local(cleaned,cleaned_log)
                            if evaluate('CGAL-covered-coplanar-cleanup',cleaned,cleaned_log):
                                return best,log
                    except RuntimeError as exc:
                        attempts.append({'strategy':'CGAL-covered-coplanar-cleanup','rejection':str(exc)})
                safe_primary=best
                safe_log=copy.deepcopy(best_log)
                try:
                    filled=geometry_backend.execute('fill',safe_primary,cfg)
                    filled_log=copy.deepcopy(safe_log)
                    filled_log.faces_added+=max(0,len(filled.faces)-len(safe_primary.faces))
                    filled,filled_log=local(filled,filled_log)
                    if not cfg.union_enabled and evaluate('CGAL-hole-triangulation',filled,filled_log):
                        return best,log
                    safe_primary=best; safe_log=copy.deepcopy(best_log)
                except RuntimeError as exc:
                    attempts.append({'strategy':'CGAL-hole-triangulation','rejection':str(exc)})
                # Boolean candidates operate on closed coherent sheets and must
                # retain the original surface unless union was explicitly chosen.
                try:
                    boolean=backend_candidate('manifold',safe_primary,cfg)
                    boolean_log=copy.deepcopy(safe_log)
                    if cfg.union_enabled:
                        # Declared solid Boolean semantics define the target boundary.
                        reference=boolean
                    if evaluate('Manifold64-boolean',boolean,boolean_log):
                        return best,log
                except RuntimeError as exc:
                    attempts.append({'strategy':'Manifold64-boolean','rejection':str(exc)})
                try:
                    refined=geometry_backend.execute('refine',base,cfg)
                    refined_before_cleanup=refined
                    refined=geometry_backend.execute('clean_coplanar',refined,cfg)
                    refined_log=copy.deepcopy(base_log)
                    refined_log.non_manifold_removed+=len(refined_before_cleanup.faces)-len(refined.faces)
                    refined,refined_log=local(refined,refined_log)
                    if evaluate('CGAL-autorefine-local',refined,refined_log):
                        return best,log
                    try:
                        boolean=backend_candidate('manifold',refined,cfg)
                        if evaluate('CGAL-Manifold64',boolean,refined_log):
                            return best,log
                    except RuntimeError as exc:
                        attempts.append({'strategy':'CGAL-Manifold64','rejection':str(exc)})
                except RuntimeError as exc:
                    attempts.append({'strategy':'CGAL-autorefine-local','rejection':str(exc)})
            log=copy.deepcopy(best_log); log.status='NEEDS_REVIEW'; log.attempts=attempts
            log.validation=best_report.to_dict() if best_report else {}
            log.reasons=['No candidate met both preservation and serialized-output validity']
            if best_report:
                log.reasons+=best_report.reasons
            return best,log
        except TimeoutError as exc:
            log.status='TIMEOUT'; log.reasons=[str(exc)]; return mesh,log
        except (ValueError,OSError) as exc:
            log.status='INVALID_INPUT'; log.reasons=[str(exc)]; return mesh,log
        except Exception as exc:
            log.status='FAILED'; log.reasons=[f'{type(exc).__name__}: {exc}']; return mesh,log
        finally:
            log.repair_time_ms=(time.perf_counter()-started)*1000


if __name__=='__main__':
    from main import main
    raise SystemExit(main())
