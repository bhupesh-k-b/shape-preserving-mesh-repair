"""
Algorithms for filling planar and non-planar holes.
Includes boundary loop extraction, ear-clipping, and centroid fans.
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional

from src.core.repair_config import RepairConfig, RepairLog
from src.core.repair_utils import _face_area, _face_normal, _recompute_normals
from src.core.mesh_analyzer import build_edge_map, check_naked_and_nonmanifold


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers & Loop Ordering
# ─────────────────────────────────────────────────────────────────────────────

def _order_boundary_loop(edges):
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    start = edges[0][0]
    loop = [start]
    prev = None
    cur = start

    while True:
        neighbors = adj[cur]
        next_v = None

        for nb in neighbors:
            if nb != prev:
                next_v = nb
                break

        if next_v is None:
            break

        if next_v == start:
            return loop

        loop.append(next_v)
        prev, cur = cur, next_v

        if len(loop) > len(edges) + 5:
            break

    return loop if len(loop) >= 3 else None


def _vertex_turn_key(prev_idx: Optional[int],
                     cur_idx: int,
                     cand_idx: int,
                     vertices: np.ndarray) -> Tuple[float, float, int]:
    if prev_idx is None:
        edge = vertices[cand_idx] - vertices[cur_idx]
        return (0.0, float(np.linalg.norm(edge)), int(cand_idx))
    incoming = vertices[cur_idx] - vertices[prev_idx]
    outgoing = vertices[cand_idx] - vertices[cur_idx]
    in_len = float(np.linalg.norm(incoming))
    out_len = float(np.linalg.norm(outgoing))
    if in_len < 1e-15 or out_len < 1e-15:
        return (float("inf"), out_len, int(cand_idx))
    cosine = float(np.clip(np.dot(incoming, outgoing) / (in_len * out_len), -1.0, 1.0))
    return (-cosine, out_len, int(cand_idx))


def robust_boundary_loop(
    naked_edges: List[Tuple[int, int]],
    vertices: np.ndarray
) -> List[Dict[str, object]]:
    if not naked_edges:
        return []

    adjacency: Dict[int, Set[int]] = defaultdict(set)
    unique_edges: Set[Tuple[int, int]] = set()
    for a, b in naked_edges:
        a_i, b_i = int(a), int(b)
        if a_i == b_i:
            continue
        edge = (a_i, b_i) if a_i < b_i else (b_i, a_i)
        if edge in unique_edges:
            continue
        unique_edges.add(edge)
        adjacency[a_i].add(b_i)
        adjacency[b_i].add(a_i)

    unused = set(unique_edges)
    result: List[Dict[str, object]] = []

    def consume_walk(start: int, prev: Optional[int]) -> List[int]:
        walk = [start]
        cur = start
        prev_local = prev
        safety = len(unique_edges) * 2 + 100
        while safety > 0:
            safety -= 1
            candidates = []
            for nb in adjacency[cur]:
                edge = (cur, nb) if cur < nb else (nb, cur)
                if edge in unused and nb != prev_local:
                    candidates.append(nb)
            if not candidates and prev_local is not None:
                for nb in adjacency[cur]:
                    edge = (cur, nb) if cur < nb else (nb, cur)
                    if edge in unused:
                        candidates.append(nb)
            if not candidates:
                break
            next_v = min(
                candidates,
                key=lambda nb: _vertex_turn_key(prev_local, cur, nb, vertices)
            )
            edge = (cur, next_v) if cur < next_v else (next_v, cur)
            unused.remove(edge)
            prev_local, cur = cur, next_v
            if cur == start:
                walk.append(cur)
                break
            walk.append(cur)
        return walk

    while unused:
        seed = next(iter(unused))
        component = set()
        stack = [seed[0], seed[1]]
        while stack:
            v = stack.pop()
            if v in component:
                continue
            component.add(v)
            stack.extend(list(adjacency[v]))

        while True:
            component_edges = [e for e in unused if e[0] in component or e[1] in component]
            if not component_edges:
                break
            odd_vertices = sorted(v for v in component if len(adjacency[v]) % 2 == 1)
            start = odd_vertices[0] if odd_vertices else min(min(e) for e in component_edges)
            walk = consume_walk(start, None)
            if len(walk) < 2:
                unused.discard(component_edges[0])
                continue
            closed = len(walk) > 2 and walk[0] == walk[-1]
            ordered = walk[:-1] if closed else walk
            result.append({
                "vertices": ordered,
                "closed": closed,
                "edges": len(ordered) if closed else max(len(ordered) - 1, 0),
            })

    return result


def _clean_loop_vertices(loop_vertices: List[int]) -> List[int]:
    cleaned: List[int] = []
    for v in loop_vertices:
        vi = int(v)
        if not cleaned or cleaned[-1] != vi:
            cleaned.append(vi)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    if len(cleaned) < 3:
        return cleaned

    unique_loop: List[int] = []
    seen: Set[int] = set()
    for v in cleaned:
        if v in seen:
            break
        unique_loop.append(v)
        seen.add(v)
    return unique_loop


def _loop_edges_are_present(loop_vertices: List[int],
                            edge_set: Set[Tuple[int, int]]) -> bool:
    if len(loop_vertices) < 3:
        return False
    for i, a in enumerate(loop_vertices):
        b = loop_vertices[(i + 1) % len(loop_vertices)]
        key = (a, b) if a < b else (b, a)
        if key not in edge_set:
            return False
    return True


def _split_repeated_vertex_cycles(loop_vertices: List[int]) -> List[List[int]]:
    """Split a closed boundary walk with repeated vertices into simple cycles."""
    work = [int(v) for v in loop_vertices]
    if len(work) > 1 and work[0] == work[-1]:
        work = work[:-1]
    if len(work) < 3:
        return []

    cycles: List[List[int]] = []

    def consume(path: List[int]) -> None:
        seen: Dict[int, int] = {}
        for idx, vertex in enumerate(path):
            if vertex not in seen:
                seen[vertex] = idx
                continue

            start = seen[vertex]
            cycle = path[start:idx]
            if len(cycle) >= 3:
                cycles.append(cycle)

            remainder = path[:start + 1] + path[idx + 1:]
            if len(remainder) >= 3:
                consume(remainder)
            return

        if len(path) >= 3:
            cycles.append(path)

    consume(work)
    return cycles


def _patch_delta_if_safe(
    patch_faces: List[np.ndarray],
    edge_usage: Dict[Tuple[int, int], int],
    existing_face_keys: Set[Tuple[int, int, int]],
    max_faces_per_edge: int
) -> Optional[Tuple[Dict[Tuple[int, int], int], Set[Tuple[int, int, int]]]]:
    """
    Validate a candidate patch against current topology and return
    incremental edge/face updates when safe.
    """
    edge_delta: Dict[Tuple[int, int], int] = {}
    face_delta: Set[Tuple[int, int, int]] = set()

    for face in patch_faces:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        if a == b or b == c or a == c:
            return None

        face_key = tuple(sorted((a, b, c)))
        if face_key in existing_face_keys or face_key in face_delta:
            return None
        face_delta.add(face_key)

        for u, v in ((a, b), (b, c), (c, a)):
            edge = (u, v) if u < v else (v, u)
            next_count = edge_usage.get(edge, 0) + edge_delta.get(edge, 0) + 1
            if next_count > max_faces_per_edge:
                return None
            edge_delta[edge] = edge_delta.get(edge, 0) + 1

    return edge_delta, face_delta


def _segment_cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _segment_intersection_params(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
    tol: float = 1e-12
) -> Optional[Tuple[float, float, np.ndarray]]:
    r = p1 - p0
    s = q1 - q0
    denom = _segment_cross_2d(r, s)
    if abs(denom) <= tol:
        return None
    qp = q0 - p0
    t = _segment_cross_2d(qp, s) / denom
    u = _segment_cross_2d(qp, r) / denom
    if t < -tol or t > 1.0 + tol or u < -tol or u > 1.0 + tol:
        return None
    point = p0 + t * r
    return float(t), float(u), point


def _canonical_cycle(vertices: List[int]) -> Tuple[int, ...]:
    if not vertices:
        return tuple()
    cyc = [int(v) for v in vertices]
    n = len(cyc)
    fwd = [tuple(cyc[i:] + cyc[:i]) for i in range(n)]
    rev_vertices = list(reversed(cyc))
    rev = [tuple(rev_vertices[i:] + rev_vertices[:i]) for i in range(n)]
    return min(min(fwd), min(rev))


def _component_reference_normal(
    component_edges: List[Tuple[int, int]],
    vertices: np.ndarray,
    faces: np.ndarray,
    edge_map: Dict[Tuple[int, int], List[int]]
) -> np.ndarray:
    acc = np.zeros(3, dtype=np.float64)
    for a, b in component_edges:
        edge = (a, b) if a < b else (b, a)
        for fi in edge_map.get(edge, []):
            acc += _face_normal(vertices, faces[int(fi)])
    norm = float(np.linalg.norm(acc))
    if norm > 1e-12:
        return -acc / norm

    comp_vertices = sorted({int(a) for a, _ in component_edges} | {int(b) for _, b in component_edges})
    pts = vertices[np.array(comp_vertices, dtype=np.int32)]
    centered = pts - pts.mean(axis=0)
    if len(pts) >= 3:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        n = vh[-1]
        n_norm = float(np.linalg.norm(n))
        if n_norm > 1e-12:
            return n / n_norm
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _detect_boundary_intersections_sweep(
    segments: List[Dict[str, object]],
    tol: float = 1e-12
) -> Dict[int, List[Tuple[float, np.ndarray]]]:
    if len(segments) < 2:
        return {}

    order = sorted(range(len(segments)), key=lambda idx: float(segments[idx]["min_x"]))
    active: List[int] = []
    hits: Dict[int, List[Tuple[float, np.ndarray]]] = defaultdict(list)

    for sid in order:
        seg = segments[sid]
        min_x = float(seg["min_x"])
        min_y = float(seg["min_y"])
        max_y = float(seg["max_y"])
        p0 = np.array(seg["p"], dtype=np.float64)
        p1 = np.array(seg["q"], dtype=np.float64)
        a0 = int(seg["a"])
        b0 = int(seg["b"])

        active = [
            aid for aid in active
            if float(segments[aid]["max_x"]) >= min_x - tol
        ]

        for aid in active:
            other = segments[aid]
            if min_y > float(other["max_y"]) + tol or float(other["min_y"]) > max_y + tol:
                continue

            c0 = int(other["a"])
            d0 = int(other["b"])
            shared_vertices = ({a0, b0} & {c0, d0})
            inter = _segment_intersection_params(
                p0, p1,
                np.array(other["p"], dtype=np.float64),
                np.array(other["q"], dtype=np.float64),
                tol=tol
            )
            if inter is None:
                continue
            t, u, pt = inter
            t_clamped = float(np.clip(t, 0.0, 1.0))
            u_clamped = float(np.clip(u, 0.0, 1.0))

            if shared_vertices and (
                t_clamped <= tol or t_clamped >= 1.0 - tol or
                u_clamped <= tol or u_clamped >= 1.0 - tol
            ):
                continue
            if (t_clamped <= tol or t_clamped >= 1.0 - tol) and (u_clamped <= tol or u_clamped >= 1.0 - tol):
                continue

            hits[sid].append((t_clamped, pt))
            hits[aid].append((u_clamped, pt))

        active.append(sid)

    return hits


def _split_edges_at_intersections(
    segments: List[Dict[str, object]],
    intersections: Dict[int, List[Tuple[float, np.ndarray]]],
    vertices: np.ndarray,
    vertex_xy: Dict[int, np.ndarray],
    tol: float
) -> Tuple[np.ndarray, List[Tuple[int, int]], Dict[int, np.ndarray]]:
    new_points: List[np.ndarray] = []
    point_cache: Dict[Tuple[int, int, int], int] = {}
    split_edges: Set[Tuple[int, int]] = set()
    xy_map: Dict[int, np.ndarray] = {
        int(k): np.array(v, dtype=np.float64) for k, v in vertex_xy.items()
    }
    next_idx = int(len(vertices))

    def get_or_create_vertex(p3: np.ndarray, p2: np.ndarray) -> int:
        nonlocal next_idx
        key = tuple(float(value) for value in p3)
        if key in point_cache:
            return point_cache[key]
        vid = next_idx
        next_idx += 1
        point_cache[key] = vid
        new_points.append(np.array(p3, dtype=np.float64))
        xy_map[vid] = np.array(p2, dtype=np.float64)
        return vid

    for sid, seg in enumerate(segments):
        a = int(seg["a"])
        b = int(seg["b"])
        p3a = vertices[a]
        p3b = vertices[b]
        p2a = np.array(seg["p"], dtype=np.float64)
        p2b = np.array(seg["q"], dtype=np.float64)

        params = [0.0, 1.0]
        for t, _ in intersections.get(sid, []):
            params.append(float(t))
        params = sorted(params)

        deduped: List[float] = []
        for t in params:
            if not deduped or abs(t - deduped[-1]) > 1e-10:
                deduped.append(float(np.clip(t, 0.0, 1.0)))

        chain: List[int] = []
        for t in deduped:
            if t <= 1e-10:
                vid = a
                p2 = p2a
            elif t >= 1.0 - 1e-10:
                vid = b
                p2 = p2b
            else:
                p3 = (1.0 - t) * p3a + t * p3b
                p2 = (1.0 - t) * p2a + t * p2b
                vid = get_or_create_vertex(p3, p2)
            chain.append(int(vid))
            if int(vid) not in xy_map:
                xy_map[int(vid)] = np.array(p2, dtype=np.float64)

        for i in range(len(chain) - 1):
            u = int(chain[i])
            v = int(chain[i + 1])
            if u == v:
                continue
            edge = (u, v) if u < v else (v, u)
            split_edges.add(edge)

    if new_points:
        vertices_out = np.vstack([vertices, np.array(new_points, dtype=np.float64)])
    else:
        vertices_out = vertices
    return vertices_out, list(split_edges), xy_map


def _extract_simple_cycles_from_planar_graph(
    edges: List[Tuple[int, int]],
    vertex_xy: Dict[int, np.ndarray]
) -> List[List[int]]:
    adjacency: Dict[int, Set[int]] = defaultdict(set)
    for a, b in edges:
        if a == b:
            continue
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))

    # Prune dangling chains; only the 2-core can contain cycles.
    changed = True
    while changed:
        changed = False
        dangling = [v for v, nbrs in adjacency.items() if len(nbrs) < 2]
        if not dangling:
            break
        changed = True
        for v in dangling:
            for nb in list(adjacency[v]):
                adjacency[nb].discard(v)
            adjacency.pop(v, None)

    if not adjacency:
        return []

    ordered_neighbors: Dict[int, List[int]] = {}
    neighbor_pos: Dict[int, Dict[int, int]] = {}
    for v, nbrs in adjacency.items():
        base = np.array(vertex_xy[v], dtype=np.float64)
        ordered = sorted(
            nbrs,
            key=lambda nb: float(np.arctan2(vertex_xy[nb][1] - base[1], vertex_xy[nb][0] - base[0]))
        )
        ordered_neighbors[v] = ordered
        neighbor_pos[v] = {int(nb): i for i, nb in enumerate(ordered)}

    visited_directed: Set[Tuple[int, int]] = set()
    found: List[List[int]] = []
    found_keys: Set[Tuple[int, ...]] = set()
    max_steps = max(64, 16 * len(edges))

    for u, nbrs in ordered_neighbors.items():
        for v in nbrs:
            start = (int(u), int(v))
            if start in visited_directed:
                continue

            cycle = [start[0]]
            cur_u, cur_v = start
            closed = False
            local_steps = 0

            while local_steps < max_steps:
                local_steps += 1
                visited_directed.add((cur_u, cur_v))
                cycle.append(cur_v)

                nbs = ordered_neighbors.get(cur_v)
                pos_map = neighbor_pos.get(cur_v)
                if not nbs or pos_map is None:
                    break
                if cur_u not in pos_map:
                    break
                idx = pos_map[cur_u]
                next_w = int(nbs[(idx - 1) % len(nbs)])
                cur_u, cur_v = cur_v, next_w

                if (cur_u, cur_v) == start:
                    closed = True
                    break
                if (cur_u, cur_v) in visited_directed:
                    break

            if not closed:
                continue

            cleaned = _clean_loop_vertices(cycle)
            if len(cleaned) < 3 or len(set(cleaned)) != len(cleaned):
                continue
            pts = np.array([vertex_xy[int(vtx)] for vtx in cleaned], dtype=np.float64)
            area = abs(_polygon_area_2d(pts))
            if area <= 1e-12:
                continue
            key = _canonical_cycle(cleaned)
            if key in found_keys:
                continue
            found_keys.add(key)
            found.append(cleaned)

    return found


def _build_clean_boundary_loops(
    vertices: np.ndarray,
    faces: np.ndarray,
    naked_edges: List[Tuple[int, int]],
    edge_map: Dict[Tuple[int, int], List[int]],
    config: RepairConfig
) -> Tuple[np.ndarray, List[List[int]]]:
    if not naked_edges:
        return vertices, []

    unique_edges: Set[Tuple[int, int]] = set()
    adjacency: Dict[int, Set[int]] = defaultdict(set)
    for a, b in naked_edges:
        ai, bi = int(a), int(b)
        if ai == bi:
            continue
        edge = (ai, bi) if ai < bi else (bi, ai)
        if edge in unique_edges:
            continue
        unique_edges.add(edge)
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])

    if not unique_edges:
        return vertices, []

    visited_vertices: Set[int] = set()
    components: List[List[Tuple[int, int]]] = []
    for seed in list(adjacency.keys()):
        if seed in visited_vertices:
            continue
        stack = [seed]
        comp_vertices: Set[int] = set()
        while stack:
            v = stack.pop()
            if v in comp_vertices:
                continue
            comp_vertices.add(v)
            visited_vertices.add(v)
            stack.extend(list(adjacency[v]))
        # Enumerate this component's own edges instead of scanning every edge
        # once for each disconnected boundary component.
        comp_edges = [(a, b) for a in sorted(comp_vertices) for b in sorted(adjacency[a]) if a < b]
        if comp_edges:
            components.append(comp_edges)

    cleaned_loops: List[List[int]] = []
    verts_out = vertices
    split_tol = max(config.merge_tolerance * 1e-3, 1e-7)

    for comp_edges in components:
        config.check_deadline()
        if len(comp_edges) < 3:
            continue
        comp_vertices_sorted = sorted(
            {int(a) for a, _ in comp_edges} | {int(b) for _, b in comp_edges}
        )
        comp_normal = _component_reference_normal(comp_edges, verts_out, faces, edge_map)
        comp_pts = verts_out[np.array(comp_vertices_sorted, dtype=np.int32)]
        comp_pts2d, _, _ = _project_to_2d(comp_pts, comp_normal)
        comp_xy = {
            int(v): np.array(comp_pts2d[i], dtype=np.float64)
            for i, v in enumerate(comp_vertices_sorted)
        }

        segments: List[Dict[str, object]] = []
        for a, b in comp_edges:
            p = np.array(comp_xy[int(a)], dtype=np.float64)
            q = np.array(comp_xy[int(b)], dtype=np.float64)
            segments.append({
                "a": int(a),
                "b": int(b),
                "p": p,
                "q": q,
                "min_x": float(min(p[0], q[0])),
                "max_x": float(max(p[0], q[0])),
                "min_y": float(min(p[1], q[1])),
                "max_y": float(max(p[1], q[1])),
            })

        intersections = _detect_boundary_intersections_sweep(segments)
        if not intersections:
            fallback = robust_boundary_loop(comp_edges, verts_out)
            closed = [
                _clean_loop_vertices(list(item["vertices"]))
                for item in fallback
                if item["closed"] and len(item["vertices"]) >= 3
            ]
            if closed:
                for loop in closed:
                    if len(loop) >= 3:
                        cleaned_loops.append(loop)
            continue

        verts_out, split_edges, xy_map = _split_edges_at_intersections(
            segments, intersections, verts_out, comp_xy, split_tol
        )
        cycles = _extract_simple_cycles_from_planar_graph(split_edges, xy_map)
        if not cycles:
            fallback = robust_boundary_loop(comp_edges, verts_out)
            closed = [
                _clean_loop_vertices(list(item["vertices"]))
                for item in fallback
                if item["closed"] and len(item["vertices"]) >= 3
            ]
            if not closed:
                continue
            best = max(
                closed,
                key=lambda cyc: abs(
                    _polygon_area_2d(
                        np.array([xy_map.get(v, comp_xy.get(v, np.zeros(2))) for v in cyc], dtype=np.float64)
                    )
                )
            )
            if len(best) >= 3:
                cleaned_loops.append(best)
            continue

        best_cycle = max(
            cycles,
            key=lambda cyc: abs(
                _polygon_area_2d(np.array([xy_map[int(v)] for v in cyc], dtype=np.float64))
            )
        )
        if len(best_cycle) >= 3:
            cleaned_loops.append(best_cycle)

    return verts_out, cleaned_loops

# ─────────────────────────────────────────────────────────────────────────────
#  2D Projection & Ear Clipping
# ─────────────────────────────────────────────────────────────────────────────

def _project_to_2d(pts_3d: np.ndarray, normal: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n   = normal / (np.linalg.norm(normal) + 1e-15)
    ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    u   = np.cross(ref, n);  u /= (np.linalg.norm(u) + 1e-15)
    v   = np.cross(n, u);    v /= (np.linalg.norm(v) + 1e-15)
    centered=pts_3d-pts_3d[0]
    scale=float(np.linalg.norm(np.ptp(centered,axis=0)))
    if scale>0:
        centered=centered/scale
    pts_2d = np.column_stack([centered.dot(u), centered.dot(v)])
    return pts_2d, u, v


def _cross2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _point_in_triangle_2d(p, a, b, c) -> bool:
    d1 = _cross2d(p, a, b)
    d2 = _cross2d(p, b, c)
    d3 = _cross2d(p, c, a)
    has_neg = (d1 < -1e-10) or (d2 < -1e-10) or (d3 < -1e-10)
    has_pos = (d1 >  1e-10) or (d2 >  1e-10) or (d3 >  1e-10)
    return not (has_neg and has_pos)


def _polygon_area_2d(loop_2d: np.ndarray) -> float:
    x = loop_2d[:, 0]
    y = loop_2d[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _is_simple_polygon_2d(loop_2d: np.ndarray) -> bool:
    n = int(len(loop_2d))
    if n < 3:
        return False

    for i in range(n):
        a0 = loop_2d[i]
        a1 = loop_2d[(i + 1) % n]
        for j in range(i + 1, n):
            # Adjacent edges share endpoints by definition.
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            b0 = loop_2d[j]
            b1 = loop_2d[(j + 1) % n]
            inter = _segment_intersection_params(a0, a1, b0, b1, tol=1e-12)
            if inter is None:
                continue
            t, u, _ = inter
            t = float(np.clip(t, 0.0, 1.0))
            u = float(np.clip(u, 0.0, 1.0))
            if 1e-10 < t < 1.0 - 1e-10 and 1e-10 < u < 1.0 - 1e-10:
                return False
    return True


def _ear_clip(loop_2d: np.ndarray, loop_indices: List[int]
              ) -> List[Tuple[int, int, int]]:
    n      = len(loop_2d)
    tris   = []
    idxs   = list(range(n))

    area = sum(
        _cross2d(loop_2d[idxs[i]], loop_2d[idxs[(i+1) % n]], loop_2d[idxs[(i+2) % n]])
        for i in range(len(idxs))
    )
    if area < 0:
        idxs.reverse()

    max_iters = n * n
    iters     = 0

    while len(idxs) > 3 and iters < max_iters:
        iters  += 1
        n_cur   = len(idxs)
        ear_found = False

        for i in range(n_cur):
            prev_i = idxs[(i - 1) % n_cur]
            curr_i = idxs[i]
            next_i = idxs[(i + 1) % n_cur]

            A = loop_2d[prev_i]
            B = loop_2d[curr_i]
            C = loop_2d[next_i]

            cross = _cross2d(A, B, C)
            if cross <= 1e-10:
                continue

            inside = False
            for j in range(n_cur):
                ji = idxs[j]
                if ji in (prev_i, curr_i, next_i):
                    continue
                if _point_in_triangle_2d(loop_2d[ji], A, B, C):
                    inside = True
                    break
            if inside:
                continue

            tris.append((
                loop_indices[prev_i],
                loop_indices[curr_i],
                loop_indices[next_i],
            ))
            idxs.pop(i)
            ear_found = True
            break

        if not ear_found:
            break

    if len(idxs) == 3:
        tris.append((
            loop_indices[idxs[0]],
            loop_indices[idxs[1]],
            loop_indices[idxs[2]],
        ))

    return tris


def _safe_ear_clip(loop_2d: np.ndarray,
                   loop_indices: List[int]) -> Optional[List[Tuple[int, int, int]]]:
    if len(loop_indices) < 3:
        return None
    work = list(range(len(loop_indices)))
    if _polygon_area_2d(loop_2d) < 0:
        work.reverse()
    triangles: List[Tuple[int, int, int]] = []
    safety = len(work) * len(work)
    while len(work) > 3 and safety > 0:
        safety -= 1
        ear_found = False
        for i in range(len(work)):
            i0 = work[(i - 1) % len(work)]
            i1 = work[i]
            i2 = work[(i + 1) % len(work)]
            a = loop_2d[i0]
            b = loop_2d[i1]
            c = loop_2d[i2]
            if _cross2d(a, b, c) <= 1e-12:
                continue
            contains_vertex = False
            for j in work:
                if j in (i0, i1, i2):
                    continue
                if _point_in_triangle_2d(loop_2d[j], a, b, c):
                    contains_vertex = True
                    break
            if contains_vertex:
                continue
            triangles.append((loop_indices[i0], loop_indices[i1], loop_indices[i2]))
            work.pop(i)
            ear_found = True
            break
        if not ear_found:
            return None
    if len(work) == 3:
        triangles.append((loop_indices[work[0]], loop_indices[work[1]], loop_indices[work[2]]))
    return triangles if len(triangles) == len(loop_indices) - 2 else None


def _delaunay_fill(loop_2d: np.ndarray, loop_indices: List[int]) -> Optional[List[Tuple[int, int, int]]]:
    """
    Attempts to fill the planar 2D loop using scipy's Delaunay triangulation.
    If the hole is concave and unconstrained Delaunay fails to respect boundaries,
    this returns None, allowing fallback to Ear Clipping.
    """
    n = len(loop_2d)
    if n < 3:
        return None

    try:
        from scipy.spatial import Delaunay
    except ImportError:
        return None

    try:
        tri = Delaunay(loop_2d)
        valid_tris = []

        for simplex in tri.simplices:
            p0, p1, p2 = loop_2d[simplex[0]], loop_2d[simplex[1]], loop_2d[simplex[2]]
            # Cross product area check to prevent degenerate triangles
            cross = _cross2d(p0, p1, p2)
            if abs(cross) <= 1e-12:
                continue

            centroid = (p0 + p1 + p2) / 3.0
            x, y = centroid[0], centroid[1]
            inside = False

            # Fast Ray-Casting Point-in-Polygon test
            for i in range(n):
                p_curr = loop_2d[i]
                p_next = loop_2d[(i + 1) % n]
                if (p_curr[1] > y) != (p_next[1] > y):
                    x_intersect = (p_next[0] - p_curr[0]) * (y - p_curr[1]) / (p_next[1] - p_curr[1] + 1e-15) + p_curr[0]
                    if x < x_intersect:
                        inside = not inside

            if inside:
                # Ensure CCW orientation
                if cross < 0:
                    valid_tris.append((loop_indices[simplex[0]], loop_indices[simplex[2]], loop_indices[simplex[1]]))
                else:
                    valid_tris.append((loop_indices[simplex[0]], loop_indices[simplex[1]], loop_indices[simplex[2]]))

        # A simple polygon of N vertices is triangulated by exactly N-2 triangles
        if len(valid_tris) == n - 2:
            return valid_tris

    except Exception:
        pass

    return None


def _hole_normal(loop: List[Tuple], vertices: np.ndarray,
                 mesh_faces: np.ndarray, edge_map: Dict) -> np.ndarray:
    adj_normals = []
    for a, b in loop:
        key = (min(a, b), max(a, b))
        for fi in edge_map.get(key, []):
            face = mesh_faces[fi]
            n    = _face_normal(vertices, face)
            adj_normals.append(n)
    if not adj_normals:
        return np.array([0., 0., 1.])
    avg = np.mean(adj_normals, axis=0)
    ln  = np.linalg.norm(avg)
    return -avg / ln if ln > 1e-12 else np.array([0., 0., 1.])


def _newell_normal(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    points=points-points[0]
    shifted = np.roll(points, -1, axis=0)
    nx = np.sum((points[:, 1] - shifted[:, 1]) * (points[:, 2] + shifted[:, 2]))
    ny = np.sum((points[:, 2] - shifted[:, 2]) * (points[:, 0] + shifted[:, 0]))
    nz = np.sum((points[:, 0] - shifted[:, 0]) * (points[:, 1] + shifted[:, 1]))
    normal = np.array([nx, ny, nz], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    return normal / norm if norm > 0 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _hole_reference_normal(loop_vertices: List[int],
                           vertices: np.ndarray,
                           faces: np.ndarray,
                           edge_map: Dict) -> np.ndarray:
    acc = np.zeros(3, dtype=np.float64)
    count = 0
    for i, a in enumerate(loop_vertices):
        b = loop_vertices[(i + 1) % len(loop_vertices)]
        edge = (a, b) if a < b else (b, a)
        for fi in edge_map.get(edge, []):
            acc += _face_normal(vertices, faces[fi])
            count += 1
    if count > 0:
        acc /= float(count)
    if np.linalg.norm(acc) < 1e-12:
        pts = vertices[np.array(loop_vertices, dtype=np.int32)]
        acc = _newell_normal(pts)
    norm = float(np.linalg.norm(acc))
    return -acc / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _orient_patch_faces(vertices: np.ndarray,
                        patch_faces: List[np.ndarray],
                        hole_normal: np.ndarray) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for face in patch_faces:
        face_arr = np.array(face, dtype=np.int32)
        if _face_area(vertices, face_arr) < 1e-15:
            continue
        normal = _face_normal(vertices, face_arr)
        if float(np.dot(normal, hole_normal)) < 0:
            face_arr[[1, 2]] = face_arr[[2, 1]]
        out.append(face_arr)
    return out


def _orient_patch_faces_to_boundary(
    patch_faces: List[np.ndarray],
    existing_faces: np.ndarray,
    edge_map: Dict[Tuple[int, int], List[int]],
) -> List[np.ndarray]:
    """Orient patch triangles to agree with existing boundary winding."""
    oriented: List[np.ndarray] = []

    def edge_dir(face: np.ndarray, u: int, v: int) -> int:
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            if a == u and b == v:
                return 1
            if a == v and b == u:
                return -1
        return 0

    def conflict_score(face: np.ndarray) -> int:
        score = 0
        for i in range(3):
            u, v = int(face[i]), int(face[(i + 1) % 3])
            key = (u, v) if u < v else (v, u)
            adjacent = edge_map.get(key, [])
            if len(adjacent) != 1:
                continue
            patch_dir = edge_dir(face, u, v)
            old_dir = edge_dir(existing_faces[int(adjacent[0])], u, v)
            if patch_dir != 0 and old_dir != 0 and patch_dir == old_dir:
                score += 1
        return score

    for face in patch_faces:
        face_arr = np.array(face, dtype=np.int32)
        flipped = face_arr.copy()
        flipped[[1, 2]] = flipped[[2, 1]]
        oriented.append(flipped if conflict_score(flipped) < conflict_score(face_arr) else face_arr)
    return oriented


# ─────────────────────────────────────────────────────────────────────────────
#  Patch Fillers
# ─────────────────────────────────────────────────────────────────────────────

def _advancing_front_patch(
    loop_verts: List[int],
    vertices: np.ndarray,
    edge_map: Dict,
    mesh_faces: np.ndarray
) -> Tuple[np.ndarray, List[np.ndarray]]:
    pts       = vertices[loop_verts]
    centroid  = pts.mean(axis=0)
    new_v_idx = len(vertices)

    verts_out    = np.vstack([vertices, centroid[np.newaxis]])
    new_faces    = []

    hole_normal_acc = np.zeros(3)
    n_adj = 0
    for i, a in enumerate(loop_verts):
        b = loop_verts[(i + 1) % len(loop_verts)]
        key = (min(a, b), max(a, b))
        for fi in edge_map.get(key, []):
            hole_normal_acc += _face_normal(vertices, mesh_faces[fi])
            n_adj += 1

    if n_adj > 0:
        hole_dir = -hole_normal_acc / n_adj
    else:
        hole_dir = np.array([0., 0., 1.])

    for i in range(len(loop_verts)):
        a = loop_verts[i]
        b = loop_verts[(i + 1) % len(loop_verts)]
        face = np.array([a, b, new_v_idx], dtype=np.int32)
        n    = _face_normal(verts_out, face)
        if float(np.dot(n, hole_dir)) < 0:
            face[[1, 2]] = face[[2, 1]]
        new_faces.append(face)

    return verts_out, new_faces


def _centroid_fan_patch(loop_vertices: List[int],
                        vertices: np.ndarray,
                        hole_normal: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    pts = vertices[np.array(loop_vertices, dtype=np.int32)]
    # Use median for centroid to be more robust against outliers/spikes
    centroid = np.median(pts, axis=0)

    plane_normal = hole_normal / (np.linalg.norm(hole_normal) + 1e-15)
    # Project centroid onto the "average plane" of the boundary
    dists = np.dot(pts - centroid, plane_normal)
    centroid = centroid + np.mean(dists) * plane_normal

    new_vertices = np.vstack([vertices, centroid[np.newaxis, :]])
    center_idx = len(vertices)
    patch = [
        np.array([loop_vertices[i], loop_vertices[(i + 1) % len(loop_vertices)], center_idx],
                 dtype=np.int32)
        for i in range(len(loop_vertices))
    ]
    return new_vertices, _orient_patch_faces(new_vertices, patch, hole_normal)


def _stitch_collinear_loop(
    loop_vertices: List[int],
    vertices: np.ndarray,
    faces: np.ndarray
) -> Tuple[np.ndarray, bool]:
    if len(loop_vertices) < 3:
        return faces, False

    pts = vertices[np.array(loop_vertices, dtype=np.int32)]
    centered = pts - pts.mean(axis=0)
    _, svals, _ = np.linalg.svd(centered, full_matrices=False)
    if len(svals) < 2 or float(svals[1]) > 1e-8 * max(float(svals[0]), 1e-15):
        return faces, False

    edges = [(loop_vertices[i], loop_vertices[(i + 1) % len(loop_vertices)])
             for i in range(len(loop_vertices))]
    lengths_sq = [
        float(np.dot(vertices[b] - vertices[a], vertices[b] - vertices[a]))
        for a, b in edges
    ]
    if not lengths_sq:
        return faces, False

    chord_idx = int(np.argmax(lengths_sq))
    chord_a, chord_b = edges[chord_idx]
    chain = [chord_b]
    j = (chord_idx + 1) % len(loop_vertices)
    while True:
        chain.append(loop_vertices[(j + 1) % len(loop_vertices)])
        if loop_vertices[(j + 1) % len(loop_vertices)] == chord_a:
            break
        j = (j + 1) % len(loop_vertices)
        if len(chain) > len(loop_vertices) + 1:
            return faces, False

    ordered_chain = chain
    if len(ordered_chain) < 3:
        return faces, False

    edge_map = build_edge_map(faces)
    chord_key = (min(chord_a, chord_b), max(chord_a, chord_b))
    adjacent = edge_map.get(chord_key, [])
    if len(adjacent) != 1:
        return faces, False

    fi = int(adjacent[0])
    face = faces[fi]
    third_candidates = [int(v) for v in face if int(v) not in chord_key]
    if len(third_candidates) != 1:
        return faces, False
    third = third_candidates[0]

    ref_n = _face_normal(vertices, face)
    new_tris = []
    for i in range(len(ordered_chain) - 1):
        tri = np.array([third, ordered_chain[i], ordered_chain[i + 1]], dtype=np.int32)
        if _face_area(vertices, tri) < 1e-15:
            return faces, False
        tri_n = _face_normal(vertices, tri)
        if float(np.dot(ref_n, tri_n)) < 0:
            tri[[1, 2]] = tri[[2, 1]]
        new_tris.append(tri)

    canonical_existing: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for idx, existing_face in enumerate(faces):
        canonical_existing[tuple(sorted(int(v) for v in existing_face))].append(idx)

    remove_faces: Set[int] = {fi}
    add_tris: List[np.ndarray] = []
    for tri in new_tris:
        key = tuple(sorted(int(v) for v in tri))
        matches = [idx for idx in canonical_existing.get(key, []) if idx != fi]
        if matches:
            remove_faces.update(matches)
        else:
            add_tris.append(tri)

    keep_mask = np.ones(len(faces), dtype=bool)
    for idx in remove_faces:
        keep_mask[idx] = False

    stitched = faces[keep_mask]
    if add_tris:
        stitched = np.vstack([stitched, np.array(add_tris, dtype=np.int32)])
    return stitched, True


def improved_hole_fill(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    config: RepairConfig,
    log: RepairLog
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    config.check_deadline()
    edge_map = build_edge_map(faces)
    naked, _ = check_naked_and_nonmanifold(edge_map)
    if not naked:
        return vertices, faces, normals

    naked_edge_set = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in naked}

    vertices, cleaned_loops = _build_clean_boundary_loops(
        vertices, faces, naked, edge_map, config
    )
    closed = [
        {"vertices": loop, "closed": True, "edges": len(loop)}
        for loop in cleaned_loops
        if _loop_edges_are_present(loop, naked_edge_set)
    ]
    if not closed:
        boundary_sets = robust_boundary_loop(naked, vertices)
        closed = []
        for item in boundary_sets:
            if not item["closed"] or len(item["vertices"]) < 3:
                continue
            for cycle in _split_repeated_vertex_cycles(list(item["vertices"])):
                if _loop_edges_are_present(cycle, naked_edge_set):
                    closed.append({"vertices": cycle, "closed": True, "edges": len(cycle)})
    if not closed:
        return vertices, faces, normals

    stitched = 0
    stitched_face_delta = 0
    remaining_closed = []
    for item in closed:
        loop_vertices = _clean_loop_vertices(list(item["vertices"]))
        updated_faces, fixed = _stitch_collinear_loop(loop_vertices, vertices, faces)
        if fixed:
            stitched += 1
            stitched_face_delta += len(updated_faces) - len(faces)
            faces = updated_faces
        else:
            remaining_closed.append(item)

    if stitched:
        normals = _recompute_normals(vertices, faces)
        log.planar_holes_filled += stitched
        log.faces_added += stitched_face_delta

    if not remaining_closed:
        return vertices, faces, normals

    edge_map = build_edge_map(faces)
    edge_usage={edge:len(ids) for edge,ids in edge_map.items()}
    existing_face_keys={tuple(sorted(map(int,face))) for face in faces}
    planar_filled = 0
    fan_filled = 0

    for item in remaining_closed:
        config.check_deadline()
        previous_vertex_count=len(vertices)
        loop_vertices = _clean_loop_vertices(list(item["vertices"]))
        if len(loop_vertices) < 3 or len(set(loop_vertices)) < 3:
            continue
        if len(loop_vertices)>config.max_hole_edges:
            continue
        fallback_loop_vertices = list(loop_vertices)
        pts_3d = vertices[np.array(loop_vertices, dtype=np.int32)]
        edge_lengths = np.linalg.norm(pts_3d - np.roll(pts_3d, -1, axis=0), axis=1)
        if np.any(edge_lengths <= 1e-15):
            continue

        hole_normal = _hole_reference_normal(loop_vertices, vertices, faces, edge_map)
        centered = pts_3d-pts_3d[0]
        centered = centered-centered.mean(axis=0)
        _, svals, _ = np.linalg.svd(centered, full_matrices=False)
        planar_ratio = float(svals[-1] / max(svals[0], 1e-15))

        patch: Optional[List[np.ndarray]] = None
        patch_delta: Optional[Tuple[Dict[Tuple[int, int], int], Set[Tuple[int, int, int]]]] = None
        used_fallback = True
        if planar_ratio <= config.planar_tolerance:
            pts_2d, _, _ = _project_to_2d(pts_3d, hole_normal)
            if abs(_polygon_area_2d(pts_2d)) <= 1e-15:
                pts_2d = None
            else:
                keep = []
                n2d = len(pts_2d)
                for i in range(n2d):
                    prev_p = pts_2d[(i - 1) % n2d]
                    curr_p = pts_2d[i]
                    next_p = pts_2d[(i + 1) % n2d]
                    if abs(_cross2d(prev_p, curr_p, next_p)) > 1e-14:
                        keep.append(i)
                if len(keep) >= 3 and len(keep) < len(loop_vertices):
                    kept_loop_vertices = [loop_vertices[i] for i in keep]
                    if _loop_edges_are_present(kept_loop_vertices, naked_edge_set):
                        loop_vertices = kept_loop_vertices
                        pts_3d = pts_3d[np.array(keep, dtype=np.int32)]
                        pts_2d = pts_2d[np.array(keep, dtype=np.int32)]
                elif len(keep) < 3:
                    pts_2d = None
            if pts_2d is not None:
                tris = _delaunay_fill(pts_2d, loop_vertices)
                if tris is not None:
                    patch = _orient_patch_faces(
                        vertices,
                        [np.array(tri, dtype=np.int32) for tri in tris],
                        hole_normal
                    )
                    if len(patch) == len(loop_vertices) - 2:
                        patch_delta = _patch_delta_if_safe(
                            patch, edge_usage, existing_face_keys, config.max_faces_per_edge
                        )
                        if patch_delta is not None:
                            used_fallback = False
                        else:
                            patch = None
                    else:
                        patch = None

                if patch is None:
                    tris = _safe_ear_clip(pts_2d, loop_vertices)
                    if tris is not None:
                        patch = _orient_patch_faces(
                            vertices,
                            [np.array(tri, dtype=np.int32) for tri in tris],
                            hole_normal
                        )
                        if len(patch) == len(loop_vertices) - 2:
                            patch_delta = _patch_delta_if_safe(
                                patch, edge_usage, existing_face_keys, config.max_faces_per_edge
                            )
                            if patch_delta is not None:
                                used_fallback = False
                            else:
                                patch = None
                        else:
                            patch = None

                # Advancing-front fallback after ear clipping.
                if patch is None:
                    candidate_vertices, candidate_patch = _advancing_front_patch(
                        loop_vertices, vertices, edge_map, faces
                    )
                    candidate_patch = _orient_patch_faces(candidate_vertices, candidate_patch, hole_normal)
                    candidate_delta = _patch_delta_if_safe(
                        candidate_patch, edge_usage, existing_face_keys, config.max_faces_per_edge
                    )
                    if candidate_delta is not None:
                        vertices = candidate_vertices
                        patch = candidate_patch
                        patch_delta = candidate_delta
                        used_fallback = True

                if patch is None:
                    tris = _safe_ear_clip(pts_2d, loop_vertices)
                    if tris is not None:
                        patch = _orient_patch_faces(
                            vertices,
                            [np.array(tri, dtype=np.int32) for tri in tris],
                            hole_normal
                        )
                        if len(patch) == len(loop_vertices) - 2:
                            patch_delta = _patch_delta_if_safe(
                                patch, edge_usage, existing_face_keys, config.max_faces_per_edge
                            )
                            if patch_delta is not None:
                                used_fallback = False
                            else:
                                patch = None
                        else:
                            patch = None
            else:
                patch = None

        if patch is None:
            candidate_vertices, candidate_patch = _centroid_fan_patch(
                fallback_loop_vertices, vertices, hole_normal
            )
            patch_delta = _patch_delta_if_safe(
                candidate_patch, edge_usage, existing_face_keys, config.max_faces_per_edge
            )
            if patch_delta is None:
                continue
            vertices = candidate_vertices
            patch = candidate_patch
            used_fallback = True

        if not patch or patch_delta is None:
            continue

        patch = _orient_patch_faces_to_boundary(patch, faces, edge_map)

        # Validate local incidence and directed edges before committing. Global
        # geometry is independently checked at candidate acceptance. Rebuilding
        # the entire mesh for every hole was quadratic in holes * faces.
        directions=defaultdict(list)
        for face in patch:
            for k in range(3):
                u,v=int(face[k]),int(face[(k+1)%3])
                directions[(min(u,v),max(u,v))].append(1 if u<v else -1)
        safe=True; naked_delta=0
        for key,added_directions in directions.items():
            old_count=edge_usage.get(key,0)
            new_count=old_count+len(added_directions)
            naked_delta+=int(new_count==1)-int(old_count==1)
            if new_count==2:
                combined=list(added_directions)
                if old_count==1:
                    face=faces[edge_map[key][0]]
                    for k in range(3):
                        u,v=int(face[k]),int(face[(k+1)%3])
                        if (min(u,v),max(u,v))==key:
                            combined.append(1 if u<v else -1)
                            break
                if len(combined)==2 and combined[0]==combined[1]:
                    safe=False; break
        patch_array=np.array(patch,dtype=np.int64)
        triangle_points=vertices[patch_array]
        cross=np.cross(triangle_points[:,1]-triangle_points[:,0],triangle_points[:,2]-triangle_points[:,0])
        if np.any(np.linalg.norm(cross,axis=1)<=config.area_tolerance*2):
            safe=False
        max_edge=float(np.linalg.norm(np.ptp(pts_3d,axis=0)))*(1+1e-12)
        for face in patch:
            points=vertices[face]
            if max(np.linalg.norm(points[k]-points[(k+1)%3]) for k in range(3))>max_edge:
                safe=False; break
        if not safe or naked_delta>0:
            vertices=vertices[:previous_vertex_count]
            continue
        start_face=len(faces)
        faces=np.vstack([faces,patch_array])
        normals=np.vstack([normals,_recompute_normals(vertices,patch_array)])
        edge_delta,face_delta=patch_delta
        existing_face_keys.update(face_delta)
        for edge,count in edge_delta.items():
            edge_usage[edge]=edge_usage.get(edge,0)+count
        for offset,face in enumerate(patch_array):
            for k in range(3):
                u,v=int(face[k]),int(face[(k+1)%3])
                edge_map.setdefault((min(u,v),max(u,v)),[]).append(start_face+offset)
        log.faces_added += len(patch)
        if used_fallback:
            fan_filled += 1
        else:
            planar_filled += 1

    log.planar_holes_filled += planar_filled
    log.non_planar_filled += fan_filled
    return vertices, faces, normals


def repair_planar_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    config: RepairConfig,
    log: RepairLog
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Run the hole filling algorithm up to 2 times to handle complex holes
    # that may require multiple passes to fully close.
    for _ in range(4):
        prev_added = log.faces_added
        vertices, faces, normals = improved_hole_fill(vertices, faces, normals, config, log)
        if log.faces_added == prev_added:
            break
    return vertices, faces, normals


def repair_non_planar_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    config: RepairConfig,
    log: RepairLog
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # All holes are handled inside improved_hole_fill now
    return vertices, faces, normals



# ─────────────────────────────────────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def cli_repair_holes(input_path: str, output_path: str):
    """
    Standard CLI-style wrapper for hole repair.
    Loads input model, fills holes, and writes output.
    """
    from src.core.mesh_analyzer import MeshAnalyzer
    import struct
    import sys

    # 1. Load
    print(f"  [CLI] Loading: {input_path}")
    analyzer = MeshAnalyzer()
    mesh, _ = analyzer.analyze(input_path)

    # 2. Repair
    import os
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "repair_config.ini")
    if not os.path.exists(config_path):
        config_path = "repair_config.ini"
    config = RepairConfig.from_ini(config_path)
    log = RepairLog()
    v, f, n = repair_planar_holes(mesh.vertices, mesh.faces, mesh.normals, config, log)

    print(f"  [CLI] Planar holes filled: {log.planar_holes_filled}")
    print(f"  [CLI] Non-planar holes filled: {log.non_planar_filled}")
    print(f"  [CLI] Total faces added: {log.faces_added}")

    # 3. Save (inline binary STL writer)
    print(f"  [CLI] Saving to: {output_path}")
    recomputed = _recompute_normals(v, f)
    with open(output_path, "wb") as out:
        hdr = b"Hole Repair CLI output"
        out.write((hdr + b"\x00" * 80)[:80])
        out.write(struct.pack("<I", len(f)))
        for fi, face in enumerate(f):
            out.write(struct.pack("<3f", *recomputed[fi]))
            for vi in face:
                out.write(struct.pack("<3f", *v[vi]))
            out.write(struct.pack("<H", 0))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python repair_holes.py <input.stl> <output.stl>")
        sys.exit(1)
    cli_repair_holes(sys.argv[1], sys.argv[2])
