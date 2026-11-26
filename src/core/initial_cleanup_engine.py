"""
Initial cleanup mesh repair engine ported from C to Python/NumPy.
Implements hash-based edge matching, neighbor-fan vertex snapping,
simple hole filling, and BFS normal fixing.
"""
import struct
import re
import collections
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

HASH_M = 81397

@dataclass
class StlStats:
    number_of_facets: int = 0
    original_num_facets: int = 0
    min: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    max: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    size: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    bounding_diameter: float = 0.0
    shortest_edge: float = float('inf')
    volume: float = 0.0
    connected_edges: int = 0
    connected_facets_1_edge: int = 0
    connected_facets_2_edge: int = 0
    connected_facets_3_edge: int = 0
    facets_w_1_bad_edge: int = 0
    facets_w_2_bad_edge: int = 0
    facets_w_3_bad_edge: int = 0
    edges_fixed: int = 0
    degenerate_facets: int = 0
    facets_removed: int = 0
    facets_added: int = 0
    facets_reversed: int = 0
    backwards_edges: int = 0
    normals_fixed: int = 0
    number_of_parts: int = 0
    facets_malloced: int = 0

def _is_ascii_stl(path: str) -> bool:
    with open(path, 'rb') as f:
        head = f.read(256)
    try:
        text = head.decode('ascii', errors='ignore').lower()
        return text.strip().startswith('solid') and 'facet' in text
    except Exception:
        return False

def _load_ascii_stl(path: str):
    with open(path, 'r', errors='ignore') as f:
        content = f.read()
    facet_blocks = re.findall(
        r'facet\s+normal\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+'
        r'outer\s+loop\s+'
        r'vertex\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+'
        r'vertex\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+'
        r'vertex\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+'
        r'endloop\s+endfacet',
        content, re.IGNORECASE)
    n = len(facet_blocks)
    normals = np.zeros((n, 3), dtype=np.float32)
    verts = np.zeros((n, 3, 3), dtype=np.float32)
    for i, m in enumerate(facet_blocks):
        vals = [float(x) for x in m]
        normals[i] = vals[0:3]
        verts[i, 0] = vals[3:6]
        verts[i, 1] = vals[6:9]
        verts[i, 2] = vals[9:12]
    return verts, normals

class StlFile:
    def __init__(self):
        self.facets = np.zeros((0, 3, 3), dtype=np.float32)
        self.normals = np.zeros((0, 3), dtype=np.float32)
        self.neighbors = np.zeros((0, 3), dtype=np.int32)
        self.which_vertex_not = np.zeros((0, 3), dtype=np.int8)
        self.stats = StlStats()

    @staticmethod
    def from_file(path: str) -> 'StlFile':
        stl = StlFile()
        if _is_ascii_stl(path):
            verts, normals = _load_ascii_stl(path)
        else:
            with open(path, 'rb') as f:
                f.read(80)  # header
                count_data = f.read(4)
                if len(count_data) < 4:
                    return stl
                n = struct.unpack('<I', count_data)[0]
                if n == 0:
                    return stl
                data = f.read(50 * n)
                if len(data) < 50 * n:
                    n = len(data) // 50
                raw = np.frombuffer(data[:50*n], dtype=np.uint8).reshape(n, 50)
                normals = np.frombuffer(raw[:, :12].tobytes(), dtype=np.float32).reshape(n, 3)
                verts = np.frombuffer(raw[:, 12:48].tobytes(), dtype=np.float32).reshape(n, 3, 3)
        n = len(verts)
        stl.facets = verts.copy()
        stl.normals = normals.copy()
        stl.stats.number_of_facets = n
        stl.stats.original_num_facets = n
        stl.stats.facets_malloced = n
        stl.neighbors = np.full((n, 3), -1, dtype=np.int32)
        stl.which_vertex_not = np.zeros((n, 3), dtype=np.int8)
        _get_size(stl)
        stl.stats.shortest_edge = float('inf')
        return stl

    def write_binary(self, path: str):
        with open(path, 'wb') as f:
            hdr = b'Repaired by Initial Cleanup engine'
            f.write((hdr + b'\x00' * 80)[:80])
            n = self.stats.number_of_facets
            f.write(struct.pack('<I', n))
            for i in range(n):
                f.write(struct.pack('<3f', *self.normals[i]))
                for j in range(3):
                    f.write(struct.pack('<3f', *self.facets[i, j]))
                f.write(struct.pack('<H', 0))


def _get_size(stl: StlFile):
    if stl.stats.number_of_facets == 0:
        return
    n = stl.stats.number_of_facets
    all_v = stl.facets[:n].reshape(-1, 3)
    stl.stats.min = all_v.min(axis=0).copy()
    stl.stats.max = all_v.max(axis=0).copy()
    stl.stats.size = stl.stats.max - stl.stats.min
    stl.stats.bounding_diameter = float(np.linalg.norm(stl.stats.size))


# ── Edge hashing  ─────────────────────────

def _edge_key_exact(a: np.ndarray, b: np.ndarray) -> Tuple[bytes, int]:
    """Create canonical edge key from two vertices. Returns (key_bytes, direction_offset)."""
    diff = np.abs(a - b)
    max_diff = diff.max()
    if diff[0] == max_diff:
        if a[0] > b[0]:
            key = a.tobytes() + b.tobytes()
            return key, 0
        else:
            key = b.tobytes() + a.tobytes()
            return key, 3
    elif diff[1] == max_diff:
        if a[1] > b[1]:
            key = a.tobytes() + b.tobytes()
            return key, 0
        else:
            key = b.tobytes() + a.tobytes()
            return key, 3
    else:
        if a[2] > b[2]:
            key = a.tobytes() + b.tobytes()
            return key, 0
        else:
            key = b.tobytes() + a.tobytes()
            return key, 3


def _edge_key_nearby(a: np.ndarray, b: np.ndarray, mn: np.ndarray, tol: float):
    """Grid-quantized edge key for nearby matching."""
    va = ((a - mn) / tol).astype(np.uint32)
    vb = ((b - mn) / tol).astype(np.uint32)
    if np.array_equal(va, vb):
        return None, 0
    diff = np.abs(a - b)
    max_diff = diff.max()
    if diff[0] == max_diff:
        if a[0] > b[0]:
            key = va.tobytes() + vb.tobytes()
            return key, 0
        else:
            key = vb.tobytes() + va.tobytes()
            return key, 3
    elif diff[1] == max_diff:
        if a[1] > b[1]:
            key = va.tobytes() + vb.tobytes()
            return key, 0
        else:
            key = vb.tobytes() + va.tobytes()
            return key, 3
    else:
        if a[2] > b[2]:
            key = va.tobytes() + vb.tobytes()
            return key, 0
        else:
            key = vb.tobytes() + va.tobytes()
            return key, 3


def _record_neighbors(stl: StlFile, fi_a, edge_a, fi_b, edge_b):
    ea = edge_a % 3
    eb = edge_b % 3
    stl.neighbors[fi_a, ea] = fi_b
    stl.which_vertex_not[fi_a, ea] = (edge_b + 2) % 3
    stl.neighbors[fi_b, eb] = fi_a
    stl.which_vertex_not[fi_b, eb] = (edge_a + 2) % 3
    if (edge_a < 3 and edge_b < 3) or (edge_a > 2 and edge_b > 2):
        stl.which_vertex_not[fi_a, ea] += 3
        stl.which_vertex_not[fi_b, eb] += 3
    stl.stats.connected_edges += 2
    for fi in (fi_a, fi_b):
        uc = int(np.sum(stl.neighbors[fi] == -1))
        if uc == 2:
            stl.stats.connected_facets_1_edge += 1
        elif uc == 1:
            stl.stats.connected_facets_2_edge += 1
        elif uc == 0:
            stl.stats.connected_facets_3_edge += 1


# ── Phase 1: Exact edge check ──────────────────────────────────────

def check_facets_exact(stl: StlFile, verbose=False):
    if verbose:
        print("  Checking exact...")
    n = stl.stats.number_of_facets
    stl.stats.connected_edges = 0
    stl.stats.connected_facets_1_edge = 0
    stl.stats.connected_facets_2_edge = 0
    stl.stats.connected_facets_3_edge = 0
    stl.neighbors[:n] = -1
    stl.which_vertex_not[:n] = 0
    hash_table: Dict[bytes, list] = {}
    i = 0
    while i < stl.stats.number_of_facets:
        f = stl.facets[i]
        if (np.array_equal(f[0], f[1]) or np.array_equal(f[1], f[2])
                or np.array_equal(f[0], f[2])):
            stl.stats.degenerate_facets += 1
            _remove_facet(stl, i)
            continue
        for j in range(3):
            a, b = f[j], f[(j + 1) % 3]
            key, d_off = _edge_key_exact(a, b)
            diff = np.abs(a - b).max()
            stl.stats.shortest_edge = min(stl.stats.shortest_edge, float(diff))
            which_edge = j + d_off
            if key in hash_table:
                entry = hash_table[key]
                matched = False
                for idx, (ofi, oedge) in enumerate(entry):
                    if ofi != i:
                        _record_neighbors(stl, i, which_edge, ofi, oedge)
                        entry.pop(idx)
                        if not entry:
                            del hash_table[key]
                        matched = True
                        break
                if not matched:
                    entry.append((i, which_edge))
            else:
                hash_table[key] = [(i, which_edge)]
        i += 1


# ── Phase 2: Nearby edge check ─────────────────────────────────────

def check_facets_nearby(stl: StlFile, tolerance: float, verbose=False):
    n = stl.stats.number_of_facets
    if (stl.stats.connected_facets_1_edge == n and
        stl.stats.connected_facets_2_edge == n and
        stl.stats.connected_facets_3_edge == n):
        return
    if verbose:
        print(f"  Checking nearby (tol={tolerance:.6f})...")
    hash_table: Dict[bytes, list] = {}
    mn = stl.stats.min
    for i in range(n):
        f = stl.facets[i]
        for j in range(3):
            if stl.neighbors[i, j] != -1:
                continue
            a, b = f[j], f[(j + 1) % 3]
            key, d_off = _edge_key_nearby(a, b, mn, tolerance)
            if key is None:
                continue
            which_edge = j + d_off
            if key in hash_table:
                entry = hash_table[key]
                matched = False
                for idx, (ofi, oedge) in enumerate(entry):
                    if ofi != i:
                        _match_neighbors_nearby(stl, i, which_edge, ofi, oedge)
                        entry.pop(idx)
                        if not entry:
                            del hash_table[key]
                        matched = True
                        break
                if not matched:
                    entry.append((i, which_edge))
            else:
                hash_table[key] = [(i, which_edge)]


def _which_vertices_to_change(stl, fi_a, edge_a, fi_b, edge_b):
    """Port of stl_which_vertices_to_change from thesis.
    A vertex is changed only if BOTH adjacent edges of that vertex are unconnected.
    """
    if edge_a < 3:
        v1a, v2a = edge_a, (edge_a + 1) % 3
    else:
        v2a, v1a = edge_a % 3, (edge_a + 1) % 3
    if edge_b < 3:
        v1b, v2b = edge_b, (edge_b + 1) % 3
    else:
        v2b, v1b = edge_b % 3, (edge_b + 1) % 3
    results = []
    for (fa, va, fb, vb) in [(fi_a, v1a, fi_b, v1b), (fi_a, v2a, fi_b, v2b)]:
        if np.array_equal(stl.facets[fa, va], stl.facets[fb, vb]):
            results.append(None)  # already equal
        else:
            # Change the vertex that has both adjacent edges unconnected
            if (stl.neighbors[fa, va] == -1 and
                    stl.neighbors[fa, (va + 2) % 3] == -1):
                results.append((fa, va, stl.facets[fb, vb].copy()))
            else:
                results.append((fb, vb, stl.facets[fa, va].copy()))
    return results

def _change_vertices(stl, facet_num, vnot, new_vertex):
    """Port of stl_change_vertices: fan-walks all facets sharing this vertex."""
    first_facet = facet_num
    direction = 0
    while True:
        if vnot > 2:
            if direction == 0:
                pivot_vertex = (vnot + 2) % 3
                next_edge = pivot_vertex
                direction = 1
            else:
                pivot_vertex = (vnot + 1) % 3
                next_edge = vnot % 3
                direction = 0
        else:
            if direction == 0:
                pivot_vertex = (vnot + 1) % 3
                next_edge = vnot
            else:
                pivot_vertex = (vnot + 2) % 3
                next_edge = pivot_vertex
        stl.facets[facet_num, pivot_vertex] = new_vertex
        vnot = int(stl.which_vertex_not[facet_num, next_edge])
        facet_num = int(stl.neighbors[facet_num, next_edge])
        if facet_num == -1:
            break
        if facet_num == first_facet:
            break  # completed the fan loop

def _match_neighbors_nearby(stl, fi_a, edge_a, fi_b, edge_b):
    _record_neighbors(stl, fi_a, edge_a, fi_b, edge_b)
    changes = _which_vertices_to_change(stl, fi_a, edge_a, fi_b, edge_b)
    for ch in changes:
        if ch is None:
            continue
        facet_num, vertex_idx, new_vertex = ch
        if facet_num == fi_a:
            vnot = (edge_a + 2) % 3
        else:
            vnot = (edge_b + 2) % 3
        if ((vnot + 2) % 3) == vertex_idx:
            vnot += 3
        _change_vertices(stl, facet_num, vnot, new_vertex)
    stl.stats.edges_fixed += 2


# ── Remove facets ──────────────────────────────────────────────────

def _remove_facet(stl: StlFile, fi: int):
    """Faithful port of stl_remove_facet from thesis connect.c.
    Decrements connected-edge counters and fixes neighbor backlinks.
    """
    stl.stats.facets_removed += 1
    # Decrement connected-edge counters based on how many neighbors this facet has
    j = int(np.sum(stl.neighbors[fi] == -1))  # number of unconnected edges
    if j == 2:    # had 1 connected edge
        stl.stats.connected_facets_1_edge -= 1
    elif j == 1:  # had 2 connected edges
        stl.stats.connected_facets_2_edge -= 1
        stl.stats.connected_facets_1_edge -= 1
    elif j == 0:  # had 3 connected edges (fully connected)
        stl.stats.connected_facets_3_edge -= 1
        stl.stats.connected_facets_2_edge -= 1
        stl.stats.connected_facets_1_edge -= 1
    last = stl.stats.number_of_facets - 1
    # Move last facet into the removed slot
    if fi != last:
        stl.facets[fi] = stl.facets[last]
        stl.normals[fi] = stl.normals[last]
        stl.neighbors[fi] = stl.neighbors[last]
        stl.which_vertex_not[fi] = stl.which_vertex_not[last]
        # Fix backlinks: neighbors of the moved facet that pointed to 'last' now point to 'fi'
        for j in range(3):
            nb = int(stl.neighbors[fi, j])
            if nb != -1 and nb < last:
                vnot = int(stl.which_vertex_not[fi, j])
                idx = (vnot + 1) % 3
                if stl.neighbors[nb, idx] == last:
                    stl.neighbors[nb, idx] = fi
    stl.stats.number_of_facets -= 1


def remove_unconnected_facets(stl: StlFile, verbose=False):
    if verbose:
        print("  Removing unconnected facets...")
    # Remove degenerates
    i = 0
    while i < stl.stats.number_of_facets:
        f = stl.facets[i]
        if (np.array_equal(f[0], f[1]) or np.array_equal(f[1], f[2])
                or np.array_equal(f[0], f[2])):
            _remove_facet(stl, i)
            continue
        i += 1
    # Remove fully unconnected
    if stl.stats.connected_facets_1_edge < stl.stats.number_of_facets:
        i = 0
        while i < stl.stats.number_of_facets:
            if np.all(stl.neighbors[i] == -1):
                _remove_facet(stl, i)
                continue
            i += 1


# ── Hole filling ───────────────────────────────────────────────────

def fill_holes(stl: StlFile, verbose=False):
    """Faithful port of stl_fill_holes from thesis connect.c.
    Uses neighbor-fan walk to find the third vertex of each gap triangle.
    """
    if stl.stats.connected_facets_3_edge >= stl.stats.number_of_facets:
        if verbose:
            print("  No holes to fill.")
        return
    if verbose:
        print("  Filling holes...")
    # Build hash table of all unconnected edges (exact matching)
    hash_table: Dict[bytes, list] = {}
    for i in range(stl.stats.number_of_facets):
        for j in range(3):
            if stl.neighbors[i, j] != -1:
                continue
            a = stl.facets[i, j]
            b = stl.facets[i, (j + 1) % 3]
            key, d_off = _edge_key_exact(a, b)
            which_edge = j + d_off
            if key in hash_table:
                entry = hash_table[key]
                matched = False
                for idx, (ofi, oedge) in enumerate(entry):
                    if ofi != i:
                        _record_neighbors(stl, i, which_edge, ofi, oedge)
                        entry.pop(idx)
                        if not entry:
                            del hash_table[key]
                        matched = True
                        break
                if not matched:
                    entry.append((i, which_edge))
            else:
                hash_table[key] = [(i, which_edge)]

    # Fan-walk to find third vertex for each remaining unconnected edge.
    # NOTE: use a live index (not a frozen original_n snapshot) so that
    # newly-added patch facets are also visited — matching C behaviour.
    i = 0
    while i < stl.stats.number_of_facets:
        neighbors_initial = stl.neighbors[i].copy()
        for j in range(3):
            if stl.neighbors[i, j] != -1:
                continue
            # Start building new_facet
            new_v0 = stl.facets[i, j].copy()
            new_v1 = stl.facets[i, (j + 1) % 3].copy()
            # Determine initial direction
            if neighbors_initial[(j + 2) % 3] == -1:
                direction = 1
            else:
                direction = 0
            facet_num = i
            vnot = (j + 2) % 3
            first_facet = i
            # Fan-walk to find the third vertex.
            # Guard: step count prevents infinite loops on malformed meshes.
            max_steps = stl.stats.number_of_facets * 4 + 8
            steps = 0
            while steps < max_steps:
                steps += 1
                if vnot > 2:
                    if direction == 0:
                        pivot_vertex = (vnot + 2) % 3
                        next_edge = pivot_vertex
                        direction = 1
                    else:
                        pivot_vertex = (vnot + 1) % 3
                        next_edge = vnot % 3
                        direction = 0
                else:
                    if direction == 0:
                        pivot_vertex = (vnot + 1) % 3
                        next_edge = vnot
                    else:
                        pivot_vertex = (vnot + 2) % 3
                        next_edge = pivot_vertex
                next_facet = int(stl.neighbors[facet_num, next_edge])
                if next_facet == -1:
                    # Found the boundary - third vertex is pivot_vertex of facet_num
                    new_v2 = stl.facets[facet_num, vnot % 3].copy()
                    new_facet = np.array([new_v0, new_v1, new_v2], dtype=np.float32)
                    _add_facet(stl, new_facet)
                    # Insert new facet's edges into hash for matching
                    new_fi = stl.stats.number_of_facets - 1
                    for k in range(3):
                        a = stl.facets[new_fi, k]
                        b = stl.facets[new_fi, (k + 1) % 3]
                        key, d_off = _edge_key_exact(a, b)
                        we = k + d_off
                        if key in hash_table:
                            entry = hash_table[key]
                            matched = False
                            for idx2, (ofi2, oedge2) in enumerate(entry):
                                if ofi2 != new_fi:
                                    _record_neighbors(stl, new_fi, we, ofi2, oedge2)
                                    entry.pop(idx2)
                                    if not entry:
                                        del hash_table[key]
                                    matched = True
                                    break
                            if not matched:
                                entry.append((new_fi, we))
                        else:
                            hash_table[key] = [(new_fi, we)]
                    break
                # Advance the fan-walk
                vnot = int(stl.which_vertex_not[facet_num, next_edge])
                facet_num = next_facet
                # Möbius guard (C-correct: check AFTER advancing)
                if facet_num == first_facet:
                    break

        i += 1




def _add_facet(stl: StlFile, new_facet: np.ndarray):
    n = stl.stats.number_of_facets
    if n >= len(stl.facets):
        extra = 256
        stl.facets = np.concatenate([stl.facets, np.zeros((extra, 3, 3), dtype=np.float32)])
        stl.normals = np.concatenate([stl.normals, np.zeros((extra, 3), dtype=np.float32)])
        stl.neighbors = np.concatenate([stl.neighbors, np.full((extra, 3), -1, dtype=np.int32)])
        stl.which_vertex_not = np.concatenate([stl.which_vertex_not, np.zeros((extra, 3), dtype=np.int8)])
        stl.stats.facets_malloced += extra
    stl.facets[n] = new_facet
    stl.normals[n] = 0.0
    stl.neighbors[n] = -1
    stl.which_vertex_not[n] = 0
    stl.stats.number_of_facets += 1
    stl.stats.facets_added += 1


# ── Normal direction fixing (BFS) ─────────────────────────────────

def fix_normal_directions(stl: StlFile, verbose=False):
    """Fast BFS normal fixer.

    Phase 1: Graph-coloring BFS — propagates a 'flip' flag through the
    neighbour graph using `which_vertex_not > 2` as the backwards-edge
    indicator.  No facet data is mutated during traversal.

    Phase 2: Batch-flip all marked facets in one vectorised NumPy pass,
    then update `which_vertex_not` for their neighbours in bulk.

    This replaces the original O(n²) approach (calling `_reverse_facet`
    for every visited facet, each of which re-walked 3 neighbour entries
    individually in Python).
    """
    if verbose:
        print("  Fixing normal directions...")
    n = stl.stats.number_of_facets
    if n == 0:
        return

    # ── Phase 1: BFS orientation labelling ──────────────────────────
    # flip[i] = True  →  facet i must be reversed
    # fixed[i] = True →  already assigned an orientation
    flip  = np.zeros(n, dtype=np.bool_)
    fixed = np.zeros(n, dtype=np.bool_)
    num_parts = 0

    # seed: check facet 0 against its stored normal
    if _check_normal_vector(stl, 0) == 2:
        flip[0] = True
    fixed[0] = True
    checked = 1

    queue = collections.deque([0])
    while True:
        while queue:
            fi = queue.popleft()
            for j in range(3):
                nb = int(stl.neighbors[fi, j])
                if nb < 0 or nb >= n:
                    continue
                # which_vertex_not > 2 means winding is opposite → propagate inverted flip
                backwards = (int(stl.which_vertex_not[fi, j]) > 2)
                nb_flip = flip[fi] ^ backwards
                if not fixed[nb]:
                    flip[nb]  = nb_flip
                    fixed[nb] = True
                    checked  += 1
                    queue.append(nb)

        # finished a connected component
        num_parts += 1
        if checked >= n:
            break
        # find the next unvisited seed
        seeds = np.where(~fixed)[0]
        if len(seeds) == 0:
            break
        seed = int(seeds[0])
        if _check_normal_vector(stl, seed) == 2:
            flip[seed] = True
        fixed[seed] = True
        checked += 1
        queue.append(seed)

    stl.stats.number_of_parts += num_parts

    # ── Phase 2: Batch-flip all marked facets ────────────────────────
    to_flip = np.where(flip)[0]
    if len(to_flip) == 0:
        return

    stl.stats.facets_reversed += len(to_flip)

    # Swap vertex 0 ↔ vertex 1 for every marked facet (reverses winding)
    tmp = stl.facets[to_flip, 0].copy()
    stl.facets[to_flip, 0] = stl.facets[to_flip, 1]
    stl.facets[to_flip, 1] = tmp

    # Update which_vertex_not for neighbours of flipped facets.
    # The original per-facet offsets from the thesis: nb[0]→+3, nb[1]→+4, nb[2]→+2
    offsets = np.array([3, 4, 2], dtype=np.int32)
    vnot_snap = stl.which_vertex_not[to_flip]          # (m, 3) int8
    nb_snap   = stl.neighbors[to_flip]                  # (m, 3) int32
    for j in range(3):
        nbs  = nb_snap[:, j]                            # neighbour index for edge j
        vnot = vnot_snap[:, j].astype(np.int32)         # which_vertex_not for edge j
        valid = (nbs >= 0) & (nbs < n)
        nbs_v   = nbs[valid]
        vnot_v  = vnot[valid]
        idx_col = (vnot_v + 1) % 3                      # column to update in neighbour
        old = stl.which_vertex_not[nbs_v, idx_col].astype(np.int32)
        stl.which_vertex_not[nbs_v, idx_col] = ((old + offsets[j]) % 6).astype(np.int8)

    # Swap neighbour entries 1↔2 and their vnots for every flipped facet
    nb1 = stl.neighbors[to_flip, 1].copy()
    nb2 = stl.neighbors[to_flip, 2].copy()
    stl.neighbors[to_flip, 1] = nb2
    stl.neighbors[to_flip, 2] = nb1

    vn1 = stl.which_vertex_not[to_flip, 1].copy()
    vn2 = stl.which_vertex_not[to_flip, 2].copy()
    stl.which_vertex_not[to_flip, 1] = vn2
    stl.which_vertex_not[to_flip, 2] = vn1

    # Add 3 (mod 6) to all three vnot values of every flipped facet
    stl.which_vertex_not[to_flip] = (
        stl.which_vertex_not[to_flip].astype(np.int32) + 3
    ).astype(np.int8) % 6


def _check_normal_vector(stl: StlFile, fi: int) -> int:
    v0, v1, v2 = stl.facets[fi]
    e1 = v1 - v0
    e2 = v2 - v0
    normal = np.cross(e1.astype(np.float64), e2.astype(np.float64)).astype(np.float32)
    ln = np.linalg.norm(normal)
    if ln > 1e-12:
        normal /= ln
    stored = stl.normals[fi].copy()
    sn = np.linalg.norm(stored)
    if sn > 1e-12:
        stored /= sn
    if np.allclose(normal, stored, atol=0.001):
        return 0
    if np.allclose(normal, -stored, atol=0.001):
        return 2
    return 4


def _reverse_facet(stl: StlFile, fi: int):
    """Faithful port of stl_reverse_facet from thesis normals.c.
    Neighbor 0 gets +3, neighbor 1 gets +4, neighbor 2 gets +2 (all mod 6).
    """
    stl.stats.facets_reversed += 1
    nb = stl.neighbors[fi].copy()
    vnot = stl.which_vertex_not[fi].copy()
    # reverse the facet: swap vertex[0] and vertex[1]
    tmp = stl.facets[fi, 0].copy()
    stl.facets[fi, 0] = stl.facets[fi, 1].copy()
    stl.facets[fi, 1] = tmp
    # fix the vnots of each neighboring facet (different offsets per neighbor)
    offsets = [3, 4, 2]  # from thesis: nb[0]->+3, nb[1]->+4, nb[2]->+2
    for j in range(3):
        if nb[j] != -1 and nb[j] < stl.stats.number_of_facets:
            idx = (int(vnot[j]) + 1) % 3
            old = int(stl.which_vertex_not[nb[j], idx])
            stl.which_vertex_not[nb[j], idx] = (old + offsets[j]) % 6
    # swap neighbors 1 and 2
    stl.neighbors[fi, 1], stl.neighbors[fi, 2] = int(nb[2]), int(nb[1])
    # swap vnots 1 and 2
    stl.which_vertex_not[fi, 1] = vnot[2]
    stl.which_vertex_not[fi, 2] = vnot[1]
    # reverse all three vnot values
    for j in range(3):
        stl.which_vertex_not[fi, j] = (int(stl.which_vertex_not[fi, j]) + 3) % 6


def fix_normal_values(stl: StlFile, verbose=False):
    if verbose:
        print("  Fixing normal values...")
    n = stl.stats.number_of_facets
    if n == 0:
        return
    # Vectorized: compute all normals at once with NumPy
    v0 = stl.facets[:n, 0].astype(np.float64)
    v1 = stl.facets[:n, 1].astype(np.float64)
    v2 = stl.facets[:n, 2].astype(np.float64)
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)  # (n, 3) float64
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)  # (n, 1)
    mask = (lengths[:, 0] > 1e-12)
    normals[mask] /= lengths[mask]
    stl.normals[:n] = normals.astype(np.float32)


def calculate_volume(stl: StlFile):
    n = stl.stats.number_of_facets
    if n == 0:
        stl.stats.volume = 0.0
        return
    # Vectorized divergence-theorem volume calculation
    v0 = stl.facets[:n, 0].astype(np.float64)
    v1 = stl.facets[:n, 1].astype(np.float64)
    v2 = stl.facets[:n, 2].astype(np.float64)
    # Signed volume contribution of each triangle: (v0 · (v1 × v2)) / 6
    cross = np.cross(v1, v2)          # (n, 3)
    stl.stats.volume = float(np.sum(v0 * cross)) / 6.0


# ── Full repair pipeline ──────────────────────────────────────────

def repair(stl: StlFile, iterations: int = 2, verbose: bool = True):
    """Conservative exact cleanup. Approximate changes require the main policy.

    Never derive a weld distance from legitimate feature lengths or delete open
    facets. Indexed local hole/topology repairs are handled by MeshRepairPipeline.
    """
    check_facets_exact(stl, verbose=verbose)
    fix_normal_values(stl, verbose=verbose)
    calculate_volume(stl)


def stats_summary(stl: StlFile) -> str:
    s = stl.stats
    lines = [
        "=" * 60,
        f"  Results produced by Intial Cleanup Process ",
        "=" * 60,
        f"  Number of facets       : {s.original_num_facets:>8}  ->  {s.number_of_facets:>8}",
        f"  Degenerate facets      : {s.degenerate_facets}",
        f"  Edges fixed            : {s.edges_fixed}",
        f"  Facets removed         : {s.facets_removed}",
        f"  Facets added           : {s.facets_added}",
        f"  Facets reversed        : {s.facets_reversed}",
        f"  Normals fixed          : {s.normals_fixed}",
        f"  Number of parts        : {s.number_of_parts}",
        f"  Volume                 : {s.volume:.6f}",
        "=" * 60,
    ]
    return "\n".join(lines)
