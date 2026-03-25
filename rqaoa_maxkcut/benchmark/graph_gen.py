"""
benchmark/graph_gen.py
======================
3-colourable regular graph generator: G[d, n] ensemble.

Reference: Bravyi et al. 2022, Section 5.

Construction
------------
1. Partition n vertices into three equal parts V_1, V_2, V_3 (n must be
   divisible by 3).
2. For each pair (r, s) with r < s generate a random d-regular bipartite
   graph between V_r and V_s.
3. Verify the result is triangle-free and connected; retry if not.

The resulting graph is 3-colourable by construction (assign colour r to
every vertex in V_r).  It has 3n/3 = n vertices and 3 × d × n/3 = d·n
edges (each part-pair contributes d × n/3 edges), so it is 2d-regular.

Note
----
Strict d-regularity on *both* sides of each bipartite subgraph is only
achievable when d divides n/3.  We use a configuration-model approach:

    For each (r, s): randomly match the d·(n/3) half-edges on each side.
    This gives exactly d edges per vertex on both sides.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Optional


def generate_3colorable_regular_graph(
    n: int,
    d: int,
    seed: Optional[int] = None,
    max_attempts: int = 1000,
) -> nx.Graph:
    """
    Generate a random graph from the G[d, n] ensemble.

    Parameters
    ----------
    n           : int   Number of vertices (must be divisible by 3).
    d           : int   Degree of the bipartite sub-graphs (2d-regular overall).
    seed        : int | None
    max_attempts: int   Restart limit for triangle-free + connected check.

    Returns
    -------
    G : nx.Graph  A 3-colourable, (attempt to be) triangle-free, connected graph.

    Raises
    ------
    ValueError  if n % 3 != 0 or if generation fails after max_attempts.
    """
    if n % 3 != 0:
        raise ValueError(f"n must be divisible by 3, got n={n}")
    if d < 1:
        raise ValueError(f"d must be at least 1, got d={d}")

    rng = np.random.default_rng(seed)
    part_size = n // 3

    # Vertex sets: V_r = [r * part_size .. (r+1) * part_size - 1]
    parts = [
        list(range(r * part_size, (r + 1) * part_size))
        for r in range(3)
    ]

    for attempt in range(max_attempts):
        G = nx.Graph()
        G.add_nodes_from(range(n))

        success = True
        for r in range(3):
            for s in range(r + 1, 3):
                bipartite_ok = _add_regular_bipartite(
                    G, parts[r], parts[s], d, rng
                )
                if not bipartite_ok:
                    success = False
                    break
            if not success:
                break

        if not success:
            continue

        # Check connectivity
        if not nx.is_connected(G):
            continue

        # Check triangle-free (optional but preferred)
        if _has_triangle(G):
            continue

        return G

    # If we could not get triangle-free in max_attempts, relax that constraint
    rng2 = np.random.default_rng(seed)
    for attempt in range(max_attempts):
        G = nx.Graph()
        G.add_nodes_from(range(n))

        success = True
        for r in range(3):
            for s in range(r + 1, 3):
                bipartite_ok = _add_regular_bipartite(
                    G, parts[r], parts[s], d, rng2
                )
                if not bipartite_ok:
                    success = False
                    break
            if not success:
                break

        if not success:
            continue

        if nx.is_connected(G):
            return G

    raise ValueError(
        f"Could not generate a valid G[d={d}, n={n}] graph "
        f"after {max_attempts} attempts."
    )


def _add_regular_bipartite(
    G: nx.Graph,
    left: list,
    right: list,
    d: int,
    rng: np.random.Generator,
) -> bool:
    """
    Add a random d-regular bipartite graph between `left` and `right` to G.

    Uses the configuration model:
      - Create d copies of each left vertex → stubs_left (length d*|left|)
      - Create d copies of each right vertex → stubs_right (length d*|right|)
      - Shuffle stubs_right and match with stubs_left

    Returns True if successful (no self-loops, no multi-edges within G).
    """
    nl = len(left)
    nr = len(right)
    if nl * d != nr * d:
        # Both sides must have equal total degree
        if nl != nr:
            return False

    stubs_left = [v for v in left for _ in range(d)]
    stubs_right = [v for v in right for _ in range(d)]

    if len(stubs_left) != len(stubs_right):
        return False

    rng.shuffle(stubs_right)

    for u, v in zip(stubs_left, stubs_right):
        if G.has_edge(u, v):
            return False   # multi-edge
        G.add_edge(u, v)

    return True


def _has_triangle(G: nx.Graph) -> bool:
    """Return True if G contains at least one triangle."""
    for node in G.nodes():
        neighbors = set(G.neighbors(node))
        for nb in neighbors:
            if len(neighbors & set(G.neighbors(nb))) > 0:
                return True
    return False
