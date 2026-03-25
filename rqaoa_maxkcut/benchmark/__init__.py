"""
rqaoa_maxkcut.benchmark
=======================
Benchmark utilities: graph generation and RQAOA vs Newman comparison.

Modules
-------
graph_gen     : generate_3colorable_regular_graph — G[d,n] ensemble
run_benchmark : run_comparison, compute_approx_ratio
"""
from .graph_gen import generate_3colorable_regular_graph
from .run_benchmark import run_comparison, compute_approx_ratio

__all__ = [
    "generate_3colorable_regular_graph",
    "run_comparison",
    "compute_approx_ratio",
]
