"""
rqaoa_maxkcut
=============
Level-1 RQAOA for MAX-k-CUT based on Bravyi et al. (Quantum, 2022).

Subpackages
-----------
core       : Hamiltonian, density matrix, angle optimiser, RQAOA solver
classical  : Newman SDP-based approximation algorithm
benchmark  : Graph generation and benchmark runner
"""
from .core import (
    MaxKCutHamiltonian,
    DensityMatrixSimulator,
    AngleOptimizer,
    RQAOA1Solver,
)

__all__ = [
    "MaxKCutHamiltonian",
    "DensityMatrixSimulator",
    "AngleOptimizer",
    "RQAOA1Solver",
]
