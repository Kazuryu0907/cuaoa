"""
rqaoa_maxkcut.classical
=======================
Classical MAX-k-CUT approximation algorithms for benchmarking.

Modules
-------
newman : NewmanMaxKCut — SDP-based approximation (Newman 2018)
"""
from .newman import NewmanMaxKCut

__all__ = ["NewmanMaxKCut"]
