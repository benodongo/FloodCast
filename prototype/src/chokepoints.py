"""Chokepoint locations for the South C spatial risk map.

Six representative hydraulic chokepoints / vulnerable nodes across the South C
area of Nairobi. Coordinates are approximate and for demonstration only. Each
maps to one synthetic sub-catchment (``choke_j``) and carries a susceptibility
label used to differentiate local risk on the map.
"""
from __future__ import annotations

# South C sits roughly around lat -1.315, lon 36.832.
CHOKEPOINTS = [
    {"id": 0, "name": "Mater Hospital roundabout", "lat": -1.3078, "lon": 36.8360,
     "note": "Major junction, low-lying"},
    {"id": 1, "name": "Muhoho Avenue culvert", "lat": -1.3141, "lon": 36.8291,
     "note": "Undersized culvert"},
    {"id": 2, "name": "Boma Road drainage node", "lat": -1.3196, "lon": 36.8338,
     "note": "Chronic ponding"},
    {"id": 3, "name": "Kasarani Close outfall", "lat": -1.3225, "lon": 36.8262,
     "note": "Blocked outfall"},
    {"id": 4, "name": "Mugoya Estate gutter", "lat": -1.3167, "lon": 36.8399,
     "note": "Dense impervious cover"},
    {"id": 5, "name": "South C shopping centre", "lat": -1.3112, "lon": 36.8315,
     "note": "Community node, high footfall"},
]
