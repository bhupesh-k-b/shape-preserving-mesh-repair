import numpy as np
import pytest
from src.core.mesh_analyzer import MeshData
from src.core.repair_utils import _recompute_normals


@pytest.fixture
def cube():
    v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                  [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=float)
    f = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],
                  [0,1,5],[0,5,4],[2,3,7],[2,7,6],
                  [0,4,7],[0,7,3],[1,2,6],[1,6,5]], dtype=np.int32)
    return MeshData(v, f, _recompute_normals(v, f))
