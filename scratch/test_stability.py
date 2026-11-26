import sys
import numpy as np
sys.path.append(".")
from src.core.repair_utils import _signed_volume

def test_stability():
    # Unit cube at origin
    v_orig = np.array([
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,1], [1,0,1], [1,1,1], [0,1,1]
    ], dtype=np.float64)
    f = np.array([
        [0,2,1], [0,3,2], [4,5,6], [4,6,7], # bottom, top
        [0,1,5], [0,5,4], [1,2,6], [1,6,5], # front, right
        [2,3,7], [2,7,6], [3,0,4], [3,4,7]  # back, left
    ])
    
    vol_orig = _signed_volume(v_orig, f)
    print(f"Volume at origin: {vol_orig:.6f}")
    
    # Unit cube translated far away
    offset = np.array([1e6, 1e6, 1e6])
    v_far = v_orig + offset
    vol_far = _signed_volume(v_far, f)
    print(f"Volume at 1e6:    {vol_far:.6f}")
    
    assert abs(vol_orig - 1.0) < 1e-10
    assert abs(vol_far - 1.0) < 1e-10
    print("Test PASSED!")

if __name__ == "__main__":
    test_stability()
