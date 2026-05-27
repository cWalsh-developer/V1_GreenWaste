import numpy as np

from greenwaste.size_estimation import (
    CaptureIntrinsics,
    estimate_size_from_roi,
)


def test_estimate_size_from_roi_uses_median_depth():
    depth = np.array(
        [
            [0, 0, 0, 0],
            [0, 1000, 1000, 0],
            [0, 1000, 4000, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    roi = (1, 1, 2, 2)
    intrinsics = CaptureIntrinsics(fx=500.0, fy=500.0, ppx=0.0, ppy=0.0)
    estimate = estimate_size_from_roi(depth, roi, intrinsics, depth_scale=0.001)

    # median depth in ROI is 1000 mm => 1.0 m
    assert np.isclose(estimate.distance_cm, 100.0)
    # thickness from 30th/70th percentiles: 0 mm => 0 cm
    assert np.isclose(estimate.depth_cm, 0.0)
    assert estimate.roi_refined == roi
    assert np.isclose(estimate.width_cm, (2 / 500.0) * 100.0)
    assert np.isclose(estimate.height_cm, (2 / 500.0) * 100.0)


def test_estimate_size_from_roi_requires_valid_depth():
    depth = np.zeros((3, 3), dtype=np.uint16)
    roi = (0, 0, 3, 3)
    intrinsics = CaptureIntrinsics(fx=500.0, fy=500.0, ppx=0.0, ppy=0.0)

    try:
        estimate_size_from_roi(depth, roi, intrinsics, depth_scale=0.001)
    except ValueError as exc:
        assert "valid depth" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing depth")
