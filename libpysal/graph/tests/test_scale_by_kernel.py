import numpy as np
import pandas as pd

from libpysal import graph


class TestScaleByKernel:
    def setup_method(self):
        # Weights are all 1 initially
        self.adj = pd.Series(
            [1, 1, 1, 1],
            index=pd.MultiIndex.from_tuples(
                [(0, 1), (1, 0), (1, 2), (2, 1)], names=["focal", "neighbor"]
            ),
            name="weight",
        )
        self.g = graph.Graph(self.adj)

    def test_scale_by_kernel_identity(self):
        # Time values are all the same, distance is 0

        values = [0, 0, 0]
        # set a bandwidth
        g_scaled = self.g.scale_by_kernel(values, bandwidth=1.0, kernel="gaussian")

        expected_scaling = 1.0 / np.sqrt(2 * np.pi)
        pd.testing.assert_series_equal(
            self.g.adjacency * expected_scaling, g_scaled.adjacency, check_dtype=False
        )

    def test_scale_by_kernel_temporal_decay(self):
        # Node 0 at t=0, Node 1 at t=1, Node 2 at t=2
        values = [0, 1, 2]

        g_scaled = self.g.scale_by_kernel(values, bandwidth=1.0, kernel="gaussian")

        weights = g_scaled.adjacency
        # 1/sqrt(2pi) * exp(-0.5)
        expected_weight = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5)

        # Check 0-1 (index (0,1))
        assert np.isclose(weights.loc[0, 1], expected_weight)
        # Check 1-2 (index (1,2))
        assert np.isclose(weights.loc[1, 2], expected_weight)

    def test_scale_by_kernel_different_bandwidth(self):
        values = [0, 10, 20]
        # If bandwidth = 10, then d/b = 1. Kernel should be same as above.

        g_scaled = self.g.scale_by_kernel(values, bandwidth=10.0, kernel="gaussian")

        weights = g_scaled.adjacency
        expected_weight = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5)

        assert np.isclose(weights.loc[0, 1], expected_weight)

    def test_scale_by_kernel_bisquare(self):
        values = [0, 0.5, 1.0]
        g_scaled = self.g.scale_by_kernel(values, bandwidth=1.0, kernel="bisquare")

        weights = g_scaled.adjacency
        expected = (15 / 16) * 0.5625

        assert np.isclose(weights.loc[0, 1], expected)
