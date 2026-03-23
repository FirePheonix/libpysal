import numpy as np
import pytest

from libpysal import graph
from libpysal.graph._spatial_lag import _lag_spatial, _weighted_covariance
from libpysal.weights import lat2W


class TestLag:
    def setup_method(self):
        self.neighbors = {
            "a": ["b"],
            "b": ["c", "a"],
            "c": ["b"],
            "d": [],
        }
        self.weights = {"a": [1.0], "b": [1.0, 1.0], "c": [1.0], "d": []}
        self.g = graph.Graph.from_dicts(self.neighbors, self.weights)
        self.y = np.array([0, 1, 2, 3])
        self.yc = np.array([*"ababcbcbc"])
        w = lat2W(3, 3)
        w.transform = "r"
        self.gc = graph.Graph.from_W(w)

    def test_lag_spatial(self):
        yl = _lag_spatial(self.g, self.y)
        np.testing.assert_array_almost_equal(yl, [1.0, 2.0, 1.0, 0])
        g = graph.Graph.from_W(lat2W(3, 3))
        y = np.arange(9)
        yl = _lag_spatial(g, y)
        ylc = np.array([4.0, 6.0, 6.0, 10.0, 16.0, 14.0, 10.0, 18.0, 12.0])
        np.testing.assert_array_almost_equal(yl, ylc)
        g_row = g.transform("r")
        yl = _lag_spatial(g_row, y)
        ylc = np.array([2.0, 2.0, 3.0, 3.33333333, 4.0, 4.66666667, 5.0, 6.0, 6.0])
        np.testing.assert_array_almost_equal(yl, ylc)

    def test_lag_spatial_categorical(self):
        yl = _lag_spatial(self.gc, self.yc)
        ylc = np.array(["b", "a", "b", "c", "b", "c", "b", "c", "b"], dtype=object)
        np.testing.assert_array_equal(yl, ylc)
        self.yc[3] = "a"  # create ties
        np.random.seed(12345)
        yl = _lag_spatial(self.gc, self.yc, categorical=True, ties="random")
        ylc = np.array(["a", "a", "b", "c", "b", "c", "b", "c", "b"], dtype=object)
        yl1 = _lag_spatial(self.gc, self.yc, categorical=True, ties="random")
        yls = _lag_spatial(self.gc, self.yc, categorical=True, ties="tryself")
        np.testing.assert_array_equal(yl, ylc)
        yl1c = np.array(["b", "a", "b", "c", "b", "c", "b", "c", "b"], dtype=object)
        np.testing.assert_array_equal(yl1, yl1c)
        ylsc = np.array(["a", "a", "b", "c", "b", "c", "a", "c", "b"], dtype=object)
        np.testing.assert_array_equal(yls, ylsc)
        # self-weight
        neighbors = self.gc.neighbors
        neighbors[0] = (0, 3, 1)  # add self neighbor for observation 0
        gc = graph.Graph.from_dicts(neighbors)
        self.yc[3] = "b"
        yls = _lag_spatial(gc, self.yc, categorical=True, ties="tryself")
        assert yls[0] in ["b", "a"]
        self.yc[3] = "a"
        yls = _lag_spatial(gc, self.yc, categorical=True, ties="tryself")
        assert yls[0] == "a"

    def test_ties_raise(self):
        with pytest.raises(ValueError, match="There are 2 ties that must be broken"):
            self.yc[3] = "a"  # create ties
            _lag_spatial(self.gc, self.yc, categorical=True)

    def test_categorical_custom_index(self):
        expected = np.array(["bar", "foo", "bar", "foo"])
        np.testing.assert_array_equal(
            expected, self.g.lag(["foo", "bar", "foo", "foo"])
        )

    def test_2d_array(self):
        ys = np.arange(27).reshape(9, 3)
        lag = self.gc.lag(ys)

        expected = np.array(
            [
                [6.0, 7.0, 8.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [10.0, 11.0, 12.0],
                [12.0, 13.0, 14.0],
                [14.0, 15.0, 16.0],
                [15.0, 16.0, 17.0],
                [18.0, 19.0, 20.0],
                [18.0, 19.0, 20.0],
            ]
        )

        np.testing.assert_array_almost_equal(lag, expected)

        # test equality to 1d
        for i in range(2):
            np.testing.assert_array_equal(self.gc.lag(ys[:, i]), lag[:, i])


class TestWeightedCovariance:
    """Tests for _weighted_covariance and Graph.weighted_covariance."""

    def setup_method(self):
        # Simple 4-observation, 2-feature dataset with known structure
        # graph: a-b, b-c, c-d (chain)
        neighbors = {"a": ["b", "a"], "b": ["a", "b", "c"], "c": ["b", "c", "d"], "d": ["c", "d"]}
        weights = {
            "a": [0.5, 1.0],
            "b": [0.5, 1.0, 0.5],
            "c": [0.5, 1.0, 0.5],
            "d": [0.5, 1.0],
        }
        self.g = graph.Graph.from_dicts(neighbors, weights)
        self.X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    def test_output_shapes(self):
        means, covs, ids = _weighted_covariance(self.g, self.X)
        n = len(self.g.unique_ids)
        p = self.X.shape[1]
        assert means.shape == (n, p)
        assert covs.shape == (n, p, p)
        assert len(ids) == n

    def test_covariances_symmetric(self):
        _, covs, _ = _weighted_covariance(self.g, self.X)
        for i in range(len(covs)):
            np.testing.assert_allclose(covs[i], covs[i].T, atol=1e-12)

    def test_covariances_psd(self):
        """All non-degenerate covariance matrices must be positive semi-definite."""
        _, covs, _ = _weighted_covariance(self.g, self.X)
        for cov in covs:
            eigenvalues = np.linalg.eigvalsh(cov)
            assert np.all(eigenvalues >= -1e-10), f"Not PSD: min eigenvalue = {eigenvalues.min()}"

    def test_weighted_mean_correct(self):
        """Verify the weighted mean for focal 'a' manually."""
        # focal 'a': neighbors a (w=1.0) and b (w=0.5)
        # w_mean = (1.0*X[0] + 0.5*X[1]) / 1.5
        expected_mean = (1.0 * self.X[0] + 0.5 * self.X[1]) / 1.5
        means, _, ids = _weighted_covariance(self.g, self.X)
        idx = ids.index("a")
        np.testing.assert_allclose(means[idx], expected_mean, atol=1e-12)

    def test_matches_manual_computation(self):
        """Cross-check focal 'b' covariance against a manual calculation."""
        # focal 'b': neighbors a (w=0.5), b (w=1.0), c (w=0.5)
        nbr_X = self.X[[0, 1, 2]]
        wt = np.array([0.5, 1.0, 0.5])
        wt_sum = wt.sum()
        w_mean = np.average(nbr_X, axis=0, weights=wt)
        X_c = nbr_X - w_mean
        X_sc = X_c * np.sqrt(wt[:, np.newaxis])
        expected_cov = (X_sc.T @ X_sc) / wt_sum

        _, covs, ids = _weighted_covariance(self.g, self.X)
        idx = ids.index("b")
        np.testing.assert_allclose(covs[idx], expected_cov, atol=1e-12)

    def test_accepts_dataframe(self):
        """_weighted_covariance should accept both ndarray and DataFrame."""
        import pandas as pd

        X_df = pd.DataFrame(self.X, columns=["x1", "x2"])
        means_arr, covs_arr, ids_arr = _weighted_covariance(self.g, self.X)
        means_df, covs_df, ids_df = _weighted_covariance(self.g, X_df)
        np.testing.assert_allclose(means_arr, means_df, atol=1e-12)
        np.testing.assert_allclose(covs_arr, covs_df, atol=1e-12)

    def test_graph_method_matches_function(self):
        """Graph.weighted_covariance() must return identical results to the function."""
        means_fn, covs_fn, ids_fn = _weighted_covariance(self.g, self.X)
        means_m, covs_m, ids_m = self.g.weighted_covariance(self.X)
        np.testing.assert_allclose(means_fn, means_m, atol=1e-12)
        np.testing.assert_allclose(covs_fn, covs_m, atol=1e-12)
        assert ids_fn == ids_m

    def test_isolated_node_returns_nan(self):
        """A node with zero total weight should return NaN covariance."""
        neighbors = {"a": ["b"], "b": ["a"], "c": []}
        weights = {"a": [1.0], "b": [1.0], "c": []}
        g = graph.Graph.from_dicts(neighbors, weights)
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        means, covs, ids = _weighted_covariance(g, X)
        idx = ids.index("c")
        assert np.all(np.isnan(covs[idx]))
