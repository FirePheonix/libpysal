import numpy as np
import pandas as pd


def _weighted_covariance(graph, X):
    """Geographically weighted covariance matrix at each focal point.

    For each focal observation *i*, computes the geographically weighted
    covariance matrix using the kernel weights stored in the graph:

        Σ(uᵢ, vᵢ) = Xᵀ W(uᵢ, vᵢ) X

    where W(uᵢ, vᵢ) is the diagonal matrix of kernel weights for location *i*.
    This is the core building block for Geographically Weighted PCA (GWPCA) as
    described in Harris, Brunsdon & Charlton (2011, Eq. 4) and for Geographically
    Weighted Mahalanobis Distance (GWMD) as described in Harris et al. (2014).

    The implementation translates GWmodel's ``wpca`` weighted centering step::

        # R (GWmodel):
        local.center <- function(x, wt)
            sweep(x, 2, colSums(sweep(x, 1, wt, '*')) / sum(wt))

    Parameters
    ----------
    graph : Graph
        A ``libpysal.graph.Graph`` whose weights are kernel weights (e.g. built
        with ``Graph.build_kernel``). Row-standardised graphs are *not*
        appropriate here as they destroy the relative weight magnitudes.
    X : array-like, shape (n, p)
        Multivariate data matrix. Rows must align with ``graph.unique_ids`` in
        the same order. Should typically be standardised (zero mean, unit
        variance) before calling, following Harris et al. (2011, §3).

    Returns
    -------
    means : numpy.ndarray, shape (n_focal, p)
        Geographically weighted mean vector at each focal point.
    covariances : numpy.ndarray, shape (n_focal, p, p)
        Geographically weighted covariance matrix at each focal point.
    focal_ids : list
        Ordered list of focal IDs matching the first axis of ``means`` and
        ``covariances``.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import numpy as np
    >>> from geodatasets import get_path
    >>> from libpysal.graph import Graph
    >>> gdf = gpd.read_file(get_path("geoda.guerry")).set_geometry(
    ...     lambda g: g.centroid
    ... )
    >>> X = gdf[["Crm_prs", "Litercy", "Wealth"]].values.astype(float)
    >>> g = Graph.build_kernel(gdf.geometry, kernel="bisquare", bandwidth=50)
    >>> means, covs, ids = _weighted_covariance(g, X)
    >>> covs.shape  # (n_focal, 3, 3)
    (85, 3, 3)

    References
    ----------
    Harris P, Brunsdon C, Charlton M (2011). Geographically weighted principal
    components analysis. International Journal of Geographical Information
    Science, 25(11), 1717-1736.

    Harris P, Brunsdon C, Charlton M, Juggins S, Clarke A (2014). Multivariate
    spatial outlier detection using robust geographically weighted methods.
    Mathematical Geosciences, 46(1), 1-31.
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=float)
    else:
        X = X.astype(float)

    adjacency = graph._adjacency
    focal_ids = list(graph.unique_ids)
    n_focal = len(focal_ids)
    p = X.shape[1]

    # Build an index mapping from focal_id → row position in X
    id_to_pos = {fid: i for i, fid in enumerate(focal_ids)}

    means = np.empty((n_focal, p), dtype=float)
    covariances = np.empty((n_focal, p, p), dtype=float)

    for fi, focal_id in enumerate(focal_ids):
        # Neighbour weights for this focal point
        nbr = adjacency.loc[focal_id]
        wt = nbr.values.astype(float)
        wt_sum = wt.sum()

        if wt_sum == 0 or len(wt) == 0:
            means[fi] = np.full(p, np.nan)
            covariances[fi] = np.full((p, p), np.nan)
            continue

        # Gather neighbour rows from X
        nbr_positions = [id_to_pos[nid] for nid in nbr.index]
        X_nbr = X[nbr_positions]

        # Geographically weighted mean  —  Harris et al. (2011), §2.3
        # μ̂ = Σ(wⱼ xⱼ) / Σwⱼ
        w_mean = np.average(X_nbr, axis=0, weights=wt)
        means[fi] = w_mean

        # Geographically weighted covariance  —  Harris et al. (2011), Eq. 4
        # Σ(u,v) = Xᵀ W X / Σwⱼ
        # Equivalent to: X_scaled.T @ X_scaled  where X_scaled = √w · (X − μ̂)
        X_c = X_nbr - w_mean
        X_sc = X_c * np.sqrt(wt[:, np.newaxis])
        covariances[fi] = (X_sc.T @ X_sc) / wt_sum

    return means, covariances, focal_ids


def _lag_spatial(graph, y, categorical=False, ties="raise"):
    """Spatial lag operator

    Constructs spatial lag based on neighbor relations of the graph.


    Parameters
    ----------
    graph : Graph
        libpysal.graph.Graph
    y : array
        numpy array with dimensionality conforming to w. Can be 2D if all
        columns are numerical.
    categorical : bool
        True if y is categorical, False if y is continuous.
    ties : {'raise', 'random', 'tryself'}, optional
        Policy on how to break ties when a focal unit has multiple
        modes for a categorical lag.
        - 'raise': This will raise an exception if ties are
          encountered to alert the user (Default).
        - 'random': modal label ties Will be broken randomly.
        - 'tryself': check if focal label breaks the tie between label
          modes.  If the focal label does not break the modal tie, the
          tie will be be broken randomly. If the focal unit has a
          self-weight, focal label is not used to break any tie,
          rather any tie will be broken randomly.


    Returns
    -------
    numpy.array
        array of numeric|categorical values for the spatial lag


    Examples
    --------
    >>> from libpysal.graph._spatial_lag import _lag_spatial
    >>> import numpy as np
    >>> from libpysal.weights.util import lat2W
    >>> from libpysal.graph import Graph
    >>> graph = Graph.from_W(lat2W(3,3))
    >>> y = np.arange(9)
    >>> _lag_spatial(graph, y)
    array([ 4.,  6.,  6., 10., 16., 14., 10., 18., 12.])

    Row standardization
    >>> w = lat2W(3,3)
    >>> w.transform = 'r'
    >>> graph = Graph.from_W(w)
    >>> y = np.arange(9)
    >>> _lag_spatial(graph, y)
    array([2.        , 2.        , 3.        , 3.33333333, 4.        ,
           4.66666667, 5.        , 6.        , 6.        ])


    Categorical Lag (no ties)
    >>> y = np.array([*'ababcbcbc'])
    >>> _lag_spatial(graph, y, categorical=True)
    array(['b', 'a', 'b', 'c', 'b', 'c', 'b', 'c', 'b'], dtype=object)

    Handling ties
    >>> y[3] = 'a'
    >>> np.random.seed(12345)
    >>> _lag_spatial(graph, y, categorical=True, ties='random')
    array(['a', 'a', 'b', 'c', 'b', 'c', 'b', 'c', 'b'], dtype=object)
    >>> _lag_spatial(graph, y, categorical=True, ties='random')
    array(['b', 'a', 'b', 'c', 'b', 'c', 'b', 'c', 'b'], dtype=object)
    >>> _lag_spatial(graph, y, categorical=True, ties='tryself')
    array(['a', 'a', 'b', 'c', 'b', 'c', 'a', 'c', 'b'], dtype=object)

    """
    sp = graph.sparse
    if len(y) != sp.shape[0]:
        raise ValueError(
            "The length of `y` needs to match the number of observations "
            f"in Graph. Expected {sp.shape[0]}, got {len(y)}."
        )

    # coerce list to array
    if isinstance(y, list):
        y = np.array(y)

    if y.ndim == 1 and (
        categorical
        or isinstance(y.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(y.dtype)
        or pd.api.types.is_bool_dtype(y.dtype)
        or pd.api.types.is_string_dtype(y.dtype)
    ):
        return _categorical(graph, y, ties=ties)

    return sp @ y


def _categorical(graph, y, ties):
    """
    Compute the categorical spatial lag for each observation in a graph.

    Parameters
    ----------
    graph : object
    y : array-like (numpy.ndarray or pandas.Series)
        Categorical labels for each observation.
    ties : {'raise', 'random', 'tryself'}
        How to handle ties when multiple neighbor categories are equally frequent:
          - 'raise' : raise a ValueError if any tie exists.
          - 'random': break ties uniformly at random.
          - 'tryself': if the focal unit's own label is among the tied labels,
                       choose the focal label; otherwise break ties (deterministic
                       choice defined by helper routine).

    Returns
    -------
    numpy.ndarray
        An array of categorical spatial lag values aligned with graph.unique_ids.

    Raises
    ------
    ValueError
        - If ties are present and ties == 'raise'.
        - If ties is not one of 'raise', 'random', or 'tryself'.

    Notes
    -----
    The implementation groups adjacency entries by focal unit and counts neighbor
    labels to determine the modal category per focal. Tie detection and
    resolution are delegated to the helper functions _check_ties and
    _get_categorical_lag. Using 'random' produces nondeterministic outputs unless
    a random seed is fixed externally.
    """
    if isinstance(y, np.ndarray):
        y = pd.Series(y, index=graph.unique_ids)

    df = pd.DataFrame(data=graph.adjacency)
    df["neighbor_label"] = y.loc[graph.adjacency.index.get_level_values(1)].values
    df["own_label"] = y.loc[graph.adjacency.index.get_level_values(0)].values
    df["neighbor_idx"] = df.index.get_level_values(1)
    df["focal_idx"] = df.index.get_level_values(0)
    gb = df.groupby(["focal", "neighbor_label"]).count().groupby(level="focal")
    n_ties = gb.apply(_check_ties).sum()
    if n_ties and ties == "raise":
        raise ValueError(
            f"There are {n_ties} ties that must be broken "
            f"to define the categorical "
            "spatial lag for these observations. To address this "
            "issue, consider setting `ties='tryself'` "
            "or `ties='random'` or consult the documentation "
            "about ties and the categorical spatial lag."
        )
    # either there are ties and random|tryself specified or
    # there are no ties
    gb = df.groupby(by=["focal"])
    if ties == "random" or ties == "raise":
        return gb.apply(_get_categorical_lag).values
    elif ties == "tryself" or ties == "raise":
        return gb.apply(_get_categorical_lag, ties="tryself").values
    else:
        raise ValueError(
            f"Received option ties='{ties}', but only options "
            "'raise','random','tryself' are supported."
        )


def _check_ties(focal):
    """Reduction to determine if a focal unit has multiple modes for neighbor labels.

    Parameters
    ----------
    focal: row from pandas Dataframe
          Data is a Graph with an additional column having the labels for the neighbors

    Returns
    -------
    bool
    """

    max_count = focal.weight.max()
    return (focal.weight == max_count).sum() > 1


def _get_categorical_lag(focal, ties="random"):
    """Reduction to determine categorical spatial lag for a focal unit.

    Parameters
    ----------
    focal: row from pandas Dataframe
          Data is a Graph with an additional column having the labels for the neighbors

    ties : {'raise', 'random', 'tryself'}, optional
        Policy on how to break ties when a focal unit has multiple
        modes for a categorical lag.
        - 'raise': This will raise an exception if ties are
          encountered to alert the user (Default).
        - 'random': Will break ties randomly.
        - 'tryself': check if focal label breaks the tie between label
          modes.  If the focal label does not break the modal tie, the
          tie will be be broken randomly. If the focal unit has a
          self-weight, focal label is not used to break any tie,
          rather any tie will be broken randomly.


    Returns
    -------
    str|int|float:
      Label for the value of the categorical lag
    """
    self_weight = focal.focal_idx.values[0] in focal.neighbor_idx.values
    labels, counts = np.unique(focal.neighbor_label, return_counts=True)
    node_label = labels[counts == counts.max()]
    if ties == "random" or (ties == "tryself" and self_weight):
        return np.random.choice(node_label, 1)[0]
    elif ties == "tryself" and not self_weight:
        self_label = focal.own_label.values[0]
        if self_label in node_label:  # focal breaks tie
            return self_label
        else:
            return np.random.choice(node_label, 1)[0]
