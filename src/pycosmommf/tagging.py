from __future__ import annotations

import numpy as np
import skimage.measure


def _cast_threshold(threshold, dtype, op):
    """
    Cast a float64 ``threshold`` to ``dtype`` so that comparing a ``dtype``
    array against the cast value gives the exact float64 answer.

    For a float32 value ``x`` and a float64 threshold ``t``, ``x > t`` is
    equivalent to ``x > below`` where ``below`` is the largest float32
    ``<= t``, and ``x < t`` is equivalent to ``x < above`` where ``above`` is
    the smallest float32 ``>= t``. Comparing against the cast value keeps the
    comparison in ``dtype`` (no float64 copy of the array) and makes the
    result independent of numpy's scalar promotion rules, which differ
    between numpy 1.x (compares in float32 with ``t`` rounded to nearest) and
    numpy 2.x (compares in float64). Both tagging backends use this helper so
    their masks agree voxel for voxel.

    Args:
        threshold (:obj:`float`):
            The threshold value.
        dtype (:obj:`numpy.dtype`):
            The dtype of the array that will be compared against the threshold.
        op (:obj:`str`):
            ``">"`` or ``"<"``, the comparison the threshold will be used in.

    Returns:
        (:obj:`numpy scalar`): The threshold as a ``dtype`` scalar (or as a
        float64 scalar when ``dtype`` is not a float type narrower than 64 bits).
    """
    dtype = np.dtype(dtype)
    threshold = float(threshold)
    if dtype.kind != "f" or dtype.itemsize >= 8:
        return np.float64(threshold)
    cast = dtype.type(threshold)
    if op == ">" and float(cast) > threshold:
        cast = np.nextafter(cast, dtype.type(-np.inf))
    elif op == "<" and float(cast) < threshold:
        cast = np.nextafter(cast, dtype.type(np.inf))
    return cast


def _cluster_signature_thresholds():
    """The cluster-signature thresholds scanned by ``make_the_clusbool()``."""
    return np.geomspace(0.1, 30, 10)  # hard coded for now


def _clump_mean_overdensity(clumps, n_clumps, delta):
    """
    Mean δ of each labelled clump.

    ``clumps`` holds labels ``1..n_clumps`` with ``0`` as background. Sums are
    accumulated in float64 with ``np.bincount``, which is deterministic (the
    previous Numba ``prange`` loop raced on the per-clump accumulators and
    silently dropped voxels from clumps that spanned several threads).
    """
    labels = clumps.ravel()
    counts = np.bincount(labels, minlength=n_clumps + 1)[1:]
    sums = np.bincount(labels, weights=delta.ravel(), minlength=n_clumps + 1)[1:]
    return sums / counts


def _virialized_fraction(mean_overdensity, overdensity_threshold):
    """Fraction of clumps whose mean δ exceeds ``overdensity_threshold``; NaN if there are no clumps."""
    if mean_overdensity.size == 0:
        return np.nan
    return np.sum(mean_overdensity > overdensity_threshold) / mean_overdensity.size


def _select_cluster_threshold(signature_thresholds, virialized_fractions):
    """
    Pick the threshold whose virialized fraction is closest to 0.5.

    Thresholds without any clumps have a NaN fraction and are skipped. If no
    threshold has clumps, no voxel exceeds the lowest threshold, so the
    resulting cluster filter is empty whichever threshold is returned; the
    lowest one is used.
    """
    distance = np.abs(virialized_fractions - 0.5)
    if np.all(np.isnan(distance)):
        return signature_thresholds[0]
    return signature_thresholds[np.nanargmin(distance)]


def make_the_clusbool(delta, max_sigs, overdensity_threshold):
    """
    A function that creates a bool Filter which selects only
    the clusters in a given cosmological δ field. This function accomplishes
    this by finding a threshold value for 𝒮_cluster by looking at the
    change in virialization fraction as a function of 𝒮_cluster.

    Args:
        delta (:obj:`3D float np.ndarray`):
            delta refers to the (δ = ρ / <ρ> - 1) field, this is NOT the (δ + 1 = ρ / <ρ>) field.
        max_sigs (:obj:`4D float np.ndarray`):
            the maximum signatures array from the ``maximum_signature()`` function.
        verbose_flag (:obj:`bool`):
            a flag to allow the function to be more helpful and verbose.
        overdensity_threshold (:obj:`float`):
            The overdensity parameter threshold for determining virialization. Commonly used values
            that are physically motivated can be 370, 200, or 500.

    Returns:
        (tuple): a tuple containing:
            - **clusbool** (:obj:`3D bool np.ndarray`): a boolean filter that selects only the clusters in a given cosmological δ field.
            - **S_th** (:obj:`float`): the threshold value for 𝒮_cluster.
            - **signature_thresholds** (:obj:`1D float np.ndarray`): an array of threshold values for 𝒮_cluster.
            - **virialized_fractions** (:obj:`1D float np.ndarray`): an array of virialized fractions as a function of 𝒮_cluster. Thresholds with no clumps get NaN.
    """

    signature_thresholds = _cluster_signature_thresholds()
    virialized_fractions = np.zeros_like(signature_thresholds)
    cluster_signature = max_sigs[:, :, :, 0]

    for i, S_th in enumerate(signature_thresholds):
        # Identify Clumps with S > S_th
        _clusbool = cluster_signature > _cast_threshold(
            S_th, cluster_signature.dtype, ">"
        )

        # 0 is background and 1, 2, 3, ... are the clumps (26-connected)
        clumps, n_clumps = skimage.measure.label(
            _clusbool, connectivity=_clusbool.ndim, return_num=True
        )
        averaged_overdensity_per_clump = _clump_mean_overdensity(
            clumps, n_clumps, delta
        )

        virialized_fractions[i] = _virialized_fraction(
            averaged_overdensity_per_clump, overdensity_threshold
        )

    S_th = _select_cluster_threshold(signature_thresholds, virialized_fractions)
    clusbool = cluster_signature > _cast_threshold(S_th, cluster_signature.dtype, ">")

    return clusbool, S_th, signature_thresholds, virialized_fractions


def _log10S_grid(Smin, Smax):
    """The log10(𝒮) grid on which the mass change curves are evaluated."""
    return np.arange(Smin, Smax + 0.1, 0.1)


def _mass_change_curve(log10S, M):
    """
    Turn the enclosed masses ``M`` (one per ``log10S`` value) into the
    ``|dM²/dlog(𝒮)|`` curve evaluated at the midpoints of ``log10S``.
    """
    # Compute the derivative |dM^2/dlog(𝒮)|
    ΔM_2 = np.abs(np.diff(M**2) / np.diff(log10S))

    # Compute log(𝒮) array used in derivatives (midpoints)
    midx = (log10S[:-1] + log10S[1:]) / 2
    S = 10**midx

    return S, ΔM_2


def calc_mass_change(sig_vec, density_vec, Smin, Smax):
    """
    Calculate the mass change curve for a given structure type.

    Args:
        sig_vec (:obj:`1D float np.ndarray`):
            the signature values for a given structure type
        density_vec (:obj:`1D float np.ndarray`):
            the density values for a given structure type
        Smin (:obj:`float`):
            the minimum signature value
        Smax (:obj:`float`):
            the maximum signature value

    Returns:
        (tuple): a tuple containing:
            - **S** (:obj:`1D float np.ndarray`): the signature values
            - **ΔΜ_2** (:obj:`1D float np.ndarray`): the mass change curve
    """
    # Initialize our arrays
    log10S = _log10S_grid(Smin, Smax)
    M = np.zeros_like(log10S)

    # Sum up all the mass in a structure type as a function of log(𝒮)
    for i in range(len(M)):
        filter = sig_vec > _cast_threshold(10 ** log10S[i], sig_vec.dtype, ">")
        M[i] = np.sum(density_vec[filter], dtype=np.float64)

    return _mass_change_curve(log10S, M)


def _check_not_overdensity(mean_density):
    """Raise ``ValueError`` if the field mean is ~0, i.e. δ was passed instead of δ+1."""
    if np.isclose(np.abs(mean_density), 0, atol=1e-3):
        msg = "make sure that you are not inputing the overdensity field δ = ρ/<ρ> - 1, but rather δ + 1 = ρ/<ρ>."
        raise ValueError(msg)


def _structure_bools_cpu(
    density_cube,
    max_sigs,
    verbose_flag,
    clusbool,
    Smin,
    Smax,
    overdensity_threshold,
):
    """
    NumPy implementation of ``calc_structure_bools()``. Returns a dict with
    the four boolean filters, the thresholds and curves behind them and,
    when ``verbose_flag`` is set, the volume and mass fractions.
    """
    N_cells = np.prod(density_cube.shape)

    # Check if the input density_cube is actually δ
    _check_not_overdensity(np.mean(density_cube))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #                 Step 1. Create Cluster bool Filter
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if clusbool is None:
        clusbool, cluster_thresh, S_clus, f_vir_clus = make_the_clusbool(
            density_cube - 1.0, max_sigs, overdensity_threshold
        )
    else:
        cluster_thresh = None
        S_clus = None
        f_vir_clus = None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #                 Step 2. Create Filament bool Filter
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # we create 3 N^3 arrays with the signature corresponding to each structure
    filament_signature = np.reshape(max_sigs[:, :, :, 1], N_cells)
    wall_signature = np.reshape(max_sigs[:, :, :, 2], N_cells)

    # Isolate Valid Filaments (not clusters)
    not_clus_flat = np.reshape(clusbool == False, N_cells)  # noqa: E712
    filament_valid = filament_signature[not_clus_flat]
    flat_density_cube_valid = np.reshape(density_cube, N_cells)[not_clus_flat]

    # Compute Mass Change Curves and Find Filament Threshold
    S_fil, dM2_fil = calc_mass_change(
        filament_valid, flat_density_cube_valid, Smin, Smax
    )
    ind = np.argmax(dM2_fil)
    filament_thresh = S_fil[ind]
    filbool = (
        max_sigs[:, :, :, 1] > _cast_threshold(filament_thresh, max_sigs.dtype, ">")
    ) & (~clusbool)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #                   Step 3. Create Wall bool Filter
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Isolate Valid Walls (not clusters or filaments)
    wall_valid_filt = (not_clus_flat) & (
        filament_signature < _cast_threshold(filament_thresh, max_sigs.dtype, "<")
    )
    wall_valid = wall_signature[wall_valid_filt]
    wall_density_cube_valid = np.reshape(density_cube, N_cells)[wall_valid_filt]

    # Compute Mass Change Curves and Find Wall Threshold
    S_wall, dM2_wall = calc_mass_change(wall_valid, wall_density_cube_valid, Smin, Smax)
    ind = np.argmax(dM2_wall)
    wall_thresh = S_wall[ind]
    wallbool = (
        max_sigs[:, :, :, 2] > _cast_threshold(wall_thresh, max_sigs.dtype, ">")
    ) & ~(filbool | clusbool)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #                   Step 4. Create Void bool Filter
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    voidbool = (clusbool + filbool + wallbool) == 0

    result = {
        "clusbool": clusbool,
        "filbool": filbool,
        "wallbool": wallbool,
        "voidbool": voidbool,
        "cluster_thresh": cluster_thresh,
        "S_clus": S_clus,
        "f_vir_clus": f_vir_clus,
        "S_fil": S_fil,
        "dM2_fil": dM2_fil,
        "filament_thresh": filament_thresh,
        "S_wall": S_wall,
        "dM2_wall": dM2_wall,
        "wall_thresh": wall_thresh,
    }

    if verbose_flag:
        masks = (clusbool, filbool, wallbool, voidbool)
        total_mass = np.sum(density_cube, dtype=np.float64)
        result["volume_fractions"] = [np.sum(b) / N_cells for b in masks]
        result["mass_fractions"] = [
            np.sum(density_cube[b], dtype=np.float64) / total_mass for b in masks
        ]

    return result


def _report_and_summarize(result):
    """Print the tagging report and return the ``summary_data`` dict of ``calc_structure_bools()``."""
    (
        volume_fraction_clus,
        volume_fraction_fil,
        volume_fraction_wall,
        volume_fraction_void,
    ) = result["volume_fractions"]
    mass_fraction_clus, mass_fraction_fil, mass_fraction_wall, mass_fraction_void = (
        result["mass_fractions"]
    )

    print("---------------------------------------")
    print("     Cluster bool Filter Report    ")
    print("---------------------------------------\n")

    print("Signature Threshold of Clusters:", result["cluster_thresh"])
    print("Volume Fraction of Clusters:", volume_fraction_clus)
    print("Mass Fraction of Clusters:", mass_fraction_clus)
    print("")

    print("---------------------------------------")
    print("     Filament bool Filter Report    ")
    print("---------------------------------------\n")

    print("Signature Threshold of Filaments:", result["filament_thresh"])
    print("Volume Fraction of Filaments:", volume_fraction_fil)
    print("Mass Fraction of Filaments:", mass_fraction_fil)
    print("")

    print("---------------------------------------")
    print("       Wall bool Filter Report      ")
    print("---------------------------------------\n")

    print("Signature Threshold of Walls:", result["wall_thresh"])
    print("Volume Fraction of Walls:", volume_fraction_wall)
    print("Mass Fraction of Walls:", mass_fraction_wall)
    print("")

    print("---------------------------------------")
    print("       Void bool Filter Report      ")
    print("---------------------------------------\n")

    print("Volume Fraction of Voids:", volume_fraction_void)
    print("Mass Fraction of Voids:", mass_fraction_void)
    print("\n---------------------------------------\n")

    return {
        "S_clus": result["S_clus"],
        "f_vir_clus": result["f_vir_clus"],
        "S_fil": result["S_fil"],
        "dM2_fil": result["dM2_fil"],
        "S_fil_thresh": result["filament_thresh"],
        "S_wall": result["S_wall"],
        "dM2_wall": result["dM2_wall"],
        "S_wall_thresh": result["wall_thresh"],
        "mass_fractions": result["mass_fractions"],
        "volume_fractions": result["volume_fractions"],
    }


def calc_structure_bools(
    density_cube,
    max_sigs,
    verbose_flag,
    clusbool=None,
    Smin=-3,
    Smax=2,
    overdensity_threshold=370,
    backend="cpu",
):
    """
    Calculate the boolean filters for clusters, filaments, walls, and voids.

    Args:
        density_cube (:obj:`3D float np.ndarray`):
            the (δ+1 = ρ/<ρ>) ``density_cube``, this is NOT (δ = ρ/<ρ> - 1).
        max_sigs (:obj:`4D float np.ndarray`):
            the maximum signatures array from CosmoMMF.maximum_signature()
        verbose_flag (:obj:`bool`) :
            a flag to allow the function to be more helpful and verbose.
        clusbool (:obj:`bool np.ndarray`, optional):
            the cluster boolean filter
        Smin (:obj:`float`, optional):
            the minimum signature value
        Smax (:obj:`float`, optional):
            the maximum signature value
        overdensity_threshold (:obj:`int`, optional):
            the overdensity parameter threshold for determining virialization
        backend (:obj:`str`, optional):
            ``"cpu"`` uses NumPy and scikit-image. ``"cupy"`` runs the tagging
            on the GPU with CuPy; requires ``pip install 'pycosmommf[cupy]'``.
            With ``"cupy"``, ``density_cube``, ``max_sigs`` and ``clusbool`` may
            also be CuPy arrays or JAX GPU arrays (for example the output of
            ``maximum_signature(..., backend="jax", return_numpy=False)``),
            which are used in place without a copy. Both backends return NumPy
            arrays and give the same filters. Defaults to ``"cpu"``.

    Returns:
        (tuple): a tuple containing:
            - **clusbool** (:obj:`3D bool np.ndarray`): the cluster boolean filter
            - **filbool** (:obj:`3D bool np.ndarray`): the filament boolean filter
            - **wallbool** (:obj:`3D bool np.ndarray`): the wall boolean filter
            - **voidbool** (:obj:`3D bool np.ndarray`): the void boolean filter
            - **summary_data** (:obj:`dict`): a dictionary returned only when ``verbose_flag = True``, it contains the following keys:
                - **S_clus** (:obj:`1D float np.ndarray`): the signature values for clusters
                - **f_vir_clus** (:obj:`1D float np.ndarray`): the virialization fraction for clusters
                - **S_fil** (:obj:`1D float np.ndarray`): the signature values for filaments
                - **dM2_fil** (:obj:`1D float np.ndarray`): the mass change curve for filaments
                - **S_fil_thresh** (:obj:`float`): the threshold value for filaments
                - **S_wall** (:obj:`1D float np.ndarray`): the signature values for walls
                - **dM2_wall** (:obj:`1D float np.ndarray`): the mass change curve for walls
                - **S_wall_thresh** (:obj:`float`): the threshold value for walls
                - **mass_fractions** (:obj:`1D float np.ndarray`): a list of mass fractions for clusters, filaments, walls, and voids
                - **volume_fractions** (:obj:`1D float np.ndarray`): a list of volume fractions for clusters, filaments, walls, and voids
    """

    if backend not in ["cpu", "cupy"]:
        msg = "backend must be either 'cpu' or 'cupy'"
        raise ValueError(msg)

    if backend == "cupy":
        from ._cupy_tagging import structure_bools_cupy

        result = structure_bools_cupy(
            density_cube,
            max_sigs,
            verbose_flag,
            clusbool,
            Smin,
            Smax,
            overdensity_threshold,
        )
    else:
        result = _structure_bools_cpu(
            density_cube,
            max_sigs,
            verbose_flag,
            clusbool,
            Smin,
            Smax,
            overdensity_threshold,
        )

    bools = (
        result["clusbool"],
        result["filbool"],
        result["wallbool"],
        result["voidbool"],
    )

    if verbose_flag:
        return (*bools, _report_and_summarize(result))

    return bools
