from __future__ import annotations

import numba as nb
import numpy as np
import skimage.measure

# Values of the uint8 ``structure_filter`` returned by ``calc_structure_bools()``.
_CLUSTER, _FILAMENT, _WALL, _VOID = 1, 2, 3, 4


def _cluster_signature_thresholds():
    """The float32 cluster-signature thresholds scanned by ``make_the_clusbool()``."""
    return np.geomspace(0.1, 30, 10, dtype=np.float32)  # hard coded for now


@nb.njit
def _clump_sums(labels, delta, n_clumps):  # pragma: no cover
    """
    float32 sum of ``delta`` and voxel count for each clump label ``1..n_clumps``
    (``0`` is background). The loop is serial on purpose: a ``prange`` loop
    with ``+=`` on the shared per-clump arrays races and drops voxels.
    """
    sums = np.zeros(n_clumps, dtype=np.float32)
    counts = np.zeros(n_clumps, dtype=np.int64)
    for i in range(labels.size):
        label = labels[i]
        if label != 0:
            sums[label - 1] += delta[i]
            counts[label - 1] += 1
    return sums, counts


def _clump_mean_overdensity(clumps, n_clumps, delta):
    """Mean δ (float32) of each labelled clump; ``clumps`` holds labels ``1..n_clumps``."""
    sums, counts = _clump_sums(clumps.ravel(), delta.ravel(), n_clumps)
    return sums / counts.astype(np.float32)


def _virialized_fraction(n_virialized, n_clumps):
    """Fraction of clumps that are virialized, as float32; NaN if there are no clumps."""
    if n_clumps == 0:
        return np.float32(np.nan)
    return np.float32(n_virialized) / np.float32(n_clumps)


def _select_cluster_threshold(signature_thresholds, virialized_fractions):
    """
    Pick the threshold whose virialized fraction is closest to 0.5.

    Thresholds without any clumps have a NaN fraction and are skipped. Returns
    ``None`` when no clump is virialized at any threshold (every fraction is 0
    or NaN): the grid does not resolve virialized structures, so there are no
    clusters to tag.
    """
    if not np.any(virialized_fractions > 0):
        return None
    distance = np.abs(virialized_fractions - np.float32(0.5))
    return signature_thresholds[np.nanargmin(distance)]


def _warn_unresolved_clusters(overdensity_threshold):
    """Tell the user why the cluster filter is empty."""
    print(
        "WARNING: no clump has a mean overdensity δ above "
        f"{overdensity_threshold} at any cluster-signature threshold. The "
        "resolution of the grid is not sufficiently high to resolve virialized "
        "clusters/nodes of the cosmic web, so the cluster filter is empty."
    )


def make_the_clusbool(delta, max_sigs, overdensity_threshold):
    """
    A function that creates a bool Filter which selects only
    the clusters in a given cosmological δ field. This function accomplishes
    this by finding a threshold value for 𝒮_cluster by looking at the
    change in virialization fraction as a function of 𝒮_cluster.

    If no clump is virialized at any threshold, the grid does not resolve
    clusters: a warning is printed, ``clusbool`` is empty and ``S_th`` is ``None``.

    Args:
        delta (:obj:`3D float np.ndarray`):
            delta refers to the (δ = ρ / <ρ> - 1) field, this is NOT the (δ + 1 = ρ / <ρ>) field.
        max_sigs (:obj:`4D float np.ndarray`):
            the maximum signatures array from the ``maximum_signature()`` function.
        overdensity_threshold (:obj:`float`):
            The overdensity parameter threshold for determining virialization. Commonly used values
            that are physically motivated can be 370, 200, or 500.

    Returns:
        (tuple): a tuple containing:
            - **clusbool** (:obj:`3D bool np.ndarray`): a boolean filter that selects only the clusters in a given cosmological δ field.
            - **S_th** (:obj:`float32` or ``None``): the threshold value for 𝒮_cluster, ``None`` if no clump is virialized.
            - **signature_thresholds** (:obj:`1D float32 np.ndarray`): an array of threshold values for 𝒮_cluster.
            - **virialized_fractions** (:obj:`1D float32 np.ndarray`): an array of virialized fractions as a function of 𝒮_cluster. Thresholds with no clumps get NaN.
    """
    delta = np.asarray(delta, dtype=np.float32)
    cluster_signature = np.asarray(max_sigs[:, :, :, 0], dtype=np.float32)

    signature_thresholds = _cluster_signature_thresholds()
    virialized_fractions = np.zeros_like(signature_thresholds)

    for i, S_th in enumerate(signature_thresholds):
        # Identify Clumps with S > S_th
        _clusbool = cluster_signature > S_th

        # 0 is background and 1, 2, 3, ... are the clumps (26-connected)
        clumps, n_clumps = skimage.measure.label(
            _clusbool, connectivity=_clusbool.ndim, return_num=True
        )
        averaged_overdensity_per_clump = _clump_mean_overdensity(
            clumps, n_clumps, delta
        )

        virialized_fractions[i] = _virialized_fraction(
            np.count_nonzero(averaged_overdensity_per_clump > overdensity_threshold),
            n_clumps,
        )

    S_th = _select_cluster_threshold(signature_thresholds, virialized_fractions)
    if S_th is None:
        _warn_unresolved_clusters(overdensity_threshold)
        clusbool = np.zeros(cluster_signature.shape, dtype=bool)
    else:
        clusbool = cluster_signature > S_th

    return clusbool, S_th, signature_thresholds, virialized_fractions


def _log10S_grid(Smin, Smax):
    """
    The float32 log10(𝒮) grid on which the mass change curves are evaluated:
    the points of ``np.arange(Smin, Smax + 0.1, 0.1)``, built in float32.
    """
    n = int(np.ceil(round((Smax + 0.1 - Smin) / 0.1, 6)))
    return np.float32(Smin) + np.float32(0.1) * np.arange(n, dtype=np.float32)


def _mass_change_curve(log10S, M):
    """
    Turn the enclosed masses ``M`` (one per ``log10S`` value) into the
    ``|dM²/dlog(𝒮)|`` curve evaluated at the midpoints of ``log10S``.
    """
    # Compute the derivative |dM^2/dlog(𝒮)|
    ΔM_2 = np.abs(np.diff(M**2) / np.diff(log10S))

    # Compute log(𝒮) array used in derivatives (midpoints)
    midx = (log10S[:-1] + log10S[1:]) * np.float32(0.5)
    S = np.power(np.float32(10.0), midx)

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
            - **S** (:obj:`1D float32 np.ndarray`): the signature values
            - **ΔΜ_2** (:obj:`1D float32 np.ndarray`): the mass change curve
    """
    sig_vec = np.asarray(sig_vec, dtype=np.float32)
    density_vec = np.asarray(density_vec, dtype=np.float32)

    # Initialize our arrays
    log10S = _log10S_grid(Smin, Smax)
    thresholds = np.power(np.float32(10.0), log10S)
    M = np.zeros_like(log10S)

    # Sum up all the mass in a structure type as a function of log(𝒮)
    for i in range(len(M)):
        filter = sig_vec > thresholds[i]
        M[i] = np.sum(density_vec[filter])

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
    the uint8 ``structure_filter``, the thresholds and curves behind it and,
    when ``verbose_flag`` is set, the volume and mass fractions.
    """
    density_cube = np.asarray(density_cube, dtype=np.float32)
    max_sigs = np.asarray(max_sigs, dtype=np.float32)
    N_cells = np.prod(density_cube.shape)

    # Check if the input density_cube is actually δ
    _check_not_overdensity(np.mean(density_cube))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #                 Step 1. Create Cluster bool Filter
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if clusbool is None:
        clusbool, cluster_thresh, S_clus, f_vir_clus = make_the_clusbool(
            density_cube - np.float32(1.0), max_sigs, overdensity_threshold
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
    filbool = (max_sigs[:, :, :, 1] > filament_thresh) & (~clusbool)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #                   Step 3. Create Wall bool Filter
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Isolate Valid Walls (not clusters or filaments)
    wall_valid_filt = (not_clus_flat) & (filament_signature < filament_thresh)
    wall_valid = wall_signature[wall_valid_filt]
    wall_density_cube_valid = np.reshape(density_cube, N_cells)[wall_valid_filt]

    # Compute Mass Change Curves and Find Wall Threshold
    S_wall, dM2_wall = calc_mass_change(wall_valid, wall_density_cube_valid, Smin, Smax)
    ind = np.argmax(dM2_wall)
    wall_thresh = S_wall[ind]
    wallbool = (max_sigs[:, :, :, 2] > wall_thresh) & ~(filbool | clusbool)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #       Step 4. Voids are everything else; build the structure filter
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # The three boolean filters are mutually exclusive by construction.
    structure_filter = np.full(density_cube.shape, _VOID, dtype=np.uint8)
    structure_filter[wallbool] = _WALL
    structure_filter[filbool] = _FILAMENT
    structure_filter[clusbool] = _CLUSTER

    result = {
        "structure_filter": structure_filter,
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
        masks = [structure_filter == k for k in (_CLUSTER, _FILAMENT, _WALL, _VOID)]
        total_mass = np.sum(density_cube)
        result["volume_fractions"] = [
            np.float32(np.count_nonzero(b)) / np.float32(N_cells) for b in masks
        ]
        result["mass_fractions"] = [np.sum(density_cube[b]) / total_mass for b in masks]

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
    Tag every voxel as a cluster, filament, wall or void.

    Args:
        density_cube (:obj:`3D float np.ndarray`):
            the (δ+1 = ρ/<ρ>) ``density_cube``, this is NOT (δ = ρ/<ρ> - 1).
            It is converted to float32.
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
            ``"cpu"`` uses NumPy, Numba and scikit-image. ``"gpu"`` runs the
            tagging on the GPU with CuPy; requires ``pip install
            'pycosmommf[gpu]'``. With ``"gpu"``, ``density_cube``, ``max_sigs``
            and ``clusbool`` may also be CuPy arrays or JAX GPU arrays (for
            example the output of ``maximum_signature(..., backend="gpu",
            return_numpy=False)``), which are used in place without a copy.
            Both backends compute in float32 and return a NumPy array; their
            results agree up to float32 summation-order differences. Defaults
            to ``"cpu"``.

    If no clump is virialized at any cluster-signature threshold, the grid is
    too coarse to resolve clusters: a warning is printed and no voxel is
    tagged as a cluster.

    Returns:
        **structure_filter** (:obj:`3D uint8 np.ndarray`) when ``verbose_flag = False``,
        otherwise the tuple ``(structure_filter, summary_data)``:
            - **structure_filter** (:obj:`3D uint8 np.ndarray`): the structure
              type of every voxel: ``1`` = cluster, ``2`` = filament,
              ``3`` = wall, ``4`` = void. Every voxel gets exactly one type,
              so there are no zeros. For example ``structure_filter == 2``
              selects the filaments.
            - **summary_data** (:obj:`dict`): a dictionary returned only when ``verbose_flag = True``, it contains the following keys:
                - **S_clus** (:obj:`1D float32 np.ndarray`): the signature values for clusters
                - **f_vir_clus** (:obj:`1D float32 np.ndarray`): the virialization fraction for clusters
                - **S_fil** (:obj:`1D float32 np.ndarray`): the signature values for filaments
                - **dM2_fil** (:obj:`1D float32 np.ndarray`): the mass change curve for filaments
                - **S_fil_thresh** (:obj:`float32`): the threshold value for filaments
                - **S_wall** (:obj:`1D float32 np.ndarray`): the signature values for walls
                - **dM2_wall** (:obj:`1D float32 np.ndarray`): the mass change curve for walls
                - **S_wall_thresh** (:obj:`float32`): the threshold value for walls
                - **mass_fractions** (:obj:`1D float32 np.ndarray`): a list of mass fractions for clusters, filaments, walls, and voids
                - **volume_fractions** (:obj:`1D float32 np.ndarray`): a list of volume fractions for clusters, filaments, walls, and voids
    """

    if backend not in ["cpu", "gpu"]:
        msg = "backend must be either 'cpu' or 'gpu'"
        raise ValueError(msg)

    if backend == "gpu":
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

    if verbose_flag:
        return result["structure_filter"], _report_and_summarize(result)

    return result["structure_filter"]
