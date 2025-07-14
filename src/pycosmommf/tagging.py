from __future__ import annotations

import numba as nb
import numpy as np
import skimage.measure

jit_compiler = nb.njit(parallel=True, fastmath=True)


def make_the_clusbool(delta, max_sigs, overdensity_threshold):  # pragma: no cover
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
            - **virialized_fractions** (:obj:`1D float np.ndarray`): an array of virialized fractions as a function of 𝒮_cluster.
    """

    signature_thresholds = np.geomspace(0.1, 30, 10)  # hard coded for now
    virialized_fractions = np.zeros_like(signature_thresholds)

    @jit_compiler
    def calc_virialized_fraction(clumps):
        nx, ny, nz = clumps.shape

        total_clumps_detected = np.max(clumps)
        summed_overdensity_per_clump = np.zeros(total_clumps_detected)
        num_cells_per_clump = np.zeros(total_clumps_detected)

        for i in nb.prange(nx):
            for j in range(ny):
                for k in range(nz):
                    if clumps[i, j, k] != 0:
                        clump_id = int(
                            clumps[i, j, k]
                        )  # clump_id is the clump number ranging from 1 to total_clumps_detected
                        summed_overdensity_per_clump[clump_id - 1] += delta[i, j, k]
                        num_cells_per_clump[clump_id - 1] += 1

        return summed_overdensity_per_clump / num_cells_per_clump

    for i, S_th in enumerate(signature_thresholds):
        # Identify Clumps with S > S_th
        _clusbool = max_sigs[:, :, :, 0] > S_th

        clumps = skimage.measure.label(
            _clusbool.astype(int)
        )  # an array where 0 is background and 1, 2, 3, ... are the clumps
        averaged_overdensity_per_clump = calc_virialized_fraction(clumps)

        virialized_fractions[i] = np.sum(
            averaged_overdensity_per_clump > overdensity_threshold
        ) / np.max(clumps)

    S_th = signature_thresholds[np.abs(virialized_fractions - 0.5).argmin()]
    clusbool = max_sigs[:, :, :, 0] > S_th

    return clusbool, S_th, signature_thresholds, virialized_fractions


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
    log10S = np.arange(Smin, Smax + 0.1, 0.1)
    M = np.zeros_like(log10S)

    # Sum up all the mass in a structure type as a function of log(𝒮)
    for i in range(len(M)):
        filter = sig_vec > 10 ** log10S[i]
        M[i] = np.sum(density_vec[filter])

    # Compute the derivative |dM^2/dlog(𝒮)|
    ΔM_2 = np.abs(np.diff(M**2) / np.diff(log10S))

    # Compute log(𝒮) array used in derivatives (midpoints)
    midx = (log10S[:-1] + log10S[1:]) / 2
    S = 10**midx

    return S, ΔM_2


def calc_structure_bools(
    density_cube,
    max_sigs,
    verbose_flag,
    clusbool=None,
    Smin=-3,
    Smax=2,
    overdensity_threshold=370,
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

    N_cells = np.prod(density_cube.shape)

    # Check if the input density_cube is actually δ
    if np.isclose(np.abs(np.mean(density_cube)), 0, atol=1e-3):
        msg = "make sure that you are not inputing the overdensity field δ = ρ/<ρ> - 1, but rather δ + 1 = ρ/<ρ>."
        raise ValueError(msg)

    # # Check if the input density_cube is actually δ + 1 = ρ/<ρ> or just ρ
    # if np.mean(density_cube) != 1.0:
    #     # If the mean is greater than 1, we assume it is simply ρ, but we want to transform to δ + 1 = ρ/<ρ>
    #     if verbose_flag:  # pragma: no cover
    #         print(
    #             "Input density_cube is in density units, transforming to δ + 1 = ρ/<ρ>."
    #         )
    #     density_cube /= np.mean(density_cube)  # normalize to mean = 1

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #                 Step 1. Create Cluster bool Filter
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if clusbool is None:  # pragma: no cover
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
    #                   Step 4. Create Void bool Filter
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    voidbool = (clusbool + filbool + wallbool) == 0

    if verbose_flag:
        print("---------------------------------------")
        print("     Cluster bool Filter Report    ")
        print("---------------------------------------\n")

        volume_fraction_clus = np.sum(clusbool) / N_cells
        mass_fraction_clus = np.sum(density_cube[clusbool]) / np.sum(density_cube)

        print("Signature Threshold of Clusters:", cluster_thresh)
        print("Volume Fraction of Clusters:", volume_fraction_clus)
        print("Mass Fraction of Clusters:", mass_fraction_clus)
        print("")

        print("---------------------------------------")
        print("     Filament bool Filter Report    ")
        print("---------------------------------------\n")

        volume_fraction_fil = np.sum(filbool) / N_cells
        mass_fraction_fil = np.sum(density_cube[filbool]) / np.sum(density_cube)

        print("Signature Threshold of Filaments:", filament_thresh)
        print("Volume Fraction of Filaments:", volume_fraction_fil)
        print("Mass Fraction of Filaments:", mass_fraction_fil)
        print("")

        print("---------------------------------------")
        print("       Wall bool Filter Report      ")
        print("---------------------------------------\n")

        volume_fraction_wall = np.sum(wallbool) / N_cells
        mass_fraction_wall = np.sum(density_cube[wallbool]) / np.sum(density_cube)

        print("Signature Threshold of Walls:", wall_thresh)
        print("Volume Fraction of Walls:", volume_fraction_wall)
        print("Mass Fraction of Walls:", mass_fraction_wall)
        print("")

        print("---------------------------------------")
        print("       Void bool Filter Report      ")
        print("---------------------------------------\n")

        volume_fraction_void = np.sum(voidbool) / N_cells
        mass_fraction_void = np.sum(density_cube[voidbool]) / np.sum(density_cube)

        print("Volume Fraction of Voids:", volume_fraction_void)
        print("Mass Fraction of Voids:", mass_fraction_void)
        print("\n---------------------------------------\n")

        summary_data = {
            "S_clus": S_clus,
            "f_vir_clus": f_vir_clus,
            "S_fil": S_fil,
            "dM2_fil": dM2_fil,
            "S_fil_thresh": filament_thresh,
            "S_wall": S_wall,
            "dM2_wall": dM2_wall,
            "S_wall_thresh": wall_thresh,
            "mass_fractions": [
                mass_fraction_clus,
                mass_fraction_fil,
                mass_fraction_wall,
                mass_fraction_void,
            ],
            "volume_fractions": [
                volume_fraction_clus,
                volume_fraction_fil,
                volume_fraction_wall,
                volume_fraction_void,
            ],
        }

        return clusbool, filbool, wallbool, voidbool, summary_data

    return clusbool, filbool, wallbool, voidbool
