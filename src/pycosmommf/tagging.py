from __future__ import annotations

import functools

import numba as nb
import numpy as np
import skimage.measure

jit_compiler = nb.njit(fastmath=True)


def make_the_clusbool(delta, max_sigs, verbose, Δ):
    """
    make_the_clusbool() - Documentation

    A function that creates a Boolean Filter which selects only
    the clusters in a given cosmological δ field. This function accomplishes
    this by finding a threshold value for 𝒮_cluster by looking at the
    change in virialization fraction as a function of 𝒮_cluster.

    Arguments:

    delta - [3D Float Array] - delta refers to the δ delta.
    max_sigs - [4D Float Array] - the maximum signatures array from CosmoMMF.maximum_signature()
    verbose - [Boolean] - a flag to allow the function to be more helpful and verbose.
    Δ - [Int] - The overdensity parameter threshold for determining virialization. This
    parameter comes from a paper by Gunn & Gott on spherical collapse. Commonly
    used values that are physically motivated can be 370, 200, or 500 (for R_200 or R_500).
    """

    signature_thresholds = np.geomspace(0.1, 30, 10)  # hard coded for now
    virialized_fractions = np.zeros_like(signature_thresholds)

    def check_virialized(j, delta, clumps, Δ):
        δ_ave = np.mean(delta[clumps == j])
        return int(δ_ave > Δ)

    for i, S_th in enumerate(signature_thresholds):
        # Identify Clumps with S > S_th
        _clusbool = max_sigs[:, :, :, 0] > S_th
        clumps = skimage.measure.label(
            _clusbool
        )  # an array where 0 is background and 1, 2, 3, ... are the clumps
        total_clumps_detected = np.max(clumps)

        num_clumps_virialized = np.sum(
            list(
                map(
                    functools.partial(
                        check_virialized, delta=delta, clumps=clumps, Δ=Δ
                    ),
                    range(1, total_clumps_detected + 1),
                )
            )
        )  # number of clumps that are virialized

        # Calculate the number of virialized clumps
        virialized_fractions[i] = num_clumps_virialized / total_clumps_detected

    S_th = signature_thresholds[np.abs(virialized_fractions - 0.5).argmin()]
    clusbool = max_sigs[:, :, :, 0] > S_th

    if verbose:
        return clusbool, S_th, signature_thresholds, virialized_fractions
    return clusbool


def calc_mass_change(sig_vec, delta_vec, Smin, Smax):
    """
    Calculate the mass change curve for a given structure type.

    Arguments:
    -----------
        sig_vec - [1D Float Array] - the signature values for a given structure type
        delta_vec - [1D Float Array] - the delta values for a given structure type
        Smin - [Float] - the minimum signature value
        Smax - [Float] - the maximum signature value

    Returns:
    -----------
        S - [1D Float Array] - the signature values
        ΔM_2 - [1D Float Array] - the mass change curve
    """
    # Initialize our arrays
    log10S = np.arange(Smin, Smax + 0.1, 0.1)
    M = np.zeros_like(log10S)

    # Sum up all the mass in a structure type as a function of log(𝒮)
    for i in range(len(M)):
        filter = sig_vec > 10 ** log10S[i]
        M[i] = np.sum(delta_vec[filter])

    # Compute the derivative |dM^2/dlog(𝒮)|
    ΔM_2 = np.abs(np.diff(M**2) / np.diff(log10S))

    # Compute log(𝒮) array used in derivatives (midpoints)
    midx = (log10S[:-1] + log10S[1:]) / 2
    S = 10**midx

    return S, ΔM_2


def calc_structure_bools(
    delta, max_sigs, verbose, clusbool=None, Smin=-3, Smax=2, Δ=370
):
    """
    Calculate the boolean filters for clusters, filaments, walls, and voids.

    Arguments:
    -----------
        delta - [3D Float Array] - the δ delta
        max_sigs - [4D Float Array] - the maximum signatures array from CosmoMMF.maximum_signature()
        verbose - [Boolean] - a flag to allow the function to be more helpful and verbose.
        clusbool - [3D Boolean Array] - the cluster boolean filter
        Smin - [Float] - the minimum signature value
        Smax - [Float] - the maximum signature value
        Δ - [Int] - the overdensity parameter threshold for determining virialization

    Returns:
    -----------
        clusbool - [3D Boolean Array] - the cluster boolean filter
        filbool - [3D Boolean Array] - the filament boolean filter
        wallbool - [3D Boolean Array] - the wall boolean filter
        voidbool - [3D Boolean Array] - the void boolean filter

        VERBOSE ONLY:
            summary_data - [Dict] - a dictionary containing the following keys:
                S_clus - [1D Float Array] - the signature values for clusters
                f_vir_clus - [1D Float Array] - the virialization fraction for clusters
                S_fil - [1D Float Array] - the signature values for filaments
                dM2_fil - [1D Float Array] - the mass change curve for filaments
                S_wall - [1D Float Array] - the signature values for walls
                dM2_wall - [1D Float Array] - the mass change curve for walls

    """

    N_cells = np.prod(delta.shape)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #                 Step 1. Create Cluster Boolean Filter
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if clusbool is None:
        if verbose:
            clusbool, cluster_thresh, S_clus, f_vir_clus = make_the_clusbool(
                delta, max_sigs, verbose, Δ
            )
        else:
            clusbool = make_the_clusbool(delta, max_sigs, verbose, Δ)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #                 Step 2. Create Filament Boolean Filter
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # we create 3 N^3 arrays with the signature corresponding to each structure
    filament_signature = np.reshape(max_sigs[:, :, :, 1], N_cells)
    wall_signature = np.reshape(max_sigs[:, :, :, 2], N_cells)

    # Isolate Valid Filaments (not clusters)
    not_clus_flat = np.reshape(clusbool == False, N_cells)  # noqa: E712
    filament_valid = filament_signature[not_clus_flat]
    flat_delta_valid = np.reshape(delta, N_cells)[not_clus_flat]

    # Compute Mass Change Curves and Find Filament Threshold
    S_fil, dM2_fil = calc_mass_change(filament_valid, flat_delta_valid, Smin, Smax)
    ind = np.argmax(dM2_fil)
    filament_thresh = S_fil[ind]
    filbool = (max_sigs[:, :, :, 1] > filament_thresh) & (~clusbool)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #                   Step 3. Create Wall Boolean Filter
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Isolate Valid Walls (not clusters or filaments)
    wall_valid_filt = (not_clus_flat) & (filament_signature < filament_thresh)
    wall_valid = wall_signature[wall_valid_filt]
    wall_delta_valid = np.reshape(delta, N_cells)[wall_valid_filt]

    # Compute Mass Change Curves and Find Wall Threshold
    S_wall, dM2_wall = calc_mass_change(wall_valid, wall_delta_valid, Smin, Smax)
    ind = np.argmax(dM2_wall)
    wall_thresh = S_wall[ind]
    wallbool = (max_sigs[:, :, :, 2] > wall_thresh) & ~(filbool | clusbool)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #                   Step 4. Create Void Boolean Filter
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    voidbool = (clusbool + filbool + wallbool) == 0

    if verbose:
        print("---------------------------------------")
        print("     Cluster Boolean Filter Report    ")
        print("---------------------------------------\n")

        volume_fraction = np.sum(clusbool) / N_cells
        mass_fraction = np.sum(delta[clusbool]) / np.sum(delta)

        print("Signature Threshold of Clusters:", cluster_thresh)
        print("Volume Fraction of Clusters:", volume_fraction)
        print("Mass Fraction of Clusters:", mass_fraction)
        print("")

        print("---------------------------------------")
        print("     Filament Boolean Filter Report    ")
        print("---------------------------------------\n")

        volume_fraction = np.sum(filbool) / N_cells
        mass_fraction = np.sum(delta[filbool]) / np.sum(delta)

        print("Signature Threshold of Filaments:", filament_thresh)
        print("Volume Fraction of Filaments:", volume_fraction)
        print("Mass Fraction of Filaments:", mass_fraction)
        print("")

        print("---------------------------------------")
        print("       Wall Boolean Filter Report      ")
        print("---------------------------------------\n")

        volume_fraction = np.sum(wallbool) / N_cells
        mass_fraction = np.sum(delta[wallbool]) / np.sum(delta)

        print("Signature Threshold of Walls:", wall_thresh)
        print("Volume Fraction of Walls:", volume_fraction)
        print("Mass Fraction of Walls:", mass_fraction)
        print("")

        print("---------------------------------------")
        print("       Void Boolean Filter Report      ")
        print("---------------------------------------\n")

        volume_fraction = np.sum(voidbool) / N_cells
        mass_fraction = np.sum(delta[voidbool]) / np.sum(delta)

        print("Volume Fraction of Voids:", volume_fraction)
        print("Mass Fraction of Voids:", mass_fraction)
        print("\n---------------------------------------\n")

        summary_data = {
            "S_clus": S_clus,
            "f_vir_clus": f_vir_clus,
            "S_fil": S_fil,
            "dM2_fil": dM2_fil,
            "S_wall": S_wall,
            "dM2_wall": dM2_wall,
        }

        return clusbool, filbool, wallbool, voidbool, summary_data

    return clusbool, filbool, wallbool, voidbool
