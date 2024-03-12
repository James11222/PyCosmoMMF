
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from numba import njit


def make_the_clusbool(data, max_sigs, verbose, Δ):
    """
    make_the_clusbool() - Documentation
    
    A function that creates a Boolean Filter which selects only
    the clusters in a given cosmological 1+δ density cube. This function accomplishes
    this by finding a threshold value for 𝒮_cluster by looking at the
    change in virialization fraction as a function of 𝒮_cluster. 
    
    Note: Grid Resolution is important here. The resolution of data and max_sigs should
    be sufficiently high that a voxel's physical size is < 1 Mpc/h so clusters can be resolved.
    
    Arguments: 
    
    data - [3D Float Array] - data refers to the δ+1 data.
    max_sigs - [4D Float Array] - the maximum signatures array from CosmoMMF.maximum_signature()
    verbose - [Boolean] - a flag to allow the function to be more helpful and verbose.
    Δ - [Int] - The overdensity parameter threshold for determining virialization. This
    parameter comes from a paper by Gunn & Gott on spherical collapse. Commonly
    used values that are physically motivated can be 370, 200, or 500 (for R_200 or R_500).
    """
    
    # if verbose: print("---------------------------------------")
    # if verbose: print("   Creating a Cluster Boolean Filter   ")
    # if verbose: print("---------------------------------------\n")
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #         Step 1. Create the Initial Cluster Boolean Filter 
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    # if verbose: print("Step 1: Creating the Initial Cluster Boolean Filter...")
    
    nx, ny, nz = data.shape
    total_volume = nx * ny * nz
    
    # bins = 10 ** np.array(np.linspace(-5, 2, num=nx))
    bins = np.logspace(-5, 2, nx)
    
    hist, bin_edges = np.histogram(max_sigs[:,:,:,0], bins)
    Smin = bin_edges[0:nx-1][np.argmax(hist)] 
    
    initial_clusbool = (max_sigs[:,:,:,0] > Smin)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #               Step 2. Identify Potential Candidates 
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    # if verbose: print("Step 2: Identifying Potential Candidates...")
    
    components = label(initial_clusbool)
    total_clusters_detected = np.max(components)
    
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #                Step 3. Summarize Candidate Cluster Statistics 
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # if verbose: print("Step 3: Summarizing Candidate Cluster Statistics...")
    
    density = np.zeros(total_clusters_detected + 1)
    volume = np.zeros(total_clusters_detected + 1)
    signatures = np.zeros(total_clusters_detected + 1)

    @njit
    def perform_loop(density, volume, signatures, data, max_sigs):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if components[i,j,k] != 0:
                        signatures[components[i,j,k]] += max_sigs[i,j,k,0]
                        density[components[i,j,k]] += data[i,j,k]
                        volume[components[i,j,k]] += 1.0

    perform_loop(density, volume, signatures, data, max_sigs)
    
    ave_signatures = signatures / volume
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #     Step 4. Find the Signature Threshold via Virialization Curve 
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    # if verbose: print("Step 4: Finding the Signature Threshold via Virialization Curve...")
    
    Ss = 10 ** np.array(np.linspace(round(np.log10(Smin)), 1, num=500))
    
    y = np.zeros(500)
    for i, s_th in enumerate(Ss):
        y[i] = np.sum((density > Δ) & (ave_signatures > s_th)) / np.sum(ave_signatures > s_th)
    
    index = np.argmin(np.abs(y - 0.5))
    S_th = Ss[index]
    
    if verbose:
        plt.figure()
        plt.plot(Ss, y, color="red", linewidth=3)
        plt.hlines(y=0.5, xmin=Smin, xmax=1e1, color="Gray", linestyle="--", alpha=0.8)
        plt.vlines(x=S_th, ymin=0, ymax=1, color="Gray", linestyle="--", alpha=0.8)
        plt.xlabel("Average Cluster Signature")
        plt.ylabel("Fraction of Valid Clusters")
        plt.xlim(Smin, 1e1)
        plt.ylim(0, 1)
        plt.semilogx()
        plt.show()
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #                Step 5. Refine the Cluster Boolean Filter
    #                    to Meet Virialization Requirement
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    # if verbose: print("Step 5: Refining the Cluster Boolean Filter to Meet Virialization Requirement...")
    
    clusbool = max_sigs[:,:,:,0] > S_th
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #               Step 6. Summary Statistics Report and Return 
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    # if verbose: print("Step 6: Summarizing Statistics Report...\n")
    
    if verbose:
        volume_fraction = np.sum(clusbool) / total_volume
        mass_fraction = np.sum(data[clusbool]) / np.sum(data)
        
        print("---------------------------------------")
        print("     Cluster Boolean Filter Report     ")
        print("---------------------------------------\n")
        
        print("Signature Threshold of Clusters:", S_th)
        print("Fraction of Valid Clusters:", np.round(y[index], 4))
        print("Volume Fraction of Clusters:", np.round(volume_fraction, 4))
        print("Mass Fraction of Clusters:", np.round(mass_fraction, 4))
        print("")
    
    return clusbool

def calc_mass_change(sig_vec, data_vec, Smin, Smax):
    """
    Calculate the mass change curve for a given structure type.

    Arguments:
    -----------
        sig_vec - [1D Float Array] - the signature values for a given structure type
        data_vec - [1D Float Array] - the data values for a given structure type
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
        M[i] = np.sum(data_vec[filter])
    
    # Compute the derivative |dM^2/dlog(𝒮)|
    ΔM_2 = np.abs(np.diff(M ** 2) / np.diff(log10S))
    
    # Compute log(𝒮) array used in derivatives (midpoints)
    midx = (log10S[:-1] + log10S[1:]) / 2
    S = 10 ** midx
    
    return S, ΔM_2

def calc_structure_bools(data, max_sigs, verbose, clusbool=None, Smin=-3, Smax=2, Δ=370):
    """
    Calculate the boolean filters for clusters, filaments, walls, and voids.

    Arguments:
    -----------
        data - [3D Float Array] - the δ+1 data
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
        S_fil - [1D Float Array] - the signature values for filaments
        dM2_fil - [1D Float Array] - the mass change curve for filaments
        S_wall - [1D Float Array] - the signature values for walls
        dM2_wall - [1D Float Array] - the mass change curve for walls

    """
    
    N_cells = np.prod(data.shape)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #                 Step 1. Create Cluster Boolean Filter
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    if clusbool is None:
        clusbool = make_the_clusbool(data, max_sigs, verbose, Δ)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #                 Step 2. Create Filament Boolean Filter
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # we create 3 N^3 arrays with the signature corresponding to each structure
    cluster_signature = np.reshape(max_sigs[:,:,:,0], N_cells)
    filament_signature = np.reshape(max_sigs[:,:,:,1], N_cells)
    wall_signature = np.reshape(max_sigs[:,:,:,2], N_cells)
    
    #Isolate Valid Filaments (not clusters)
    not_clus_flat = np.reshape( clusbool == False, N_cells )
    filament_valid = filament_signature[ not_clus_flat ] 
    flat_data_valid = np.reshape(data, N_cells)[ not_clus_flat ]

    #Compute Mass Change Curves and Find Filament Threshold
    S_fil, dM2_fil = calc_mass_change(filament_valid, flat_data_valid, Smin, Smax)
    ind = np.argmax(dM2_fil) 
    filament_thresh = S_fil[ind]
    filbool = (max_sigs[:,:,:,2] > filament_thresh) & (~clusbool)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #                   Step 3. Create Wall Boolean Filter
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    #Isolate Valid Walls (not clusters or filaments)
    wall_valid_filt = (not_clus_flat) & (filament_signature < filament_thresh)
    wall_valid = wall_signature[wall_valid_filt]
    wall_data_valid = np.reshape(data, N_cells)[wall_valid_filt]

    #Compute Mass Change Curves and Find Wall Threshold
    S_wall, dM2_wall = calc_mass_change(wall_valid, wall_data_valid, Smin, Smax)
    ind = np.argmax(dM2_wall)
    wall_thresh = S_wall[ind]
    wallbool = (max_sigs[:,:,:,2] > wall_thresh) & ~(filbool | clusbool)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #                   Step 4. Create Void Boolean Filter
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    voidbool = (clusbool + filbool + wallbool) == 0
    
    
    if verbose:
        
        print("---------------------------------------")
        print("     Filament Boolean Filter Report    ")
        print("---------------------------------------\n")
        
        volume_fraction = np.sum(filbool) / N_cells
        mass_fraction = np.sum(data[filbool]) / np.sum(data)
        
        print("Signature Threshold of Filaments:", filament_thresh)
        print("Volume Fraction of Filaments:", volume_fraction)
        print("Mass Fraction of Filaments:", mass_fraction)
        print("")


        print("---------------------------------------")
        print("       Wall Boolean Filter Report      ")
        print("---------------------------------------\n")
        
        volume_fraction = np.sum(wallbool) / N_cells
        mass_fraction = np.sum(data[wallbool]) / np.sum(data)
    
        print("Signature Threshold of Walls:", wall_thresh)
        print("Volume Fraction of Walls:", volume_fraction)
        print("Mass Fraction of Walls:", mass_fraction)  
        print("")
        
        print("---------------------------------------")
        print("       Void Boolean Filter Report      ")
        print("---------------------------------------\n")
        
        volume_fraction = np.sum(voidbool) / N_cells
        mass_fraction = np.sum(data[voidbool]) / np.sum(data)
    
        print("Volume Fraction of Voids:", volume_fraction)
        print("Mass Fraction of Voids:", mass_fraction) 
        print("\n---------------------------------------\n")
        
        
        # Plot the Mass Change Curves
        plt.figure()
        
        plt.plot(S_fil, dM2_fil / np.max(dM2_fil), color="blue", 
            linewidth=3, label="Filament")
        plt.plot(S_fil, dM2_wall / np.max(dM2_wall), color="green", 
            linewidth=3, label="Wall")
        
        plt.vlines(x=filament_thresh, ymin = 0, ymax = 1, 
            color="blue", linestyle="--", linewidth=3)
        plt.vlines(x=wall_thresh, ymin = 0, ymax = 1, 
            color="green", linestyle="--", linewidth=3)
        
        plt.xlabel("Signature Strength")
        plt.ylabel("ΔΜ^2 (arbitrary units)")
        plt.semilogx()
        plt.legend()
        plt.show()
        
        return clusbool, filbool, wallbool, voidbool, S_fil, dM2_fil, S_wall, dM2_wall
    
    else:
        
        return clusbool, filbool, wallbool, voidbool


