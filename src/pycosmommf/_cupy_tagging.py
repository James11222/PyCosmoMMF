"""
GPU (CuPy) backend for structure tagging. Requires CuPy to be installed:

    pip install 'pycosmommf[gpu]'

Reached only through ``calc_structure_bools(..., backend="gpu")``; CuPy is
imported lazily so a missing install raises ``ImportError`` only when this
backend is requested.

The computation mirrors ``tagging._structure_bools_cpu`` step for step and
shares its helpers (cluster thresholds and their selection, the log10(𝒮)
grid, mass change curve, report). Everything runs in float32, like the CPU
backend; the two backends agree up to float32 summation-order differences.
The differences are in how each step runs on the GPU:

  * Clumps are labelled with ``cupyx.scipy.ndimage.label`` using the full
    3×3×3 structure (26-connectivity, the ``skimage.measure.label`` default).
    Per-clump δ sums and voxel counts use float32/int32 atomics over the
    foreground voxels only.
  * Each point of a mass change curve is one fused ``ReductionKernel`` pass
    that sums the density over ``valid & (sig > threshold)`` in float32,
    without building a mask or a compacted copy of the arrays.
  * Inputs that already live on the GPU (CuPy arrays, JAX GPU arrays) are
    used in place: JAX arrays are shared through DLPack without a copy.

Memory model
------------
JAX's recommended GPU allocator (``XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async``)
keeps memory it has freed inside CUDA's stream-ordered memory pool. CuPy's
default pool cannot reuse that memory, so after ``maximum_signature`` ran
in the same process most of the GPU would look full. While tagging runs,
CuPy therefore allocates from the same stream-ordered pool
(``cupy.cuda.MemoryAsyncPool``) and releases its unused blocks at the end.

Peak GPU memory is roughly the signatures (3 fields), the density, δ and
the int32 clump labels (one field each) plus a few one-byte masks: about
7× the size of a float32 field, ~29 GiB at 1024³.
"""

from __future__ import annotations

import contextlib
import functools
import warnings

import numpy as np

from .tagging import (
    _check_not_overdensity,
    _cluster_signature_thresholds,
    _log10S_grid,
    _mass_change_curve,
    _select_cluster_threshold,
    _virialized_fraction,
    _warn_unresolved_clusters,
)


def _cp():
    """Lazily import cupy with a user-friendly error."""
    try:
        import cupy

        return cupy
    except ImportError as e:
        msg = (
            "The GPU tagging backend requires CuPy to be installed. "
            "Install it with:  pip install 'pycosmommf[gpu]'"
        )
        raise ImportError(msg) from e


def _to_cupy(x):
    """
    Return ``x`` as a CuPy array, without a copy when it is already on the GPU.

    NumPy arrays are uploaded. JAX GPU arrays are shared through DLPack (after
    waiting for JAX's asynchronous dispatch to finish); JAX CPU arrays and
    anything else array-like are converted through NumPy and uploaded.
    """
    cp = _cp()
    if isinstance(x, cp.ndarray):
        return x
    if isinstance(x, np.ndarray):
        return cp.asarray(x)
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    if hasattr(x, "__dlpack__"):
        with contextlib.suppress(TypeError):  # raised for CPU-resident arrays
            return cp.from_dlpack(x)
    return cp.asarray(np.asarray(x))


@contextlib.contextmanager
def _stream_ordered_allocator():
    """
    Allocate CuPy memory from CUDA's stream-ordered pool inside the block,
    then release the pool's unused blocks. See the module docstring for why.
    Falls back to CuPy's current allocator on devices without memory pools.
    """
    cp = _cp()
    runtime = cp.cuda.runtime
    device = cp.cuda.Device()
    if not runtime.deviceGetAttribute(
        runtime.cudaDevAttrMemoryPoolsSupported, device.id
    ):
        yield
        return

    # CuPy marks MemoryAsyncPool as experimental and warns on every use.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*MemoryAsyncPool is experimental",
            category=FutureWarning,
        )
        pool = cp.cuda.MemoryAsyncPool()
    try:
        with cp.cuda.using_allocator(pool.malloc):
            yield
    finally:
        device.synchronize()
        pool.free_all_blocks()


def _full_sum(kernel, *args):
    """
    Reduce ``kernel`` over every element of its (3D) array arguments and
    return the float32 result.

    CuPy runs a reduction to a single scalar on one thread block, which is
    ~75x slower than reducing each x-plane first (one block per plane) and
    then summing the per-plane partials.
    """
    partials = kernel(*args, axis=tuple(range(1, args[0].ndim)))
    return np.float32(partials.sum().get())


@functools.lru_cache(maxsize=None)
def _masked_mass_above_kernel():
    """float32 sum of ``rho`` over voxels with ``valid & (sig > thresh)``."""
    cp = _cp()
    return cp.ReductionKernel(
        "float32 sig, float32 rho, bool valid, float32 thresh",
        "float32 total",
        "(valid && sig > thresh) ? rho : 0.0f",
        "a + b",
        "total = a",
        "0",
        "pycosmommf_masked_mass_above",
        reduce_type="float",
    )


@functools.lru_cache(maxsize=None)
def _masked_mass_kernel():
    """float32 sum of ``rho`` over voxels where ``mask`` is set."""
    cp = _cp()
    return cp.ReductionKernel(
        "bool mask, float32 rho",
        "float32 total",
        "mask ? rho : 0.0f",
        "a + b",
        "total = a",
        "0",
        "pycosmommf_masked_mass",
        reduce_type="float",
    )


@functools.lru_cache(maxsize=None)
def _clump_sums_kernel():
    """Atomically add each voxel's δ (float32) and a count (int32) to its clump."""
    cp = _cp()
    return cp.ElementwiseKernel(
        "int32 label, float32 d",
        "raw float32 sums, raw int32 counts",
        "atomicAdd(&sums[label - 1], d); atomicAdd(&counts[label - 1], 1);",
        "pycosmommf_clump_sums",
    )


def _make_the_clusbool_cupy(delta, cluster_signature, overdensity_threshold):
    """
    GPU version of ``tagging.make_the_clusbool``. ``delta`` is the δ field and
    ``cluster_signature`` the cluster column of the maximum signatures, both
    float32 CuPy arrays. Returns ``(clusbool, S_th, signature_thresholds,
    virialized_fractions)`` with ``clusbool`` as a CuPy array.
    """
    cp = _cp()
    from cupyx.scipy import ndimage

    structure = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity
    clump_sums = _clump_sums_kernel()
    signature_thresholds = _cluster_signature_thresholds()
    virialized_fractions = np.zeros_like(signature_thresholds)
    flat_delta = delta.ravel()

    for i, S_th in enumerate(signature_thresholds):
        # Identify Clumps with S > S_th
        _clusbool = cluster_signature > S_th
        clumps, n_clumps = ndimage.label(_clusbool, structure=structure)

        if n_clumps == 0:
            virialized_fractions[i] = _virialized_fraction(0, 0)
            continue

        # Only foreground voxels: sending every background voxel through the
        # atomics would serialise them on label 0 (~50x slower).
        foreground = _clusbool.ravel()
        sums = cp.zeros(n_clumps, dtype=cp.float32)
        counts = cp.zeros(n_clumps, dtype=cp.int32)
        clump_sums(clumps.ravel()[foreground], flat_delta[foreground], sums, counts)
        mean_overdensity = sums / counts.astype(cp.float32)
        virialized_fractions[i] = _virialized_fraction(
            int(cp.count_nonzero(mean_overdensity > overdensity_threshold)), n_clumps
        )
        del _clusbool, clumps, foreground, sums, counts, mean_overdensity

    S_th = _select_cluster_threshold(signature_thresholds, virialized_fractions)
    if S_th is None:
        _warn_unresolved_clusters(overdensity_threshold)
        clusbool = cp.zeros(cluster_signature.shape, dtype=bool)
    else:
        clusbool = cluster_signature > S_th

    return clusbool, S_th, signature_thresholds, virialized_fractions


def _mass_change_cupy(sig, rho, valid, Smin, Smax):
    """GPU version of ``tagging.calc_mass_change`` restricted to ``valid`` voxels."""
    kernel = _masked_mass_above_kernel()
    log10S = _log10S_grid(Smin, Smax)
    thresholds = np.power(np.float32(10.0), log10S)
    M = np.array(
        [_full_sum(kernel, sig, rho, valid, t) for t in thresholds], dtype=np.float32
    )
    return _mass_change_curve(log10S, M)


def structure_bools_cupy(
    density_cube,
    max_sigs,
    verbose_flag,
    clusbool,
    Smin,
    Smax,
    overdensity_threshold,
):
    """
    CuPy implementation of ``calc_structure_bools()``. Returns the same dict
    as ``tagging._structure_bools_cpu``, with the boolean filters copied back
    to NumPy.
    """
    cp = _cp()

    with _stream_ordered_allocator():
        rho = _to_cupy(density_cube).astype(cp.float32, copy=False)
        sigs = _to_cupy(max_sigs).astype(cp.float32, copy=False)

        # Check if the input density_cube is actually δ
        _check_not_overdensity(cp.mean(rho).get())

        # Step 1. Cluster bool Filter
        if clusbool is None:
            delta = rho - cp.float32(1.0)
            clusbool, cluster_thresh, S_clus, f_vir_clus = _make_the_clusbool_cupy(
                delta, sigs[..., 0], overdensity_threshold
            )
            del delta
        else:
            clusbool = _to_cupy(clusbool).astype(bool, copy=False)
            cluster_thresh = None
            S_clus = None
            f_vir_clus = None

        # Step 2. Filament bool Filter (valid = not clusters)
        not_clus = ~clusbool
        filament_signature = sigs[..., 1]
        S_fil, dM2_fil = _mass_change_cupy(
            filament_signature, rho, not_clus, Smin, Smax
        )
        filament_thresh = S_fil[np.argmax(dM2_fil)]
        filbool = (filament_signature > filament_thresh) & not_clus

        # Step 3. Wall bool Filter (valid = not clusters or filaments)
        wall_signature = sigs[..., 2]
        wall_valid = not_clus & (filament_signature < filament_thresh)
        del not_clus
        S_wall, dM2_wall = _mass_change_cupy(
            wall_signature, rho, wall_valid, Smin, Smax
        )
        del wall_valid
        wall_thresh = S_wall[np.argmax(dM2_wall)]
        wallbool = (wall_signature > wall_thresh) & ~(filbool | clusbool)

        # Step 4. Void bool Filter
        voidbool = ~(clusbool | filbool | wallbool)

        masks = (clusbool, filbool, wallbool, voidbool)
        result = {
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
            masked_mass = _masked_mass_kernel()
            total_mass = np.float32(cp.sum(rho).get())
            result["volume_fractions"] = [
                np.float32(int(cp.count_nonzero(b))) / np.float32(rho.size)
                for b in masks
            ]
            result["mass_fractions"] = [
                _full_sum(masked_mass, b, rho) / total_mass for b in masks
            ]

        for name, b in zip(("clusbool", "filbool", "wallbool", "voidbool"), masks):
            result[name] = cp.asnumpy(b)

    return result
