"""
JAX backend for pycosmommf. Requires jax to be installed:

    pip install 'pycosmommf[jax]'

All public functions mirror the CPU API but operate on JAX arrays.
The output of maximum_signature_jax() is always a numpy ndarray so the
rest of the package (tagging, etc.) remains unchanged.

Memory model
------------
The naive translation of the CPU pipeline blows up on GPU because:

  * ``signatures_from_hessian`` (CPU) is a per-voxel loop. A direct JAX port
    materialises a ``(N,N,N,3,3)`` tensor and calls ``jnp.linalg.eigh``,
    which also allocates eigenvectors. Peak ≈ 25–30× the scalar field size.
  * Building the Hessian with ``jnp.stack`` keeps all 6 components live at
    once *and* requires a stack temporary.
  * Running the multi-scale loop un-jitted prevents XLA from reusing
    buffers across iterations.

This backend instead:

  * computes eigenvalues with the analytic Smith-1961 trigonometric formula
    for real symmetric 3×3 matrices — operates on the 6 component arrays
    directly, no ``(...,3,3)`` tensor, no eigenvector workspace,
  * processes the eigenvalue / signature stage in tiles along axis 0,
  * JITs the per-scale step with ``donate_argnums=(sigmax,)`` so XLA
    reuses the running-max buffer.

Peak GPU memory is ~9–10× the scalar field size, vs ~25–30× before.
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np


def _jnp():
    """Lazily import jax.numpy with a user-friendly error."""
    try:
        import jax.numpy as jnp

        return jnp
    except ImportError as e:
        msg = (
            "The JAX backend requires JAX to be installed. "
            "Install it with:  pip install 'pycosmommf[jax]'"
        )
        raise ImportError(msg) from e


def _jax():
    """Lazily import jax with a user-friendly error."""
    try:
        import jax

        return jax
    except ImportError as e:
        msg = (
            "The JAX backend requires JAX to be installed. "
            "Install it with:  pip install 'pycosmommf[jax]'"
        )
        raise ImportError(msg) from e


def _cpu_device_context():
    """Return a context manager that forces JAX to run on CPU.

    jax-metal (Apple Silicon) does not support complex-number operations or
    FFT, which this pipeline requires throughout. When Metal is the default
    backend we automatically fall back to the CPU device so the JAX backend
    remains usable on macOS. A one-time warning is emitted.
    """
    try:
        import jax

        default = jax.devices()[0]
        # Metal devices report as 'METAL' or contain 'metal' in their string repr
        if "metal" in str(default).lower() or "METAL" in str(type(default)):
            warnings.warn(
                "jax-metal does not support FFT/complex operations required by "
                "this pipeline. Falling back to JAX on CPU automatically. "
                "To silence this warning set backend='cpu'.",
                stacklevel=3,
            )
            cpu_devices = jax.devices("cpu")
            return jax.default_device(cpu_devices[0])
    except Exception:
        pass
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def smooth_gauss_jax(f, R_S, kv):
    """
    Apply a Gaussian filter to field ``f`` using JAX.

    Args:
        f (:obj:`3D float np.ndarray` or JAX array):
            The field to be smoothed.
        R_S (:obj:`float`):
            Smoothing scale in voxel units.
        kv (:obj:`tuple`):
            Wavevectors ``(kx, ky, kz)`` from ``wavevectors3D()``.

    Returns:
        (:obj:`3D float32 JAX array`): The smoothed field.
    """
    jnp = _jnp()
    kx, ky, kz = (jnp.asarray(k) for k in kv)
    f = jnp.asarray(f)

    KX = kx[:, None, None]
    KY = ky[None, :, None]
    KZ = kz[None, None, :]

    GF = jnp.exp(-(KX**2 + KY**2 + KZ**2) * R_S**2 / 2)
    f_Rn = jnp.fft.ifftn(GF * jnp.fft.fftn(f)).real
    f_Rn = f_Rn * (jnp.sum(f) / jnp.sum(f_Rn))
    return f_Rn.astype(jnp.float32)


def smooth_loggauss_jax(f, R_S, kv):
    """
    Apply a Gaussian filter to ``log10(f)`` using JAX.

    Args:
        f (:obj:`3D float np.ndarray` or JAX array):
            The field to be smoothed (must be strictly positive).
        R_S (:obj:`float`):
            Smoothing scale in voxel units.
        kv (:obj:`tuple`):
            Wavevectors ``(kx, ky, kz)`` from ``wavevectors3D()``.

    Returns:
        (:obj:`3D float32 JAX array`): The smoothed field.
    """
    jnp = _jnp()
    kx, ky, kz = (jnp.asarray(k) for k in kv)
    f = jnp.asarray(f)

    KX = kx[:, None, None]
    KY = ky[None, :, None]
    KZ = kz[None, None, :]

    GF = jnp.exp(-(KX**2 + KY**2 + KZ**2) * R_S**2 / 2)
    f_log = jnp.log10(f)
    f_Rn = jnp.fft.ifftn(GF * jnp.fft.fftn(f_log)).real
    f_result = (10.0**f_Rn) * (jnp.sum(f) / jnp.sum(10.0**f_Rn))
    return f_result.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Hessian
# ---------------------------------------------------------------------------


def _hessian_components(f_Rn, R_S, KX, KY, KZ):
    """
    Compute the six independent components of the R²-scaled Hessian of
    ``f_Rn`` and return them as a tuple of six real ``(N,N,N)`` arrays.

    This is the building block used by the memory-efficient pipeline — it
    never materialises a stacked ``(N,N,N,6)`` tensor. ``KX``, ``KY``, ``KZ``
    are pre-shaped (broadcast-ready) wavevector arrays.

    Returns ``(h_xx, h_xy, h_xz, h_yy, h_yz, h_zz)`` as float32 arrays.
    """
    jnp = _jnp()
    f_hat = jnp.fft.fftn(f_Rn)
    scale = -(R_S**2) * f_hat  # common factor

    # Each component: multiply by ki*kj broadcast, IFFT, take real part.
    # Inside a jit these are issued sequentially; XLA can free each complex
    # intermediate before the next IFFT.
    h_xx = jnp.fft.ifftn(KX * KX * scale).real.astype(jnp.float32)
    h_xy = jnp.fft.ifftn(KX * KY * scale).real.astype(jnp.float32)
    h_xz = jnp.fft.ifftn(KX * KZ * scale).real.astype(jnp.float32)
    h_yy = jnp.fft.ifftn(KY * KY * scale).real.astype(jnp.float32)
    h_yz = jnp.fft.ifftn(KY * KZ * scale).real.astype(jnp.float32)
    h_zz = jnp.fft.ifftn(KZ * KZ * scale).real.astype(jnp.float32)
    return h_xx, h_xy, h_xz, h_yy, h_yz, h_zz


def fast_hessian_from_smoothed_jax(f_Rn, R_S, kv):
    """
    Compute the R²-scaled Hessian of ``f_Rn`` in k-space using JAX.

    Returned in the same ``(N,N,N,6)`` packed-component layout used by the
    CPU backend: index 0=xx, 1=xy, 2=xz, 3=yy, 4=yz, 5=zz.

    .. note::
       The full-pipeline entry point ``maximum_signature_jax`` does **not**
       go through this function — it uses the unstacked
       :func:`_hessian_components` helper to avoid the memory cost of
       carrying the stacked tensor through the signature step. This
       function is kept as a public API for backward compatibility and
       direct/test use.

    Args:
        f_Rn (:obj:`3D float np.ndarray` or JAX array):
            The smoothed real-space field.
        R_S (:obj:`float`):
            Smoothing scale used to produce ``f_Rn`` (sets the R² prefactor).
        kv (:obj:`tuple`):
            Wavevectors ``(kx, ky, kz)`` from ``wavevectors3D()``.

    Returns:
        (:obj:`4D float32 JAX array`): Shape ``(N, N, N, 6)``.
    """
    jnp = _jnp()
    kx, ky, kz = (jnp.asarray(k) for k in kv)
    KX = kx[:, None, None]
    KY = ky[None, :, None]
    KZ = kz[None, None, :]
    h_xx, h_xy, h_xz, h_yy, h_yz, h_zz = _hessian_components(
        jnp.asarray(f_Rn), R_S, KX, KY, KZ
    )
    return jnp.stack([h_xx, h_xy, h_xz, h_yy, h_yz, h_zz], axis=-1)


# ---------------------------------------------------------------------------
# Analytic 3×3 symmetric eigenvalues (Smith 1961)
# ---------------------------------------------------------------------------


def _eigvalsh_3x3_sym(a, b, c, d, e, f):
    """
    Eigenvalues of the real symmetric 3×3 matrix

        [[a, b, c],
         [b, d, e],
         [c, e, f]]

    given each entry as a (potentially batched) JAX array. Returns
    ``(e_min, e_mid, e_max)`` sorted ascending, all float32, broadcast to
    the common shape of the inputs.

    Uses the closed-form trigonometric solution of Smith (1961)
    (Communications of the ACM 4, 168) — no iterative solver, no
    eigenvectors, no 3×3 tensor ever built. This trades the LAPACK call
    for a handful of pointwise float ops, which is both faster and
    dramatically lower memory than ``jnp.linalg.eigh`` on a ``(N,N,N,3,3)``
    batch.

    Numerical notes:
      * The argument to ``arccos`` is clipped to [-1, 1] to absorb tiny
        float rounding excursions outside the valid range.
      * The matrix is rescaled by its max-abs element before the trig
        formula runs. This keeps the determinant-over-p³ quotient
        well-conditioned in regions where the matrix is near-zero or
        near-degenerate (cosmological background voxels) — without the
        rescale, the cubed denominator amplifies float32 noise into
        visible signature error.
      * The all-zero matrix (``scale == 0``) is short-circuited to zero
        eigenvalues, so we never divide by zero.
    """
    jnp = _jnp()

    # Rescale to keep p^3 well-conditioned at small matrix magnitudes.
    # Eigenvalues of (M / s) are eigenvalues(M) / s, so we scale back at the end.
    scale = jnp.maximum(
        jnp.maximum(
            jnp.maximum(jnp.abs(a), jnp.abs(d)),
            jnp.maximum(jnp.abs(f), jnp.abs(b)),
        ),
        jnp.maximum(jnp.abs(c), jnp.abs(e)),
    )
    all_zero = scale == 0.0
    safe_scale = jnp.where(all_zero, 1.0, scale)
    inv_scale = 1.0 / safe_scale

    a_s = a * inv_scale
    b_s = b * inv_scale
    c_s = c * inv_scale
    d_s = d * inv_scale
    e_s = e * inv_scale
    f_s = f * inv_scale

    p1 = b_s * b_s + c_s * c_s + e_s * e_s
    q = (a_s + d_s + f_s) / 3.0
    a_q = a_s - q
    d_q = d_s - q
    f_q = f_s - q
    p2 = a_q * a_q + d_q * d_q + f_q * f_q + 2.0 * p1
    p = jnp.sqrt(p2 / 6.0)

    det_shifted = (
        a_q * (d_q * f_q - e_s * e_s)
        - b_s * (b_s * f_q - e_s * c_s)
        + c_s * (b_s * e_s - d_q * c_s)
    )

    # Scalar*I case (now in the rescaled matrix): eigenvalues are all q.
    degenerate = p == 0.0
    safe_p = jnp.where(degenerate, 1.0, p)
    r = det_shifted / (2.0 * safe_p**3)
    r = jnp.clip(r, -1.0, 1.0)

    phi = jnp.arccos(r) / 3.0
    two_p = 2.0 * p
    eig_max = q + two_p * jnp.cos(phi)
    eig_min = q + two_p * jnp.cos(phi + 2.0 * jnp.pi / 3.0)
    # Trace conservation gives the middle eigenvalue without another trig call.
    eig_mid = 3.0 * q - eig_max - eig_min

    eig_max = jnp.where(degenerate, q, eig_max)
    eig_min = jnp.where(degenerate, q, eig_min)
    eig_mid = jnp.where(degenerate, q, eig_mid)

    # Scale back; zero-matrix voxels get zero eigenvalues regardless.
    eig_min = jnp.where(all_zero, 0.0, eig_min * scale)
    eig_mid = jnp.where(all_zero, 0.0, eig_mid * scale)
    eig_max = jnp.where(all_zero, 0.0, eig_max * scale)

    return (
        eig_min.astype(jnp.float32),
        eig_mid.astype(jnp.float32),
        eig_max.astype(jnp.float32),
    )


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


def _signatures_from_eigs(e1, e2, e3):
    """
    Cluster / filament / wall signatures from the three sorted eigenvalues
    ``e1 <= e2 <= e3`` (each a JAX array of the same shape). Returns three
    arrays ``(clus, fil, wall)`` of the same shape.

    Matches the formulas used in the CPU ``signatures_from_hessian`` loop.
    """
    jnp = _jnp()

    # Guard against the (extremely rare) e1 == 0 voxel.
    zero_mask = e1 == 0.0
    safe_e1 = jnp.where(zero_mask, 1.0, e1)

    abs_e1 = jnp.abs(e1)
    abs_e2 = jnp.abs(e2)
    abs_e3 = jnp.abs(e3)

    ratio_e3_e1 = jnp.abs(e3 / safe_e1)
    ratio_e2_e1 = jnp.abs(e2 / safe_e1)

    th_e1 = (e1 < 0).astype(jnp.float32)
    th_e2 = (e2 < 0).astype(jnp.float32)
    th_e3 = (e3 < 0).astype(jnp.float32)

    xth_1_minus_e3e1 = jnp.maximum(0.0, 1.0 - ratio_e3_e1)
    xth_1_minus_e2e1 = jnp.maximum(0.0, 1.0 - ratio_e2_e1)

    clus = (ratio_e3_e1 * abs_e3) * (th_e1 * th_e2 * th_e3)
    fil = (ratio_e2_e1 * xth_1_minus_e3e1) * (abs_e2 * th_e1 * th_e2)
    wall = (xth_1_minus_e2e1 * xth_1_minus_e3e1) * (abs_e1 * th_e1)

    clus = jnp.where(zero_mask, 0.0, clus)
    fil = jnp.where(zero_mask, 0.0, fil)
    wall = jnp.where(zero_mask, 0.0, wall)

    return (
        clus.astype(jnp.float32),
        fil.astype(jnp.float32),
        wall.astype(jnp.float32),
    )


def signatures_from_hessian_jax(hessian):
    """
    Compute cluster, filament, and wall signatures from the Hessian using JAX.

    The three eigenvalues of the symmetric 3×3 Hessian at each voxel are
    obtained from an analytic closed-form formula
    (:func:`_eigvalsh_3x3_sym`) applied directly to the six stored
    components — no ``(N,N,N,3,3)`` tensor is built and no
    ``jnp.linalg.eigh`` is called. This is what makes the JAX backend
    tractable for large grids; ``eigh`` on a 1024³ batch would need on the
    order of 70 GiB just for its workspace.

    Args:
        hessian (:obj:`4D float np.ndarray` or JAX array):
            Shape ``(N, N, N, 6)`` — output of
            ``fast_hessian_from_smoothed_jax()``. Component ordering
            ``0=xx, 1=xy, 2=xz, 3=yy, 4=yz, 5=zz``.

    Returns:
        (:obj:`4D float32 JAX array`): Shape ``(N, N, N, 3)``.
            Last axis: ``[cluster, filament, wall]``.
    """
    jnp = _jnp()
    h = jnp.asarray(hessian)
    e1, e2, e3 = _eigvalsh_3x3_sym(
        h[..., 0], h[..., 1], h[..., 2], h[..., 3], h[..., 4], h[..., 5]
    )
    clus, fil, wall = _signatures_from_eigs(e1, e2, e3)
    return jnp.stack([clus, fil, wall], axis=-1).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Tiled signature stage — used inside the full pipeline
# ---------------------------------------------------------------------------


def _signature_tile_update(sigmax_tile, h_xx, h_xy, h_xz, h_yy, h_yz, h_zz):
    """
    Update one axis-0 slab of ``sigmax`` with the running max of the new
    signatures computed (analytically) from the corresponding slab of the
    Hessian components. Returns the updated slab.

    All inputs are tiles (slabs) of the same axis-0 size. The function is
    pure (no in-place mutation) so it can be safely jitted.
    """
    jnp = _jnp()
    e1, e2, e3 = _eigvalsh_3x3_sym(h_xx, h_xy, h_xz, h_yy, h_yz, h_zz)
    clus, fil, wall = _signatures_from_eigs(e1, e2, e3)
    sigs_tile = jnp.stack([clus, fil, wall], axis=-1).astype(jnp.float32)
    return jnp.maximum(sigmax_tile, sigs_tile)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def _choose_n_tiles(nx, ny, nz):
    """
    Pick a tile count along axis 0 that keeps the per-tile signature
    intermediates well under one scalar-field worth of memory.

    The signature stage produces a handful of (tile_nx, ny, nz) f32 temporaries
    in addition to the (tile_nx, ny, nz, 3) signature tile. With 8 tiles each
    temp is ~1/8 of a field; that has been comfortable on a 32 GiB V100 at
    N=512 and on a 40 GiB A100 at N=1024. Smaller grids don't need tiling.
    """
    # No need to tile small grids — overhead outweighs the savings.
    if nx * ny * nz <= 128**3:
        return 1
    # Pick the largest divisor-of-nx that is <= 8 (avoids ragged last tile).
    for n in (8, 4, 2, 1):
        if nx % n == 0:
            return n
    return 1


def _make_scale_step(nx, ny, nz, kv_jax, algorithm, n_tiles):
    """
    Build a single JIT-compiled function that, given the running ``sigmax``
    buffer and the (already-on-device) ``field``, performs one smoothing
    scale's worth of work and returns an updated ``sigmax``.

    The entire per-scale pipeline (smoothing → Hessian → tiled signature
    update → elementwise max) is one jitted graph so that:

      * ``sigmax`` is donated and the running max can be applied in place,
      * XLA can fuse the analytic-eigenvalue temporaries into single kernel
        passes (most never materialise as full arrays),
      * the Python tile loop unrolls at trace time, giving XLA the full
        graph to plan buffer reuse across tiles.

    The closure captures the wavevector arrays so they aren't re-uploaded
    per scale; a separate jitted callable is built per algorithm + tile
    count.
    """
    jax = _jax()
    jnp = _jnp()

    kx, ky, kz = kv_jax
    KX = kx[:, None, None]
    KY = ky[None, :, None]
    KZ = kz[None, None, :]

    smooth = smooth_loggauss_jax if algorithm == "NEXUSPLUS" else smooth_gauss_jax

    def step_impl(sigmax, field, R):
        f_Rn = smooth(field, R, (kx, ky, kz))
        h_xx, h_xy, h_xz, h_yy, h_yz, h_zz = _hessian_components(
            f_Rn, R, KX, KY, KZ
        )
        # f_Rn is now dead in the JAX graph; XLA can free its buffer
        # before the signature stage runs.

        if n_tiles == 1:
            return _signature_tile_update(
                sigmax, h_xx, h_xy, h_xz, h_yy, h_yz, h_zz
            )

        # Python-side tile loop, unrolled at trace time. Each iteration
        # produces an updated slab; XLA sees the full graph and (with
        # donation) places these slabs back into the original sigmax
        # buffer rather than allocating a fresh full-volume array.
        tile = nx // n_tiles
        for i in range(n_tiles):
            s = slice(i * tile, (i + 1) * tile)
            updated = _signature_tile_update(
                sigmax[s],
                h_xx[s],
                h_xy[s],
                h_xz[s],
                h_yy[s],
                h_yz[s],
                h_zz[s],
            )
            sigmax = sigmax.at[s].set(updated)
        return sigmax

    return jax.jit(step_impl, donate_argnums=(0,))


def maximum_signature_jax(Rs, density_cube, algorithm="NEXUSPLUS", eps=1e-16):
    """
    JAX implementation of ``maximum_signature()``.

    Computes the maximum structure signatures across all smoothing scales in
    ``Rs`` using JAX, which can run on GPU/TPU when available. The multi-scale
    loop runs on the Python side; each per-scale computation is JIT-compiled
    and tiled along axis 0 so peak GPU memory stays at roughly 9–10× the
    scalar field size rather than the 25–30× of a naive implementation.

    The output is converted back to a NumPy array before returning so the
    result is a drop-in replacement for the CPU backend output.

    Args:
        Rs (:obj:`list` of :obj:`float`):
            Smoothing scales in voxel units.
        density_cube (:obj:`3D float np.ndarray`):
            The (δ+1 = ρ/⟨ρ⟩) density field.
        algorithm (:obj:`str`, optional):
            ``"NEXUS"`` (Gaussian) or ``"NEXUSPLUS"`` (log-Gaussian). Defaults
            to ``"NEXUSPLUS"``.
        eps (:obj:`float`, optional):
            Floor added to the field to avoid log(0). Defaults to ``1e-16``.

    Returns:
        (:obj:`4D float32 np.ndarray`): Shape ``(N, N, N, 3)``.

    Raises:
        ValueError: If ``algorithm="NEXUSPLUS"`` and ``density_cube``
            contains negative voxels (the log-Gauss smoothing requires a
            strictly positive field).
    """
    # Lazy import to avoid pulling signatures.py at module import time.
    from .filter import wavevectors3D
    from .signatures import _validate_density_for_algorithm

    _validate_density_for_algorithm(density_cube, algorithm)

    jnp = _jnp()

    nx, ny, nz = density_cube.shape

    with _cpu_device_context():
        field = jnp.asarray(density_cube, dtype=jnp.float32) + eps
        kv_np = wavevectors3D((nx, ny, nz))
        kv_jax = tuple(jnp.asarray(k) for k in kv_np)
        sigmax = jnp.full((nx, ny, nz, 3), eps, dtype=jnp.float32)

        n_tiles = _choose_n_tiles(nx, ny, nz)
        step = _make_scale_step(nx, ny, nz, kv_jax, algorithm, n_tiles)

        for R in Rs:
            sigmax = step(sigmax, field, float(R))

        return np.asarray(sigmax)
