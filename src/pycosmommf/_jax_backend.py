"""
JAX backend for pycosmommf. Requires jax to be installed:

    pip install 'pycosmommf[jax]'

All public functions mirror the CPU API but operate on JAX arrays.
The output of maximum_signature_jax() is always a numpy ndarray so the
rest of the package (tagging, etc.) remains unchanged.
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


def fast_hessian_from_smoothed_jax(f_Rn, R_S, kv):
    """
    Compute the R²-scaled Hessian of ``f_Rn`` in k-space using JAX.

    The six independent components (xx, xy, xz, yy, yz, zz) are computed
    simultaneously via broadcasting, avoiding the explicit voxel loop used
    in the CPU backend.

    Args:
        f_Rn (:obj:`3D float np.ndarray` or JAX array):
            The smoothed real-space field.
        R_S (:obj:`float`):
            Smoothing scale used to produce ``f_Rn`` (sets the R² prefactor).
        kv (:obj:`tuple`):
            Wavevectors ``(kx, ky, kz)`` from ``wavevectors3D()``.

    Returns:
        (:obj:`4D float32 JAX array`): Shape ``(nx, ny, nz, 6)``.
    """
    jnp = _jnp()
    kx, ky, kz = (jnp.asarray(k) for k in kv)
    f_hat = jnp.fft.fftn(jnp.asarray(f_Rn))

    KX = kx[:, None, None]
    KY = ky[None, :, None]
    KZ = kz[None, None, :]

    scale = -(R_S**2) * f_hat  # common factor for all components

    def _ifftn_real(arr):
        return jnp.fft.ifftn(arr).real

    hessian = jnp.stack(
        [
            _ifftn_real(KX * KX * scale),  # 0: H_xx
            _ifftn_real(KX * KY * scale),  # 1: H_xy
            _ifftn_real(KX * KZ * scale),  # 2: H_xz
            _ifftn_real(KY * KY * scale),  # 3: H_yy
            _ifftn_real(KY * KZ * scale),  # 4: H_yz
            _ifftn_real(KZ * KZ * scale),  # 5: H_zz
        ],
        axis=-1,
    )
    return hessian.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


def signatures_from_hessian_jax(hessian):
    """
    Compute cluster, filament, and wall signatures from the Hessian using JAX.

    The three eigenvalues of the symmetric 3×3 Hessian at each voxel are
    obtained via a single batched ``jnp.linalg.eigh`` call, which is far more
    efficient than the per-voxel loop in the CPU backend and maps well to GPU
    SIMD execution.

    Args:
        hessian (:obj:`4D float np.ndarray` or JAX array):
            Shape ``(nx, ny, nz, 6)`` — output of
            ``fast_hessian_from_smoothed_jax()``.

    Returns:
        (:obj:`4D float32 JAX array`): Shape ``(nx, ny, nz, 3)``.
            Last axis: ``[cluster, filament, wall]``.
    """
    jnp = _jnp()
    h = jnp.asarray(hessian)

    # Build symmetric (nx, ny, nz, 3, 3) tensor from the 6 stored components.
    # Component ordering matches the CPU hessian: 0=xx,1=xy,2=xz,3=yy,4=yz,5=zz
    H = jnp.stack(
        [
            jnp.stack([h[..., 0], h[..., 1], h[..., 2]], axis=-1),
            jnp.stack([h[..., 1], h[..., 3], h[..., 4]], axis=-1),
            jnp.stack([h[..., 2], h[..., 4], h[..., 5]], axis=-1),
        ],
        axis=-2,
    )  # (nx, ny, nz, 3, 3)

    # eigh returns real, sorted-ascending eigenvalues for symmetric matrices.
    # This is mathematically equivalent to np.sort(np.real(np.linalg.eigvals(...)))
    # used in the CPU backend.
    eigvals = jnp.linalg.eigh(H)[0]  # (nx, ny, nz, 3)
    e1 = eigvals[..., 0]  # smallest
    e2 = eigvals[..., 1]
    e3 = eigvals[..., 2]  # largest

    # Guard against division by zero (matches the `if e1 == 0` branch in CPU).
    zero_mask = e1 == 0.0
    safe_e1 = jnp.where(zero_mask, 1.0, e1)

    ratio_e3_e1 = jnp.abs(e3 / safe_e1)
    ratio_e2_e1 = jnp.abs(e2 / safe_e1)

    # Heaviside θ(−x) = (x < 0)
    th_e1 = (e1 < 0).astype(jnp.float32)
    th_e2 = (e2 < 0).astype(jnp.float32)
    th_e3 = (e3 < 0).astype(jnp.float32)

    # xθ(x) = max(0, x)
    xth_1_minus_e3e1 = jnp.maximum(0.0, 1.0 - ratio_e3_e1)
    xth_1_minus_e2e1 = jnp.maximum(0.0, 1.0 - ratio_e2_e1)

    clus = (ratio_e3_e1 * jnp.abs(e3)) * (th_e1 * th_e2 * th_e3)
    fil = (ratio_e2_e1 * xth_1_minus_e3e1) * (jnp.abs(e2) * th_e1 * th_e2)
    wall = (xth_1_minus_e2e1 * xth_1_minus_e3e1) * (jnp.abs(e1) * th_e1)

    # Zero out voxels where e1 was exactly 0
    clus = jnp.where(zero_mask, 0.0, clus)
    fil = jnp.where(zero_mask, 0.0, fil)
    wall = jnp.where(zero_mask, 0.0, wall)

    return jnp.stack([clus, fil, wall], axis=-1).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def maximum_signature_jax(Rs, density_cube, algorithm="NEXUSPLUS", eps=1e-16):
    """
    JAX implementation of ``maximum_signature()``.

    Computes the maximum structure signatures across all smoothing scales in
    ``Rs`` using JAX, which can run on GPU/TPU when available. The multi-scale
    loop runs on the Python side; each per-scale computation is fully vectorised
    inside JAX (no explicit voxel loops).

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
        (:obj:`4D float32 np.ndarray`): Shape ``(nx, ny, nz, 3)``.
    """
    from .filter import wavevectors3D

    jnp = _jnp()

    nx, ny, nz = density_cube.shape

    with _cpu_device_context():
        field = jnp.asarray(density_cube, dtype=jnp.float32) + eps

        wave_vecs = wavevectors3D((nx, ny, nz))
        sigmax = jnp.ones((nx, ny, nz, 3), dtype=jnp.float32) * eps

        for R in Rs:
            if algorithm == "NEXUS":
                f_Rn = smooth_gauss_jax(field, R, wave_vecs)
            else:
                f_Rn = smooth_loggauss_jax(field, R, wave_vecs)

            H_Rn = fast_hessian_from_smoothed_jax(f_Rn, R, wave_vecs)
            sigs_Rn = signatures_from_hessian_jax(H_Rn)
            sigmax = jnp.maximum(sigmax, sigs_Rn)

        return np.asarray(sigmax)
