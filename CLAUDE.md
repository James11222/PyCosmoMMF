# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pycosmommf` is a Python package implementing the NEXUS/NEXUS+ Multiscale Morphological Filter (MMF) algorithm for identifying cosmic web structures (clusters, filaments, walls, voids) in 3D cosmological density fields. It is a Python port of a prior Julia implementation used in [Sunseri et al. 2022](https://ui.adsabs.harvard.edu/abs/2023PhRvD.107b3514S/abstract).

## Commands

**Install for development:**
```bash
pip install -e ".[dev]"
```

**Run tests:**
```bash
pytest
# or via nox:
nox -s tests
```

**Run a single test:**
```bash
pytest tests/test_unit.py::test_fast_hessian_from_smoothed
```

**Lint (ruff + pre-commit):**
```bash
nox -s lint
# or directly:
pre-commit run --all-files
```

**Build docs:**
```bash
nox -s docs
```

**Build package:**
```bash
nox -s build
```

## Architecture

The package exposes everything through `src/pycosmommf/__init__.py` via star imports from four core modules. The public API flows through two top-level functions: `maximum_signature()` and `calc_structure_bools()`.

### Pipeline

```
density_cube (3D float32, δ+1 = ρ/<ρ>)
    └─ maximum_signature(Rs, density_cube, algorithm)
           ├─ filter.py: smooth_gauss / smooth_loggauss  (per smoothing scale R)
           ├─ hessian.py: fast_hessian_from_smoothed      (6-component symmetric Hessian)
           ├─ signatures.py: signatures_from_hessian      (cluster/filament/wall scores)
           └─ returns sigmax (nx, ny, nz, 3)  — max over all scales
    └─ calc_structure_bools(density_cube, max_sigs, verbose_flag, ...)
           ├─ make_the_clusbool()  — virialization-based cluster threshold
           ├─ calc_mass_change()   — dM²/dlog(S) curve for filament/wall thresholds
           └─ returns (clusbool, filbool, wallbool, voidbool[, summary_data])
```

### Module responsibilities

- **`filter.py`** — k-space Gaussian, log-Gaussian, and top-hat smoothing. `wavevectors3D()` builds the wavevector arrays required by all downstream functions. Numba JIT (`njit(parallel=True, fastmath=True)`) used for the inner filter loops.
- **`hessian.py`** — computes the scaled Hessian in k-space (6 independent components stored as index 0–5: xx, xy, xz, yy, yz, zz), then IFFTs back. The inner loop is a closure JIT-compiled inside `fast_hessian_from_smoothed`.
- **`signatures.py`** — `signatures_from_hessian()` sorts eigenvalues (e1 ≤ e2 ≤ e3) and applies Heaviside-based morphology scores. `maximum_signature()` orchestrates the multi-scale loop, taking the elementwise max across scales.
- **`tagging.py`** — `calc_structure_bools()` sequences the three tagging steps. Cluster threshold uses virialization fraction (fraction of clumps with mean δ > `overdensity_threshold`); filament/wall thresholds use the peak of |dM²/dlog(S)|.
- **`utils.py`** — synthetic test geometry generators (`sphere`, `cylinder`, `wall`) and a block-sum `shrink` utility.

### JAX backend

`maximum_signature()` accepts a `backend` parameter (default `"cpu"`). Setting `backend="jax"` routes the entire computation through `_jax_backend.py`, which replaces all Numba JIT loops with fully-vectorised JAX operations:

- Smoothing filter: `kx[:,None,None]` broadcasting instead of a Numba loop
- Hessian: all 6 components computed with a single `jnp.stack` of IFFT calls
- Signatures: batched `jnp.linalg.eigh` on a `(nx,ny,nz,3,3)` tensor instead of a per-voxel loop

The JAX backend always returns a plain `np.ndarray` so `calc_structure_bools()` is unchanged. Install with `pip install 'pycosmommf[jax]'`. JAX is imported lazily — missing it raises `ImportError` only when `backend="jax"` is actually used.

Float32 FFT rounding differs slightly between XLA (JAX) and FFTW (NumPy), leading to ~1.2×10⁻⁴ relative differences in smoothing and ~1.3×10⁻³ absolute differences in final signatures. This is within float32 FFT precision and does not affect structure tagging.

### Key conventions

- All input density fields must be **δ+1 = ρ/⟨ρ⟩** (not δ). `calc_structure_bools` raises `ValueError` if the mean is near zero (i.e., δ was passed).
- Smoothing scales `Rs` are in **voxel units**.
- Hessian components are R²-scaled in k-space: `H_ij = -ki * kj * R² * f̂`.
- Numba JIT functions are decorated with `# pragma: no cover` because they cannot be directly instrumented by coverage tools.
- All source files must start with `from __future__ import annotations` (enforced by ruff `isort.required-imports`).
- Non-ASCII math symbols (ρ, δ, 𝒮, θ, Δ, Μ) are intentionally used in docstrings and variable names; several ruff rules (RUF001–003, PLC2401) are disabled to allow this.

### Testing structure

- `tests/test_unit.py` — tests individual functions (filter, hessian, signatures, utils) with a 32³ synthetic field.
- `tests/test_integration.py` — end-to-end tests of `maximum_signature` and `calc_structure_bools` including the `ValueError` path.
- `tests/test_package.py` — basic package-level smoke tests.

Pytest is configured with `--strict-markers`, `--strict-config`, and `filterwarnings = error`, so any new warnings will cause test failures.
