# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pycosmommf` is a Python package implementing the NEXUS/NEXUS+ Multiscale Morphological Filter (MMF) algorithm for identifying cosmic web structures (clusters, filaments, walls, voids) in 3D cosmological density fields. It is a Python port of a prior Julia implementation used in [Sunseri et al. 2022](https://ui.adsabs.harvard.edu/abs/2023PhRvD.107b3514S/abstract).

## Commands

**Install for development:**
```bash
pip install -e ".[dev]"
# add the GPU backend (JAX + CuPy; needed for tests/test_jax_backend.py and
# tests/test_cupy_tagging.py to run instead of skip; also needs a CUDA GPU):
pip install -e ".[dev,gpu]"
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
    └─ calc_structure_bools(density_cube, max_sigs, verbose_flag, ..., backend)
           ├─ make_the_clusbool()  — virialization-based cluster threshold
           ├─ calc_mass_change()   — dM²/dlog(S) curve for filament/wall thresholds
           └─ returns (clusbool, filbool, wallbool, voidbool[, summary_data])
```

### Module responsibilities

- **`filter.py`** — k-space Gaussian, log-Gaussian, and top-hat smoothing. `wavevectors3D()` builds the float32 wavevector arrays required by all downstream functions. FFTs use `scipy.fft`, which keeps float32 input in complex64 (numpy<2's `np.fft` always computes in complex128). Numba JIT (`njit(parallel=True, fastmath=True)`) used for the inner filter loops, with float32 constants.
- **`hessian.py`** — computes the scaled Hessian in k-space (6 independent components stored as index 0–5: xx, xy, xz, yy, yz, zz), then IFFTs back. The inner loop is a closure JIT-compiled inside `fast_hessian_from_smoothed`.
- **`signatures.py`** — `signatures_from_hessian()` computes sorted eigenvalues (e1 ≤ e2 ≤ e3) per voxel with `_eigvalsh_3x3_sym`, a scalar Numba port of the JAX backend's analytic Smith-1961 solver (float32 arithmetic, max-abs rescaling), and applies Heaviside-based morphology scores. `maximum_signature()` orchestrates the multi-scale loop, taking the elementwise max across scales.
- **`tagging.py`** — `calc_structure_bools()` sequences the three tagging steps. Cluster threshold uses virialization fraction (fraction of clumps with mean δ > `overdensity_threshold`); filament/wall thresholds use the peak of |dM²/dlog(S)|. Computation (`_structure_bools_cpu`) is separate from the printed report and `summary_data` (`_report_and_summarize`), which both backends share. Clumps are 26-connected (`skimage.measure.label(..., connectivity=3)`); per-clump δ sums come from the serial Numba loop `_clump_sums` in float32. Do not make it a `prange` loop: `+=` on the shared per-clump arrays races and drops voxels. A threshold with no clumps gets a NaN virialized fraction. If no clump is virialized at any threshold, `_select_cluster_threshold` returns `None`, `_warn_unresolved_clusters` prints a warning that the grid is too coarse to resolve virialized clusters, and the cluster filter is empty.
- **`utils.py`** — synthetic test geometry generators (`sphere`, `cylinder`, `wall`) and a block-sum `shrink` utility.

### JAX backend

`maximum_signature()` accepts a `backend` parameter (default `"cpu"`). There are exactly two backends for both `maximum_signature()` and `calc_structure_bools()`. `backend="cpu"` uses only NumPy, SciPy, Numba and scikit-image, and never imports JAX or CuPy. `backend="gpu"` uses JAX for `maximum_signature` and CuPy for the tagging. The old names `"jax"` and `"cupy"` were removed and raise `ValueError`. Install the GPU stack with `pip install 'pycosmommf[gpu]'` (`jax[cuda12]` + `cupy-cuda12x`).

`maximum_signature(..., backend="gpu")` routes the entire computation through `_jax_backend.py` (`maximum_signature_jax`). This module is **not** star-imported in `__init__.py`; it is reached only through the dispatcher, and JAX is imported lazily (via `_jax()` / `_jnp()`) so a missing install raises `ImportError` only when `backend="gpu"` is used. It returns a plain `np.ndarray` by default; `return_numpy=False` returns the JAX array on the GPU instead, for `calc_structure_bools(..., backend="gpu")` to use without a copy.

The hot path is designed around **GPU peak memory**, not a line-by-line port of the CPU code (see the module docstring for the full rationale):

- **Eigenvalues** use the analytic Smith-1961 closed-form solver (`_eigvalsh_3x3_sym`) on the 6 Hessian component arrays directly. Never build a `(N,N,N,3,3)` tensor or call `jnp.linalg.eigh` in the hot path; that was the original OOM cause. The solver rescales by max-abs before the trig formula to stay well-conditioned on near-zero Hessians.
- **Hessian** components flow as 6 separate `(N,N,N)` arrays (`_hessian_components`), not a `jnp.stack`. The public `fast_hessian_from_smoothed_jax` / `signatures_from_hessian_jax` keep the stacked `(N,N,N,6)` / `(N,N,N,3)` layouts for API parity and tests only.
- **Per-scale step** (smooth → Hessian → signatures → running max) is one `jax.jit` graph built by `_make_scale_step`, with `donate_argnums=(0,)` so the `sigmax` buffer is updated in place. The multi-scale loop over `Rs` stays in Python.
- **Tiling**: the signature stage is split along axis 0 (`_choose_n_tiles`: 1 tile for ≤128³, else the largest of 8/4/2 dividing `nx`). The tile loop is a Python loop unrolled at trace time.
- **jax-metal** (Apple) lacks FFT/complex support; `_cpu_device_context()` detects it, warns, and falls back to JAX-on-CPU. Pytest ignores that specific warning in `pyproject.toml`.

Peak GPU memory is roughly 9–10× the scalar field size. At 1024³ XLA still needs ~65 GiB, so it requires an 80 GB GPU. Run it with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`; the default BFC allocator fragments and OOMs.

Both backends use the same Smith-1961 eigensolver, so keep `signatures._eigvalsh_3x3_sym` and `_jax_backend._eigvalsh_3x3_sym` in sync. JAX vs CPU outputs differ at float32 FFT precision (XLA vs SciPy pocketfft, both complex64) and in float32 rounding of the trig formula. This gives ~10⁻⁶ relative differences in smoothing and up to ~10⁻³ absolute in final signatures, which does not affect structure tagging. Test tolerances in `tests/test_jax_backend.py` reflect this; do not tighten them without checking near-degenerate Hessian voxels.

### GPU tagging (CuPy)

`calc_structure_bools(..., backend="gpu")` runs `_cupy_tagging.py` (`structure_bools_cupy`); like `_jax_backend.py` it is not star-imported and CuPy is imported lazily via `_cp()`.

- It mirrors `_structure_bools_cpu` step for step and imports the shared helpers from `tagging.py` (`_cluster_signature_thresholds`, `_select_cluster_threshold`, `_virialized_fraction`, `_warn_unresolved_clusters`, `_log10S_grid`, `_mass_change_curve`, `_check_not_overdensity`). Change those helpers, not copies, so the two backends keep making the same decisions.
- Clumps: `cupyx.scipy.ndimage.label` with `structure=np.ones((3,3,3))` (26-connectivity), then the `_clump_sums_kernel` float32/int32 atomics over **foreground voxels only** (sending background voxels to label 0 serialises the atomics, ~50× slower).
- Mass curves: one fused float32 `ReductionKernel` per threshold, no masks or compacted copies. `_full_sum` reduces per x-plane and then sums the partials, because CuPy runs a reduction to a single scalar on one thread block (~75× slower).
- Inputs may be NumPy (uploaded), CuPy, or JAX GPU arrays (shared through DLPack, zero copy). Outputs are always NumPy bools.
- Memory: while tagging, CuPy allocates from CUDA's stream-ordered pool (`MemoryAsyncPool`, inside `_stream_ordered_allocator`). JAX's `cuda_async` allocator keeps freed memory in that pool, so this is what lets tagging reuse the memory `maximum_signature` freed in the same process. CuPy's default pool cannot. `MemoryAsyncPool` emits an "experimental" `FutureWarning`, which is suppressed where it is created.

CPU and GPU tagging agree up to float32 summation order (pairwise NumPy sums vs GPU tree reductions and atomics), not bit for bit; thresholds matched exactly on the 128³–1024³ validation cubes.

### Key conventions

- **float32 everywhere, on both backends.** Public entry points convert the density (and `max_sigs`) to float32; every array, FFT (complex64), sum and returned number (thresholds, fractions, `summary_data`) stays float32. Do not introduce float64: pass `dtype=np.float32` when creating arrays, cast Python scalars with `np.float32(...)` before they meet array data, and use float32 constants inside Numba kernels (Numba promotes `float32 op int` and Python floats to float64). Because thresholds and signatures are both float32, comparisons are exact and independent of numpy's scalar-promotion rules.

- All input density fields must be **δ+1 = ρ/⟨ρ⟩** (not δ). `maximum_signature` (both backends, via `_validate_density_for_algorithm` in `signatures.py`) raises `ValueError` for any negative voxel when `algorithm="NEXUSPLUS"`, since log-Gauss smoothing would otherwise produce NaN silently. `"NEXUS"` accepts any values. `calc_structure_bools` separately raises `ValueError` if the mean is near zero (i.e., δ was passed).
- Smoothing scales `Rs` are in **voxel units**.
- Hessian components are R²-scaled in k-space: `H_ij = -ki * kj * R² * f̂`.
- Numba JIT functions are decorated with `# pragma: no cover` because they cannot be directly instrumented by coverage tools.
- All source files must start with `from __future__ import annotations` (enforced by ruff `isort.required-imports`).
- Non-ASCII math symbols (ρ, δ, 𝒮, θ, Δ, Μ) are intentionally used in docstrings and variable names; several ruff rules (RUF001–003, PLC2401) are disabled to allow this.

### Testing structure

- `tests/test_unit.py` — tests individual functions (filter, hessian, signatures, utils) with a 32³ synthetic field.
- `tests/test_integration.py` — end-to-end tests of `maximum_signature` and `calc_structure_bools` including the `ValueError` path.
- `tests/test_tagging.py` — CPU tagging: float32 outputs from float64 input, the float32 log10(𝒮) grid, per-clump means on a large clump, NaN fractions for thresholds without clumps, the empty cluster filter + warning when nothing is virialized, rejection of the old backend names, full `calc_structure_bools` with the cluster step.
- `tests/test_cupy_tagging.py` — CuPy labelling vs skimage, and GPU vs CPU tagging (identical thresholds, masks equal up to float32 summation order), with NumPy, CuPy and JAX inputs, plus the empty-cluster warning. Skipped without CuPy or a CUDA GPU.
- `tests/test_jax_backend.py` — JAX shape tests and JAX-vs-CPU agreement tests for each stage. The whole module uses `pytest.importorskip("jax")`, so it is **silently skipped** when JAX is not installed. CI installs only `.[test]` (no JAX), so these tests run only locally; install `.[gpu]` before touching `_jax_backend.py` or `_cupy_tagging.py`.
- `tests/test_package.py` — basic package-level smoke tests.

Pytest is configured with `--strict-markers`, `--strict-config`, and `filterwarnings = error`, so any new warnings will cause test failures. CI runs Python 3.8 and 3.12.
