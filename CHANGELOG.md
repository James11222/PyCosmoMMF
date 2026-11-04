# Changelog

All notable changes to `pycosmommf` are documented here. Version numbers come
from git tags (`hatch-vcs`); releases are published on
[PyPI](https://pypi.org/project/pycosmommf/).

## Version 2.0.0

Changes since [1.0.3](https://pypi.org/project/pycosmommf/1.0.3/), the latest
release on PyPI. This release changes the return value of
`calc_structure_bools()` (see Breaking changes below).

### Highlights

- **GPU backend.** `maximum_signature()` and `calc_structure_bools()` take a new
  `backend` argument. `backend="gpu"` runs the signatures with JAX and the
  structure tagging with CuPy. Install it with `pip install 'pycosmommf[gpu]'`.
  The GPU backend is capable of computing the cosmic web structure tags on a
  1024^3 grid in just a couple minutes.
- **Faster CPU signatures.** The Hessian eigenvalues are now computed with an
  analytic closed-form solver instead of a per-voxel `np.linalg.eigvals` call,
  and the FFTs run in complex64. `maximum_signature()` on the CPU is about 4×
  faster at 128³.
- **One structure filter.** `calc_structure_bools()` returns a single `uint8`
  array that labels every voxel, instead of four boolean arrays.
- **Tagging bug fixes.** A race condition in the cluster finder dropped voxels
  from large clumps, and a threshold with no clumps could be selected as the
  cluster threshold. Both are fixed, so cluster filters can differ from 1.0.3.

### Breaking changes

- `calc_structure_bools()` returns one `uint8` array, `structure_filter`, with
  the same shape as the density field. Each voxel holds its structure type: `1`
  = cluster, `2` = filament, `3` = wall, `4` = void. Every voxel has exactly one
  type, so the array has no zeros. With `verbose_flag=True` the function returns
  `(structure_filter, summary_data)`.

  To migrate code written for 1.0.3:

  ```python
  # 1.0.3
  clusbool, filbool, wallbool, voidbool = calc_structure_bools(rho, sigs, False)

  # now
  structure_filter = calc_structure_bools(rho, sigs, False)
  clusbool = structure_filter == 1
  filbool = structure_filter == 2
  wallbool = structure_filter == 3
  voidbool = structure_filter == 4
  ```

  The `uint8` filter uses one byte per voxel. The four boolean arrays used four.

- `maximum_signature(..., algorithm="NEXUSPLUS")` raises `ValueError` if the
  density field has a negative voxel. NEXUS+ smooths `log10(density)`, so a
  negative input (usually an overdensity δ passed instead of δ+1 = ρ/⟨ρ⟩) used
  to fill the whole output with NaN without an error. The error message tells
  you to pass `density + 1`. `algorithm="NEXUS"` still accepts any values.

- All computation and all returned numbers are float32 on both backends. Inputs
  are converted to float32. The thresholds, virialized fractions, mass curves,
  and volume and mass fractions in `summary_data` are float32. In 1.0.3 some
  were float64.

### Added

#### GPU backend

- New `backend` argument for `maximum_signature()` and `calc_structure_bools()`.
  It takes `"cpu"` (default) or `"gpu"`. Any other value raises `ValueError`.
  - `backend="cpu"` uses only NumPy, SciPy, Numba and scikit-image. It never
    imports JAX or CuPy.
  - `backend="gpu"` uses JAX for `maximum_signature()` and CuPy for
    `calc_structure_bools()`.
- New install extra `gpu`: `pip install 'pycosmommf[gpu]'` installs
  `jax[cuda12]>=0.4.30` and `cupy-cuda12x`. JAX and CuPy are imported only when
  `backend="gpu"` is used. If they are missing, you get an `ImportError` that
  gives the install command.
- New `maximum_signature(..., backend="gpu", return_numpy=False)` keeps the
  signatures on the GPU as a JAX array. Pass that array directly to
  `calc_structure_bools(..., backend="gpu")`: CuPy reads it through DLPack
  without a copy. With the default `return_numpy=True`, the result is a NumPy
  array, the same as the CPU backend.
- `calc_structure_bools(..., backend="gpu")` accepts NumPy, CuPy or JAX GPU
  arrays for `density_cube`, `max_sigs` and `clusbool`. It always returns NumPy
  arrays.

Typical GPU use:

```python
import pycosmommf

sigs = pycosmommf.maximum_signature(Rs, rho, backend="gpu", return_numpy=False)
structure_filter = pycosmommf.calc_structure_bools(rho, sigs, False, backend="gpu")
```

How the GPU signatures (`_jax_backend.py`) work:

- The whole per-scale step (smoothing, Hessian, eigenvalues, signatures and
  running maximum) is one `jax.jit` graph. The running maximum buffer is donated
  (`donate_argnums`), so XLA updates it in place.
- The eigenvalues use the same analytic solver as the CPU backend (see
  Performance below). There is no `(N, N, N, 3, 3)` tensor and no call to
  `jnp.linalg.eigh`.
- The six Hessian components stay as six separate arrays, and the signature
  stage is tiled along the first axis for grids larger than 128³.
- Peak GPU memory is about 9–10× the size of one float32 field. A 512³ NEXUS+
  run needs 11.8 GiB and takes 5.4 s on a V100S; a naive JAX port ran out of
  memory on a 32 GB GPU. A 1024³ run, the largest supported grid, needs about 65
  GiB, so it needs an 80 GB GPU. A 40 GB GPU is not enough at 1024³. The backend
  runs on a single GPU.
- On Apple `jax-metal`, which has no FFT or complex support, the backend warns
  and runs JAX on the CPU.

How the GPU tagging (`_cupy_tagging.py`) works:

- It follows the CPU tagging step for step and calls the same helper functions
  for the threshold choices, so both backends make the same decisions.
- Clumps are labelled with `cupyx.scipy.ndimage.label` and 26-connectivity, the
  same connectivity as the CPU backend. Per-clump sums use GPU atomics over the
  foreground voxels only.
- Each point of a mass change curve is one fused float32 reduction kernel, with
  no masks or compacted copies.
- CuPy allocates from CUDA's stream-ordered memory pool while it tags. This lets
  it reuse memory that JAX freed in the same process. Peak GPU memory is about
  7× one float32 field (about 29 GiB at 1024³).
- At 1024³, tagging takes 3.8 s on an A100, against 334 s on the CPU.
- On the 128³, 256³ and 1024³ validation cubes, GPU and CPU tagging gave the
  same thresholds and the same structure filters. In general the two backends
  agree up to float32 summation order, not bit for bit.

Recommended settings for large grids: set `XLA_PYTHON_CLIENT_PREALLOCATE=false`
and `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` before you import JAX. The default
JAX allocator fragments memory and runs out of memory at 1024³.

GPU and CPU signatures differ by about 10⁻⁶ (relative) after smoothing and by up
to a few 10⁻³ (absolute) in the final signatures, because XLA and SciPy use
different float32 FFTs. These differences do not change the structure tagging.

#### Other additions

- If no clump is virialized at any cluster-signature threshold, the cluster
  filter is empty and a warning says that the grid resolution is too low to
  resolve virialized clusters. `make_the_clusbool()` then returns `None` as the
  threshold. In 1.0.3 the function picked a threshold anyway: the lowest one, or
  one with no clumps (see Fixed below).

### Performance

- **Analytic eigenvalues on the CPU.** `signatures_from_hessian()` used to build
  a 3×3 array for every voxel and call `np.linalg.eigvals` and `np.sort` on it.
  It now calls `_eigvalsh_3x3_sym`, a scalar Numba function that uses the
  closed-form trigonometric solution for real symmetric 3×3 matrices (Smith
  1961, _Communications of the ACM_ 4, 168). The function allocates nothing per
  voxel and returns the eigenvalues already sorted. It rescales each matrix by
  its largest absolute entry before it applies the formula, so it stays accurate
  for near-zero Hessians in voids. The CPU and GPU backends use the same solver.
- **complex64 FFTs on the CPU.** Smoothing and the Hessian use `scipy.fft`
  instead of `np.fft`. `scipy.fft` keeps float32 input in complex64. With numpy
  < 2, `np.fft` always computes in complex128.
- **Fewer float64 promotions.** Numba kernels use float32 constants, so they no
  longer promote float32 arithmetic to float64.

CPU timings, 1.0.3 against the current code, on a synthetic log-normal 128³
field with NEXUS+, on one node with 128 CPU cores. Each step was timed on its
second call, so the Numba compilation of module-level functions is excluded:

| Step                                    | 1.0.3  | Now    | Speedup |
| --------------------------------------- | ------ | ------ | ------- |
| `smooth_loggauss()`, 1 scale            | 1.00 s | 0.23 s | 4.3×    |
| `fast_hessian_from_smoothed()`, 1 scale | 4.68 s | 1.02 s | 4.6×    |
| `signatures_from_hessian()`, 1 scale    | 0.49 s | 0.17 s | 2.9×    |
| `maximum_signature()`, 1 scale          | 6.47 s | 1.60 s | 4.0×    |
| `maximum_signature()`, 10 scales        | 58.9 s | 13.8 s | 4.3×    |

At 256³ the current code takes 53 s for 10 scales. The 1.0.3 run at 256³ did not
finish (the process was killed). On an A100 GPU, the same 256³ field takes 1.7 s
for 10 scales with `backend="gpu"` (9.7 s on the first call, which includes JIT
compilation), and the GPU tagging takes 0.08 s against 5.9 s on the CPU, with an
identical structure filter.

### Fixed

- **Race condition in the cluster finder.** The per-clump overdensity sums ran
  in a Numba `prange` loop that added into shared arrays. Threads overwrote each
  other's additions and dropped voxels from large clumps, so the mean
  overdensity of a clump, and therefore the cluster threshold, could be wrong
  and change from run to run. The sums are now a serial float32 loop
  (`_clump_sums`).
- **Thresholds without clumps.** If a cluster-signature threshold had no clumps,
  its virialized fraction was 0/0 = NaN. `np.argmin` returns the first NaN, so
  that empty threshold could be selected as the cluster threshold. These
  fractions are now NaN on purpose, and `np.nanargmin` skips them.
- **Threshold comparisons.** The thresholds were float64 and the signatures were
  float32. Thresholds are now float32, so the comparisons are exact and do not
  depend on NumPy's scalar promotion rules.

### Dependencies and packaging

- `scipy` is now a declared dependency. The CPU FFTs use `scipy.fft`.
- New optional extra `gpu` (`jax[cuda12]>=0.4.30`, `cupy-cuda12x`).
- The `requires-python` upper bound (`<3.13`) is removed. The requirement is now
  `>=3.8`.

### Internal changes

- `calc_structure_bools()` separates the computation from the printed report.
  The CPU and GPU backends share the report and the `summary_data` code.
- `wavevectors3D()` builds the wavevectors directly in float32. The values are
  unchanged.
- `sphere()`, `cylinder()` and `wall()` in `utils.py` return float32 arrays.

### Documentation and tests

- `docs/basic_usage.md` describes the new `structure_filter` return value and
  the GPU backend.
- New `tests/test_tagging.py` (CPU tagging), `tests/test_jax_backend.py` (JAX vs
  CPU agreement for each stage) and `tests/test_cupy_tagging.py` (CuPy vs CPU
  tagging with NumPy, CuPy and JAX inputs). The GPU tests skip when JAX, CuPy or
  a CUDA GPU is not available, so CI runs only the CPU tests.

## 1.0.3 and earlier

See the [GitHub releases](https://github.com/James11222/pycosmommf/releases).
