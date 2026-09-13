# Benchmarks

Crossover thresholds and their measurements. The source carries one-line
pointers here; this file carries the numbers, the hardware, and the commands.

## Hardware and toolchain of record

- CPU: Intel Core Ultra 7 258V (Lunar Lake, 8 cores)
- rustc: stable (2026-08), `--release`
- Fields: GF(2^8) (`Gf8`, AES polynomial) unless noted.

## Karatsuba dispatch crossover (`KARATSUBA_CROSSOVER`)

`Polynomial::multiply` dispatches schoolbook → Karatsuba at
`min(left, right) ≥ KARATSUBA_CROSSOVER = 2048` coefficients.

Criterion, `multiply/{schoolbook,karatsuba}/<n>`, GF(2^8), equal operands:

| Coefficients | Schoolbook | Karatsuba | Ratio |
| --- | --- | --- | --- |
| 128 | 0.89 µs | 1.82 µs | 2.04 |
| 512 | 6.60 µs | 8.79 µs | 1.33 |
| 1024 | 14.7 µs | 17.9 µs | 1.22 |
| 2048 | 53.5 µs | 34.1 µs | 0.64 |
| 3072 | 113.8 µs | 124.6 µs | 1.09 |
| 4096 | 202 µs | 69.1 µs | 0.34 |
| 6144 | 448 µs | 252 µs | 0.56 |

Karatsuba wins clearly from 2048 on (one reproducible anomaly at 3072,
cache-aliasing suspect); schoolbook wins through 1024. Command:

```sh
cargo bench --bench poly -- multiply
```

The Karatsuba recursion bottoms out in the packed schoolbook convolution
below 48 coefficients (`KARATSUBA_BASE`, private).

## AFFT product crossovers

Inherited from the extraction source (`gs-engine`) unchanged:
`AFFT_{PRODUCT,BATCH4,BATCH8,BATCH16}_CROSSOVER` and their scalar twins. See
`gs-engine/BENCHMARKS.md` for the original measurement records; re-measure
on this crate's bench harness before retuning.

## Chien vs equal-degree (`chien_equal_degree_crossover`)

Analytic first cut (`|F| / log²|F|`, floor 8), pending a dedicated
measurement; the selector's structure (pure function of degree and field
order) is stable under retuning.

## Multipoint / interpolation crossovers

`MULTIPOINT_EVAL_CROSSOVER = 16` and `MODULE_INTERPOLATION_CROSSOVER = 8`
(the latter inherited from `gs-engine`). Both measured on this crate's
`poly` bench harness; retune with `cargo bench --bench poly`.

## Hasse derivative / jet panels (absorbed from `hasse` at the merger)

Absorbed from the `hasse` crate at the merger; re-measure on this crate's
harness before retuning. Every side runs the identical Goldilocks field
`p = 2^64 - 2^32 + 1`: coefficients and points generated once from the
fixed LCG constants, reduced, converted per library; outputs cross-checked
for equality before timing. Commands: `cargo bench --bench hasse` and
`cargo bench --bench competitors`.

Named competitors (dev-only): ark-poly 0.6 / ark-ff 0.6 (multiplicity one,
per-point public evaluator loop — no multipoint tree over arbitrary points),
lambdaworks-math 0.13 (same shape), winter-math 0.13.1 (`polynom::eval_many`).
**No direct competitor found** for multiplicity above one: the panel records
an explicitly labelled composed baseline (independent `u128` binomials, then
each library's per-order evaluation loop), asserted equal to this crate's
jets before timing.

Multiplicity one, steady-state evaluation (µs, median):

| Geometry | ours | ark-poly | lambdaworks | winter-math |
| --- | --- | --- | --- | --- |
| points=256, D=256 | **204** | 429 | 318 | 207 |
| points=1024, D=1024 | **2.62 ms** | 6.87 ms | 4.98 ms | 3.28 ms |

Plan construction is a one-time cost per request geometry: 453 µs at
256/256, 3.49 ms at 1024/1024.

Composed Hasse baseline, points=64, D=64, one-shot (µs, median; ours
includes plan, scratch, and output; libraries include derivative
preparation):

| Side | s=2 | s=4 | s=8 | s=16 |
| --- | --- | --- | --- | --- |
| ours `MultiplicityPlan` | 342 | 614 | 1.42 ms | 4.34 ms |
| ark-poly composed | 53 | 222 | 837 | 2.72 ms |
| lambdaworks composed | 65 | 225 | 792 | 2.66 ms |
| winter-math composed | 50 | 197 | 738 | 2.56 ms |

Prepared:

| Side | s=2 | s=4 | s=8 | s=16 |
| --- | --- | --- | --- | --- |
| ours `MultiplicityPlan` | 51 | 166 | 627 | 2.43 ms |
| ark-poly composed | 27 | 75 | 184 | 371 |
| lambdaworks composed | 40 | 77 | 151 | 283 |
| winter-math composed | 25 | 49 | 94 | 175 |

At this geometry the composed baselines win from s=4 up: `s` evaluations of
a degree-64 polynomial cost less than the remainder descent that produces
all `s·64` jet entries in one pass. The descent's structure — one
quasi-linear pass plus one prepared jet per point — is the asymptotic
argument; this small panel does not demonstrate it, and no larger panel has
been run.

`benches/hasse.rs` separates plan construction, scratch construction, and
steady-state execution over Gf8B, Gf16, Mersenne31, Goldilocks, and
QuadMersenne31: point counts 16/64/256/1024 (skipped, never shrunk, where
the count exceeds the field's element count), uniform multiplicities
1/2/4/8/16/64, degrees around W/2, W, and 2W, a `[1,2,4,8]`-repeat panel,
and a single heavy leaf.

## Named competitors (ground rule 7)

Non-copyleft libraries covering this crate's domain (polynomial rings over
binary and prime fields), ranked by coverage:

| Library | License | GF(2^m) coefficients | Ring ops | Transform domain | Measured? |
| --- | --- | --- | --- | --- | --- |
| [ark-poly](https://docs.rs/ark-poly) 0.6 / ark-ff 0.6 | Apache-2.0 OR MIT | no (prime base and prime towers only) | multiply (FFT and schoolbook), divide, evaluate; **no gcd / EEA** | multiplicative radix-2/mixed domains, `Evaluations` | yes |
| [lambdaworks-math](https://docs.rs/lambdaworks-math) 0.13 | Apache-2.0 OR MIT | no (same limitation) | multiply, divide, `xgcd`, evaluate, Lagrange interpolate | field NTT, not wired into `Polynomial` | yes |
| [libfqfft](https://github.com/scipr-lab/libfqfft) | MIT | **yes** (GF(2^m) via GMP/NTL backends), additive FFTs included | polynomial rings + FFTs | additive and multiplicative | no — C++, not linkable from cargo |

Context only, excluded from the build (copyleft): NTL (LGPL-2.1+), FLINT
(LGPL-3.0+), gf2x (LGPL-2.1+) — the classical GF(2)[x] / GF(2^m)[x]
references.

**No direct competitor found** among linkable non-copyleft libraries for:

- GF(2^m)-specific packed coefficient kernels (`fgf` SIMD/GFNI paths) —
  neither Rust competitor has a binary-field coefficient type at all;
- additive (subspace/coset) transform evaluation and interpolation — the
  only permissive char-2 implementation is libfqfft, which cannot be
  linked;
- root machinery: Chien search, equal-degree factoring, linearized/affine
  solving, Roth–Ruckenstein and Alekhnovich lifting;
- truncated power-series inversion and truncated EEA (the key-equation
  shape).

ark-poly and lambdaworks cover generic finite-field polynomial rings; the
rows below benchmark that overlap at matched 8-byte element width — this
crate over `fgf::Gf64`, the competitors over the Goldilocks prime
`2^64 − 2^32 + 1` (which ark needs anyway for power-of-two NTTs). Same
element width, different field: treat cross-library ratios as indicative of
algorithm cost, not of identical domains.

## Competitor measurements (2026-09-10)

```sh
cargo bench --bench competitors
```

Full product, equal operands of `n` coefficients:

| n | ours (dispatch) | ours (schoolbook) | ark (FFT default) | ark (schoolbook) | lambdaworks |
| --- | --- | --- | --- | --- | --- |
| 64 | 3.9 µs | 3.9 µs | 4.1 µs | 6.0 µs | 7.4 µs |
| 256 | 36.2 µs | 36.6 µs | 19.0 µs | 429 µs | 117 µs |
| 1024 | 741 µs (wide CI) | 419 µs | 228 µs | 6.78 ms | 1.74 ms |
| 4096 | 3.27 ms | 6.22 ms | 1.21 ms | 133 ms | 32.5 ms |

Reading: we win through ~64 coefficients; ark's multiplicative NTT wins the
transform region (1.21 ms vs 3.27 ms at 4096 — its one-limb Montgomery
multiplication is cheaper per butterfly than our GF(2^64) tower
multiplication). The dispatch at 1024 is a retune candidate: it enters the
AFFT region and loses to its own schoolbook twin (741 µs vs 419 µs).

Remaining overlap:

| Operation | ours | ark | lambdaworks |
| --- | --- | --- | --- |
| divide, 512×128 | **54.2 µs** | 137.8 µs (naive) | 866 µs |
| gcd/EEA, 400×300 (Bézout triple) | **525 µs** (`gcd_ext`) | not offered | 2.66 ms (`xgcd`) |
| evaluate, degree 4096 (Horner) | 180 µs | **13.8 µs** | **19.6 µs** |
| interpolate, 64 points | 80.7 µs (subproduct tree, arbitrary points) | **0.72 µs** (radix-2 IFFT over a domain) | 1.05 ms (Lagrange, arbitrary points) |

Reading: division and gcd/EEA are ours by 2.5–16× (ark ships no polynomial
gcd at all — a coverage gap, not a speed claim). Horner evaluation is
coefficient-multiplication-bound, and their Goldilocks Montgomery multiply
is ~10× cheaper per element than the GF(2^64) tower multiply — that is the
price of the binary-field domain, and structured-domain evaluation bypasses
it via `butterfly-fft`. Against the like-for-like arbitrary-point Lagrange
interpolation we are 13× faster than lambdaworks; ark's IFFT row runs over a
fixed radix-2 domain, a different contract (points not caller-chosen).
