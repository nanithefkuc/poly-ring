> [!WARNING]
> This library was made with the help of AI. Audit the code yourself, or with
> your own agent before using.

> [!WARNING]
> `poly-ring` makes no constant-time guarantee. Division, gcd,
> factorization, and root search branch on intermediate values, and field
> arithmetic comes from variable-time `fgf`. Handling secret elements,
> coefficients, or buffers requires a separate audit.

# poly-ring — The Polynomial Ring over Finite Fields

`poly-ring` provides univariate and multivariate polynomial arithmetic,
evaluation, interpolation, and factorization over finite fields. The dense
monomial-basis `Polynomial<F>`, the row-canonical `BivariatePolynomial<F>`,
and the sparse `SparsePolynomial<F, N>` with explicit monomial orders share
one ring object: multiply, divide, differentiate, evaluate, interpolate,
factor, find roots, and work modulo an arbitrary polynomial or `x^t`.

| Main property | What it provides |
| --- | --- |
| Checked public API | Typed errors describe invalid polynomial, buffer, workspace, and domain geometry. |
| Composed field arithmetic | Coefficient operations use `fgf` scalar elements and packed kernels. |
| Reusable workspace | Prepared products and weighted evaluation have scratch-backed execution, covered by counting-allocator tests. |
| Structured domains | Subspace and coset evaluation and interpolation compose `butterfly-fft` under the `fft` feature; the Horner and subproduct-tree paths are this crate's own. |
| Characteristic-exact | Hasse derivatives, `Y` substitutions, and root signs take binomial factors in the field's characteristic, never as parity masks or factorials. |
| Runtime SIMD | `fgf` field kernels and `butterfly-fft` additive butterflies are selected upstream; the crate owns no kernels and no detector. |
| Portable builds | `no_std` plus `alloc` core ring; the `fft` feature composes transforms without `std`. |

`poly-ring` is a ring library, not a codec. It does not construct codes or
coding matrices, own shards, transform buffers, or matrix layouts, or run a
recovery protocol: field arithmetic lives in
[`fgf`](https://github.com/nanithefkuc/fgf), transforms in
[`butterfly-fft`](https://github.com/nanithefkuc/butterfly-fft), and linear
algebra in [`gfm`](https://github.com/nanithefkuc/gfm). This crate receives
polynomials and returns polynomials.

## Installation

The minimum supported Rust version is 1.93, edition 2024.

```toml
[dependencies]
poly-ring = "1.0"
fgf = "=1.2.1"
```

The exact `fgf` pin keeps one field-type copy across the graph; it tracks
the release in `Cargo.toml`.

For the portable `no_std` core ring without the transform feature:

```toml
[dependencies]
poly-ring = { version = "1.0", default-features = false }
fgf = { version = "=1.2.1", default-features = false }
```

## Quick start

Polynomials store `fgf` elements packed little-endian, low degree first,
always normalized; the zero polynomial is the empty buffer:

```rust
use fgf::{Gf16, gf16};
use poly_ring::Polynomial;

let a = Polynomial::<Gf16>::from_coefficients(&[
    gf16::Elem::from_raw(1),
    gf16::Elem::from_raw(2),
])
.unwrap();
let b = Polynomial::<Gf16>::from_coefficients(&[gf16::Elem::from_raw(5)]).unwrap();

let product = a.multiply(&b).unwrap();
let (quotient, remainder) = product.div_rem(&b).unwrap();
assert_eq!(quotient, a);
assert!(remainder.is_zero());

// Bézout cofactors: s·a + t·b == g.
let relation = a.extended_gcd(&b).unwrap();
assert_eq!(
    relation.a_cofactor.multiply(&a).unwrap()
        .add(&relation.b_cofactor.multiply(&b).unwrap()).unwrap(),
    relation.gcd
);

// The key-equation primitive: stop the Euclidean algorithm the moment
// the remainder degree drops below the bound.
let step = poly_ring::truncated_eea(&a.shifted(8).unwrap(), &b, 4).unwrap();
assert!(step.remainder.degree().is_none_or(|d| d < 4));
```

## Supported fields

Ring operations use `fgf` field types implementing `FieldKernels`. The sealed
prepared product engine supports the packed byte-lane fields below; bit-packed
GF(2) is outside that engine's scope.

| Field | Marker | Available prepared transform with `fft` |
| --- | --- | --- |
| GF(2^8), AES polynomial | `Gf8B` | additive transform |
| GF(2^16) | `Gf16` | additive transform |
| GF(2^32) | `Gf32` | additive transform |
| GF(2^64) | `Gf64` | additive transform |
| Fan–Paar GF(2^8)–GF(2^64) | `FanPaar8` … `FanPaar64` | additive transform |
| GF(2^64 − 2^32 + 1) | `Goldilocks` | number-theoretic transform |
| GF((2^31 − 1)²) | `QuadMersenne31` | number-theoretic transform |
| GF(2^31 − 1) | `Mersenne31` | embedded into `QuadMersenne31` |
| GF(2^8), polynomial `0x11D` | `Gf8D` | schoolbook/Karatsuba only |
Availability does not imply automatic selection. Goldilocks and
QuadMersenne31 automatic products use the NTT once the shorter operand
reaches the measured crossover, and Mersenne31 products use the embedded
route past its own crossover; the one-shot `Polynomial::multiply` takes
the same routes past its own per-field crossovers. Without `fft`,
prepared products use schoolbook and Karatsuba.

## The ring's surface

| Surface | Contract |
| --- | --- |
| Ring arithmetic | `Polynomial::multiply` selects an algorithm from input geometry; `multiply_rows_into` handles packed batched products with reusable `ConvolutionScratch`. |
| Division and gcd | `div_rem`, extended gcd with Bézout cofactors, and `truncated_eea`, the decoding key-equation primitive. |
| Factorization | Square-free separation, distinct-degree factors, and complete irreducible factorization. |
| Hasse derivatives and jets | `DerivativePlan` prepares one fixed derivative order; `JetPlan` computes the truncated Taylor translation `f(a + T) mod T^s`. |
| Evaluation | Horner, lane-parallel Horner over the packed field kernels, and subproduct-tree multipoint evaluation selected by request geometry; `RemainderTree` reduction modulo arbitrary polynomials, Newton and Lagrange interpolation, and `EvaluationDomain` selection. |
| Roots | Binary Chien search, factorization-backed base-field roots, linearized solving, and power-series lifting over binary extension fields; affine root families and Alekhnovich lifting require `fft`. |
| Prepared quotient rings | `ModulusPlan` prepares reduction, arithmetic, and composition modulo an arbitrary modulus; `ModulusScratch` holds execution workspace. |
| Power series | Truncated inversion and series division. |

## Features

| Feature | Effect |
| --- | --- |
| default (`std`, `simd`, `fft`) | full ring, transforms, Hasse plans, roots |
| `std` | `fgf` runtime support |
| `simd` | `fgf` and `butterfly-fft` architecture kernels; implies `std` |
| `fft` | structured evaluation, additive and multiplicative product transforms, and transform-backed lifting; works without `std` |
| `internals` | this crate's own unstable benchmarking surface, never `fgf`'s; no compatibility promise |

Without default features the crate builds `no_std` plus `alloc`: gcd/EEA,
division, Chien search, Horner evaluation, Hasse plans, sparse multivariate
arithmetic, and power series, with no `butterfly-fft`. There is no separate
`alloc` feature: polynomial storage always allocates.

## Platforms and backends

`poly-ring` owns no SIMD kernels and adds no CPU detector. Packed
coefficient work runs on `fgf`'s selected field kernels; subspace and coset
transforms run on `butterfly-fft`'s additive butterflies.
`fgf::backend_for::<F>()` reports the upstream field backend.

`cost::BackendClass::detect` classifies the cached upstream selection.
The cost selectors choose routes as pure functions of explicit capability
and problem geometry, without CPU detection of their own.

[`simdispatch`](https://github.com/nanithefkuc/simdispatch) owns detection
and the downgrade-only `SIMD_BACKEND` override. It may request a weaker
backend at process startup; unsupported upgrades are ignored.

## Layout and safety

`Polynomial<F>` stores `fgf`'s packed little-endian element encodings, low
degree first, always normalized, with the zero polynomial as the empty
buffer. `BivariatePolynomial<F>` keeps canonical rows and drops trailing
all-zero `Y` rows. `SparsePolynomial<F, N>` stores exactly its nonzero
terms, sorted and merged. Conversions between the dense and sparse forms
are explicit and lossless.

The crate forbids unsafe Rust; the architecture-specific kernels are
supplied by its dependencies. Invalid geometry, domains, and configurations
return typed errors from `error` (`PolynomialError`, `ProductError`,
`EvalError`, `FactorizationError`, `RootError`, and their peers). Safe
memory access does not imply constant-time execution.

## Performance

[BENCHMARKS.md](https://github.com/nanithefkuc/poly-ring/blob/main/BENCHMARKS.md)
records public API measurements and competitor comparisons. `poly` and
`prepared` mix public paths with internal variants; the public record excludes
the internal timings. `hasse` measures weighted evaluation and construction,
and `competitors` runs the dev-only comparison panel:

```sh
FEC_GOLDEN_CORE=3 just bench-save poly
FEC_GOLDEN_CORE=3 just bench poly
```

## Building

From the repository, `just` provides the build and verification commands:

```sh
just build       # release build with all features
just features    # no-default, default, and all-feature tests
just test
just test-tiers  # one run per backend tier; dispatch resolves one backend per process
just doc
just validate    # complete CI gate, including coverage
```

## License

MIT — see [LICENSE](https://github.com/nanithefkuc/poly-ring/blob/main/LICENSE).
