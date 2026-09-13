> [!WARNING]
> This library was made with the help of AI. While the library has tests
> to check for regressions, things can break. Audit the code yourself, or with
> your own agent before using.

# univariate - Univariate Polynomial Arithmetic over Binary and Prime Fields

`univariate` is the univariate-polynomial-ring node of the FEC stack: the
dense monomial-basis [`Polynomial<F>`] over a binary field, every ring
operation on it, division and gcd with Bézout cofactors, the truncated /
partial extended Euclidean algorithm that solves the decoding key equation,
root-finding (Chien search, equal-degree factorization, linearized/affine
solving, Roth–Ruckenstein and Alekhnovich power-series lifting), evaluation
and interpolation over arbitrary and structured point sets, and truncated
power-series inversion.

> Given polynomials over GF(2^m), compute in the ring — multiply, divide,
> gcd, evaluate, find roots, interpolate, and work modulo x^t, and never
> construct a code, a matrix, or a transform buffer. Field arithmetic comes
> from `fgf`; transform-domain evaluation/interpolation from
> `butterfly-fft`; matrices from `gfm`. This crate receives polynomials and
> returns polynomials.

Field arithmetic is composed, never re-hosted: coefficient vectors run
through `fgf`'s packed kernels above the measured lane-bytes crossover and
scalar element arithmetic below it. Structured-domain evaluation and
interpolation compose `butterfly-fft` transforms under the default-on `fft`
feature; a transform-free consumer builds `--no-default-features` and drops
them.

## Usage

The MSRV is Rust 1.89.

`univariate` is distributed through git only; it is not published to
[crates.io](https://crates.io).

```toml
[dependencies]
univariate = { git = "https://github.com/nanithefkuc/univariate" }
```

Portable `no_std` builds (core ring, gcd/EEA, division, Chien, Horner,
power series; no `butterfly-fft`):

```toml
[dependencies]
univariate = { git = "https://github.com/nanithefkuc/univariate", default-features = false }
```

### Features

| Feature | Result |
| --- | --- |
| default (`std`, `simd`, `fft`) | full ring, structured-domain transforms, all root backends |
| `std` without `simd` | portable kernels with allocation-backed plans |
| `--no-default-features` | `no_std` core ring, no `butterfly-fft` |
| `fft` | subspace/coset evaluation and interpolation through `butterfly-fft` |
| `parallel` | off-by-default placeholder for batch-axis parallelism |
| `internals` | unstable benchmarking surface, no compatibility promise |

### A taste

```rust
use fgf::{Gf16, gf16};
use univariate::Polynomial;

let a = Polynomial::<Gf16>::from_coefficients(&[gf16::Elem(1), gf16::Elem(2), gf16::Elem(3)]).unwrap();
let b = Polynomial::<Gf16>::from_coefficients(&[gf16::Elem(5), gf16::Elem(7)]).unwrap();

// Ring arithmetic and division.
let product = a.multiply(&b).unwrap();
let (quotient, remainder) = product.div_rem(&b).unwrap();
assert_eq!(quotient, a);
assert!(remainder.is_zero());

// Bézout cofactors: s·a + t·b == g.
let relation = a.gcd_ext(&b).unwrap();
assert_eq!(
    relation.a_cofactor.multiply(&a).unwrap()
        .add(&relation.b_cofactor.multiply(&b).unwrap()).unwrap(),
    relation.gcd
);

// The key-equation primitive: run EEA on (x^{2t}, S), stop at deg < t.
// Equivalent to Berlekamp–Massey on the same syndrome sequence.
let step = univariate::truncated_eea(&a.shifted(8).unwrap(), &b, 4).unwrap();
assert!(step.remainder.degree().is_none_or(|d| d < 4));

// Prepared remainders modulo arbitrary moduli, batched by lane: one
// Newton short-division descent, allocation-free once prepared.
let moduli = [a.clone(), b.clone()];
let tree = univariate::RemainderTree::new(&moduli, 64).unwrap();
let mut scratch = tree.scratch(1).unwrap();
let mut output = vec![0u8; tree.leaf_offsets().last().unwrap() * fgf::Gf16::BYTES];
tree.remainders_into(product.as_packed(), product.coefficient_count(), 1,
    &mut scratch, &mut output).unwrap();
```

## Build

```sh
cargo build                       # default features
cargo build --no-default-features # no_std core ring, no butterfly-fft
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

Benchmark crossovers live in `BENCHMARKS.md`; the source carries only
one-line pointers to it.

## License

MIT. See [LICENSE](LICENSE).
