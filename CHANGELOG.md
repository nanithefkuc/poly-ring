# Changelog

All notable changes to this project are documented in this file.

## Unreleased

### Added

- Prime-field correctness across the ring: Karatsuba recombination,
  Euclidean division, series inversion (Newton and naive), subproduct-tree
  leaves, and Hasse derivatives now carry their signs and binomial factors
  through the field's characteristic instead of the characteristic-two
  shortcuts. Binary results are unchanged. Polynomial ingress
  (`from_coefficients`, `from_packed`, `assign_coefficients`,
  `assign_packed`, `set_coefficient`) canonicalizes prime-field lanes, so
  noncanonical raw inputs can no longer reach the packed kernels.
- `binomial<F>(upper, lower)`: the binomial coefficient as a field element,
  by Lucas' theorem over base-`F::CHARACTERISTIC` digits; parity rule in
  characteristic two.
- Prepared product engine (`poly::convolution`): `PolynomialField` (sealed,
  every packed field), `ConvolutionScratch`, and `multiply_rows_into` over
  coefficient-major batched lane rows. Routes: schoolbook/Karatsuba
  (always), the shared AFFT row pipeline (binary fields, `fft`), an NTT
  over Goldilocks and QuadMersenne31, and exact QuadMersenne31 embedding
  for Mersenne31. `Auto` keeps the measured binary crossovers; prime routes
  stay on Karatsuba until a benchmark panel justifies a flip (forced routes
  under `internals` for the comparison).
- Prepared remainder tree (`eval::remainder`): `RemainderTree` and
  `RemainderScratch`, fast remainders of batched lane rows modulo arbitrary
  monic-normalized moduli by one Newton short-division descent over the
  shared product-tree topology (`eval::tree`, which multipoint evaluation
  now also uses). Prepared execution allocates nothing.
- Competitor benchmark harness (`benches/competitors.rs`) measuring
  multiplication, division, gcd/EEA, evaluation, and interpolation against
  `ark-poly` and `lambdaworks-math` at matched 8-byte element width. The
  competitor record — top three non-copyleft libraries by coverage, the
  coverage gaps labeled "no direct competitor found", and the measured
  numbers — lives in `BENCHMARKS.md`.

## 0.0.0 (2026-08-17)

Initial implementation of the univariate polynomial ring over GF(2^m).

- `Polynomial<F>`: dense monomial-basis coefficients packed in `fgf`'s
  little-endian element representation, canonical form enforced everywhere,
  zero polynomial as the empty buffer.
- Ring operations: add / add-scaled / scale / shift, schoolbook product
  through `fgf`'s packed kernels, characteristic-two `O(deg)` squaring,
  Hasse and formal derivatives, affine composition.
- Product tiers: measured schoolbook↔Karatsuba dispatch and the
  `butterfly-fft`-composed AFFT batched product behind the `fft` feature.
- Division: `div_rem` / `exact_divide` / `remainder` / `monic`, `X^k`
  valuation and exact division, `multiply_mod` / `square_mod` / `pow_mod`,
  all with reusable-output forms.
- `gcd`, `gcd_ext` with Bézout cofactors, and `truncated_eea` — the
  key-equation / Padé primitive equivalent to Berlekamp–Massey.
- Truncated power series: Newton-doubling `inverse_mod_x_power`, series
  division, reversal.
- Root finding: classical Chien search, `gcd(p, X^|F|+X)` equal-degree
  extraction with deterministic trace splitting, a standalone
  linearized/affine solver, and Roth–Ruckenstein / Alekhnovich
  power-series lifting over bivariate Y-rows.
- Evaluation and interpolation: Horner, subproduct-tree multipoint over
  arbitrary points, Newton and Lagrange interpolation, `EvaluationDomain`
  over arbitrary / additive-subspace / affine-coset point sets, and
  `butterfly-fft` transform composition under the `fft` feature.
- Measured backend selectors in `cost`; crossovers recorded in
  `BENCHMARKS.md`.
