# Changelog

All notable changes to this project are documented in this file.

## Unreleased

### Added

- `poly-ring` opens at 0.1.0 as the ecosystem's single polynomial-ring
  crate, carrying the univariate ring in full plus the surfaces this
  version adds.
- `BivariatePolynomial<F>` / `WeightedTerm` (`poly::bivariate`): the dense
  `Q(X,Y) = Σ Q_j(X)·Y^j` object, general-field by construction. The
  Hasse discrepancy, the `Y` substitutions, and the root predicate
  (`Y − f(X)` divides, not `Y + f(X)`) run through `binomial::<F>` instead
  of binary parity rules; ordinary ring operations (`add`, `sub`,
  `scaled`, `multiply`, `hasse_derivative`) and the warmed bulk ingress
  `assign_y_coefficients_packed` join the surface. The batched
  `substitute_y_affine_truncated_fast` runs on the ring's own product
  engine under `fft`.
- Sparse multivariate elements (`poly::monomial`, `poly::multivariate`):
  `MultiIndex<N>`, `Term<F, N>`, `MonomialOrder<N>` (lex, graded lex,
  weighted), and `SparsePolynomial<F, N>` — canonical sparse storage,
  scalar term-pair arithmetic, ordered multivariate Hasse evaluation
  (`evaluate_hasse`, `evaluate_jet_into`), leading terms under any order,
  and lossless conversions to and from the dense univariate and bivariate
  forms.
- `HermitePlan<F>` (`eval::hermite`): the inverse of multiplicity-weighted
  evaluation. Reconstructs the unique polynomial of degree below the total
  weight from every point's local jet through a prepared scalar CRT —
  ring multiplication, exact division, extended gcd, and jet translation;
  no matrices, no Gaussian elimination. Conflicting duplicate points are
  rejected at construction; zero weights stay legal.
- `HermiteError` joins the error surface; `HasseError` now carries this
  crate's own polynomial and product errors.

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

### Fixed

- `gcd_ext` and `truncated_eea` updated their Bézout cofactors with a
  field addition where the Euclidean recurrence requires a subtraction.
  Correct in characteristic two (where subtraction is addition); over
  prime fields the cofactor identity `s·a + t·b = g` held only up to
  sign. Binary results are unchanged.
