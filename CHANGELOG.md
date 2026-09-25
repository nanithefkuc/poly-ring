# Changelog

All notable changes to this project are documented in this file.

## Unreleased

## [1.0.0] - 2026-09-25

### Added

- `poly-ring` opens at 1.0.0 as the ecosystem's single polynomial-ring
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
- Dense `Polynomial<F>` subtraction, additive negation, and the `is_one`
  identity query complete the named ring vocabulary. Square-free
  decomposition handles inseparable inputs through coefficient-field
  characteristic roots; distinct-degree and equal-degree factorization cover
  binary extension and odd-characteristic fields with deterministic factor
  ordering.
- `Polynomial::factor` and `Polynomial::is_irreducible` complete the
  factorization surface. Allocating base-field root extraction now retains
  linear factors over odd-characteristic fields as well as binary extensions;
  the scratch-backed root path remains binary to preserve its reuse contract.
- `ModulusPlan<F>` and `ModulusScratch<F>` provide reusable reduction,
  multiplication, squaring, exponentiation, inversion, and composition in
  `F[X]/(m)`. `Polynomial::compose` and `Polynomial::compose_mod` expose the
  corresponding one-shot compositions.

- Prime-field correctness across the ring: Karatsuba recombination,
  Euclidean division, Newton series inversion, subproduct-tree
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
  for Mersenne31. `Auto` keeps the measured binary crossovers and selects the
  Goldilocks NTT by shorter operand and batch width; the other prime routes
  remain explicit under `internals`.
- Prepared remainder tree (`eval::remainder`): `RemainderTree` and
  `RemainderScratch`, fast remainders of batched lane rows modulo arbitrary
  monic-normalized moduli by one Newton short-division descent over the
  shared product-tree topology (`eval::tree`, which multipoint evaluation
  now also uses). Prepared execution allocates nothing.
- Competitor benchmark harness (`benches/competitors.rs`) measuring
  multiplication, division, gcd/EEA, evaluation, and interpolation against
  `ark-poly` and `lambdaworks-math`. Its ring panel retains the matched-width
  comparison, while a separate Goldilocks product panel supplies identical
  coefficients to all libraries and checks exact agreement before timing.
  The measured numbers live in `BENCHMARKS.md`.

### Changed

- Goldilocks `Polynomial::multiply` and prepared `multiply_rows_into` select
  the existing number-theoretic transform at their measured crossovers.
  Smaller products retain the schoolbook and Karatsuba routes.

- QuadMersenne31 prepared products select the number-theoretic transform at
  the shared shorter-operand crossovers, and Mersenne31 products select the
  embedded QuadMersenne31 route past its own measured crossovers; smaller
  products retain the Karatsuba route. The one-shot `Polynomial::multiply`
  still selects the NTT for Goldilocks only.
- The manifest declares its package contents with an `include` list —
  sources, tests, benches, examples, and release documents — instead of an
  exclude list, and carries `readme`, `homepage`, `categories`, and
  `keywords` metadata.
- Coset evaluation takes `&TransformPlan<F>`. The transform crate merged its
  shifted plan into `TransformPlan::with_shift`, so the separate shifted-plan
  parameter no longer exists, and `ProductError::Transform` carries the
  transform crate's non-exhaustive `TransformError`.
- **Breaking:** Output-writing free functions now take the destination first;
  receiver methods keep the receiver and other inputs first and place output
  last. The public names now use `extended_gcd`, `divide_exact`,
  `BinaryRootScratch`, `binary_field_roots_into`, `RootLiftingBackend`,
  `RootLiftingCostKey`, `select_root_lifting`,
  `NEWTON_INTERPOLATION_CROSSOVER`, and
  `multiply_batch_truncated_into`. Implementation-only route controls and
  helpers moved under the unstable `internals` feature.
- The prepared remainder descent (`RemainderTree`, and with it
  `MultiplicityPlan` weighted evaluation) sizes each node's Newton short
  division by the live dividend length rather than by the node's prepared
  bound, so a dividend shorter than the tree's moduli no longer divides
  padded rows at every node. The quotient-times-modulus product is
  truncated to the rows the remainder reads, a single-lane descent
  multiplies the prepared reciprocal and modulus rows in place instead of
  broadcasting them per node, leaf output indices come from a prepared map
  rather than a rescan, and node depths come from the construction walk.
  Results, scratch sizes, and the allocation-free steady state are
  unchanged.
- Multipoint evaluation gains a lane-parallel Horner route that applies one
  Horner step across every point at once through the packed field kernels,
  and `MULTIPOINT_LANE_STEP_CROSSOVER` selects it against the
  subproduct-tree descent by multiply-add step count.
  `MultiplicityPlan::evaluate_batch_into` takes the same route for
  single-lane requests whose every multiplicity is one. Results and the
  allocation-free steady state are unchanged.
- The lane-parallel Horner route now multiplies its single accumulator in
  place with `fgf::ops::mul_elementwise_assign` and folds in each
  coefficient with `fgf::ops::add_assign_scalar`, and the Chien scan
  updates its running terms in place; both drop the ping-pong or
  successor buffer and the per-step broadcast fill or copy. The crate
  pins `fgf` 1.2.1 and `butterfly-fft` 1.0.2. Results and the
  allocation-free steady state are unchanged.

### Fixed

- `extended_gcd` and `truncated_eea` updated their Bézout cofactors with a
  field addition where the Euclidean recurrence requires a subtraction.
  Correct in characteristic two (where subtraction is addition); over
  prime fields the cofactor identity `s·a + t·b = g` held only up to
  sign. Binary results are unchanged.

### Removed

- The `parallel` placeholder feature and its optional `rayon` dependency.
  No source consumed it; batch-axis parallelism is not in the 1.0 surface.

- `Polynomial::inverse_mod_x_power_naive` is removed. The naive series
  solver survives only as the `naive_series_inverse` test oracle in
  `tests/oracles.rs`; no shipped path called it.
