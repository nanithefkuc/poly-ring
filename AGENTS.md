# AGENTS.md

Working rules for `poly-ring`. Rustdoc owns API contracts, `README.md` owns
scope and usage, `BENCHMARKS.md` owns public measurements, and `CHANGELOG.md`
owns release changes and migration guidance. Public files never reference
planning material.

## Scope and ownership

`poly-ring` computes in the polynomial ring over `fgf` fields: univariate
ring arithmetic, division, gcd / extended gcd / truncated EEA, complete
factorization, prepared quotient-ring arithmetic, bivariate and sparse
multivariate forms, Hasse derivatives and jets, evaluation and interpolation
(Horner, subproduct tree, Newton, Lagrange, multiplicity-weighted multipoint,
Hermite), truncated power-series inversion, and root finding (Chien search,
base-field roots, linearized solving, Roth–Ruckenstein and Alekhnovich
lifting). The crate receives polynomials and returns polynomials.

It is not a codec, matrix library, or transform buffer. Field arithmetic and
packed-buffer kernels belong to `fgf`; subspace/coset transforms belong to
`butterfly-fft`; matrices belong to `gfm`; wire formats, codes, and decoders
belong to consumers. Never re-host a lower crate's solved problem.

`src/cost.rs` holds pure strategy selectors over explicit cost keys.
`BackendClass::detect` queries upstream field selection; selectors themselves
perform no CPU detection. Threshold changes require independent correctness
checks and a paired measurement campaign.

## Required workflow

Run routine commands from the crate root through `just`:

```sh
just test [ARGS]        # all-features run on the host's best backend
just features           # no-default, default, all-features
just test-tiers         # one run per declared tier
just lint               # fmt and clippy at both feature ends, -D warnings
just doc                # rustdoc with warnings denied
just msrv               # +1.93.0 check, all features and targets
just cover              # per-tier merged coverage, 95% gate
just validate           # the pull-request gate
```

`just validate` runs before opening a pull request; a focused regression runs
first while iterating. `just doctor` confirms tooling before assuming it.
Edition 2024, MSRV 1.93 (the `fgf` floor).

`justfile` is a byte-identical vendored copy; editing it fails the umbrella's
`just drift` check. Crate-specific values (`CRATE`, `MSRV`, `TIERS`, `MIRI`,
`COV_IGNORE`) and recipes live in `crate.just`.

## Algebra and storage invariants

- **Compose `fgf`, never re-host.** Coefficient work runs through scalar
  `fgf::field::Elem` below the measured lane-bytes crossover and through
  `fgf::ops` packed kernels above it. No hand-rolled field loop anywhere.
- **Compose `butterfly-fft` for structured domains.** Subspace and affine
  coset evaluation/interpolation call `TransformPlan` and the
  monomial↔novel conversion under the `fft` feature. The arbitrary-point
  Horner and subproduct-tree paths are this crate's own; there is no second
  additive FFT.
- **One ring, canonical representations.** `Polynomial<F>` stores `fgf`'s
  packed little-endian bytes, low degree first, with no trailing zero
  coefficient surviving any constructor or mutation; the zero polynomial is
  the empty buffer and its `degree()` is `None`. `BivariatePolynomial<F>`
  keeps normalized rows and drops trailing zero `Y` rows, including on
  partial failure — a representation stays canonical even when an operation
  errors. `SparsePolynomial<F, N>` stores exactly its nonzero terms, sorted
  ascending lexicographically on the exponent arrays (variable 0 first),
  duplicates merged, no zero coefficients. Dense↔sparse conversions are
  explicit and lossless.
- **Characteristic-exact, not binary-first.** Hasse factors, `Y`
  substitutions, and root signs run through `binomial::<F>` and the field's
  negation. A parity mask is a characteristic-two optimization, never the
  general rule; no factorials over finite fields.
- **`inv(0) == 0` is inherited from `fgf`.** Test pivots and roots with
  `is_zero()`; never infer singularity or "not a root" from a division
  result.
- **Checked geometry, fallible reservation.** Validate public buffer geometry
  before execution. Document each operation's errors and failure-state
  guarantees; preserve canonical representations when an operation fails.
  Do not infer transactionality from the presence of a typed error.
- **Ecosystem mutation vocabulary.** `_into` overwrites a write-only
  destination, `_assign` folds into its destination, `_with` consumes
  prepared state, and `_scratch` borrows a caller-owned workspace. A
  reusable preparation and a mutable execution scratch are separate types.
  Output-writing free functions take the destination first.

## Modules and features

Source subtrees use the module-named file style: `eval.rs` beside `eval/`,
`poly.rs` beside `poly/`. A module-named file holds the subtree's module doc,
submodule declarations, re-exports, and genuinely shared surface; concrete
implementations are submodules. Preserve that style. Public entry points
re-export at the crate root; `src/geometry.rs` stays private.

| Module | Role |
| --- | --- |
| `poly` | the ring: arithmetic, schoolbook/Karatsuba/AFFT/NTT products, bivariate and sparse forms, division, gcd/EEA, factorization, prepared quotient rings, series |
| `eval` | multipoint evaluation and interpolation, domains, Hermite reconstruction, `fft`-gated transform composition |
| `roots` | Chien search, base-field roots, linearized solving, Roth–Ruckenstein and Alekhnovich lifting |
| `derivative`, `jet` | Hasse derivative plans; truncated Taylor translation |
| `cost` | `BackendClass` and pure strategy selectors |
| `error` | hand-rolled error enums, one per failure domain |
| `internals` | re-export-only unstable facade |

Features:

- `default = ["std", "simd", "fft"]`.
- `simd` implies `std`: `fgf`'s backend cache is a `LazyLock`, and a
  simd-without-std build would compile and silently run scalar.
- `fft` activates the optional `butterfly-fft` dependency and works without
  `std` — the transform core is `no_std`-capable and everything it gates here
  is too. `std` and `simd` forward to it through the weak `butterfly-fft?`
  activation only when `fft` selected the dependency.
- `parallel` is an off-by-default placeholder: `rayon` is declared, no source
  consumes it, and it must not enter default or `no_std` builds.
- `internals = []` exposes this crate's own unstable surface
  (`karatsuba_multiply`, `ProductRoute`, `binomial_odd`, `element_key`, the
  affine `Y` substitution) as a re-export-only facade. It activates no
  dependencies and never enables `fgf`'s `internals`. Stable APIs do not
  depend on it. The self dev-dependency turns the feature on for tests and
  benches without leaking it into the published dependency graph.

Dev-dependencies: `ark-ff`/`ark-poly`, `lambdaworks-math`, and
`winter-math` are the competitor panel; `criterion` runs without its default
features; `proptest` covers property suites; fixed-seed LCG helpers follow
`fgf`'s `noise` shape. Never `rand`, never a runtime dependency outside the
allowlist.

## Backend selection

This crate owns no SIMD kernels and no detection. `simdispatch`, reached
through `fgf`, is the single source for detection, ordering, and the
process-startup downgrade-only `SIMD_BACKEND` request; per-field selection is
`backend_for::<F>()`. `BackendClass::detect` wraps the selected backend once
per stage and threads it into the pure selectors — never a second probe,
override, or cache.

`crate.just` declares `TIERS = v3_gfni_crypto v3 v2 scalar`: the tiers `fgf`'s
packed field ops reach and, through `fft`, `butterfly-fft`'s butterflies.
`just test-tiers` and `just cover` start one process per tier because
dispatch resolves one backend per process. An unsupported requested tier
falls back; a green forced-tier run is not execution evidence — inspect the
reported backend.

## Safety and tests

Library code is `#![forbid(unsafe_code)]` at the crate root; SIMD lives
upstream. The attribute covers `src/` and its in-module tests, not the
separate test crates. The repository's only unsafe is the trait-mandated
`GlobalAlloc` surface of the two test-only allocators below. Any new unsafe
needs a per-item allowance, a SINCE–THUS proof, and an entry here.

| Item | Residue | Proof |
| --- | --- | --- |
| `tests/zero_alloc.rs`: `CountingAllocator`'s `GlobalAlloc` impl | Test-only counting callbacks | Pointers and layouts forward unchanged to `System`; only the thread whose gate is set increments the counter, so parallel sibling tests are never charged; null results are not counted. |
| `tests/behavior/alloc_resistance.rs`: `FailingAllocator`'s `GlobalAlloc` impl | Test-only refusal gate | Refuses only above the thread-local threshold and never while a thread unwinds, so assertion messages still allocate and failures stay visible; a matched-size gate refuses an exact allocation count once, then disarms; failed thread-local access defaults to not refusing. |

Allocation contracts:

- `tests/zero_alloc.rs` proves steady-state zero allocation for the
  `*_into` and scratch-owning paths over binary and prime fields, including
  first calls from a fresh scratch and packed row-buffer reuse. This is a
  per-path contract, not a crate-wide property: operations without a scratch
  form may allocate by design.
- `tests/behavior/alloc_resistance.rs` exercises fallible reservation paths
  with an allocation-refusal gate. Preserve the operation-specific error and
  state guarantees it checks; do not generalize these to untested paths.

Test layout and discipline:

- `tests/oracles.rs` holds naive scalar references, structurally unrelated to
  every dispatched path and never optimized. An implementation is never its
  own oracle.
- `tests/behavior.rs` pulls the `tests/behavior/` files through explicit
  `#[path]` declarations; the `*_instantiations`, `*_coverage`, and
  `*_edges` files each defend one surface. Public behavior is tested from
  outside the crate; private state is exercised by in-module tests.
- Tests assert exact values, error variants with payloads, boundaries, and
  state preservation. Panic tests match only a stable operation-specific
  fragment. No source-text, field-copy, or forwarding assertions.
- Deterministic inputs only: fixed-seed LCG in `fgf`'s `noise` shape. No
  `rand`, no snapshot framework, no async test runtime.
- `MIRI` is empty, so `just unsafe-check` reports and skips. `COV_IGNORE` is
  empty: every library line counts toward the coverage gate.

## Benchmarks

Pin every run: `FEC_GOLDEN_CORE=3` on the local Intel Core Ultra 7 258V, core
8 on `suisei-cachy` (the Core i7-12700K over SSH). The bench recipes taskset
to `$FEC_GOLDEN_CORE` and warn when it is unset; verify actual affinity.

```sh
FEC_GOLDEN_CORE=3 just bench-save poly   # baseline on the comparison commit
FEC_GOLDEN_CORE=3 just bench poly        # candidate against the baseline
```

A performance comparison reruns its baseline in the same session, interleaves
variants, and retains an unchanged control. Public measurement refreshes must
state their sampling and aggregation without claiming an optimization result.
Separate construction from prepared execution; include allocation when it is
part of the operation being measured. Check correctness before timing.
Record CPU, OS, toolchain, resolved backends, geometry, aggregation, and warmup
in `BENCHMARKS.md`: units in headers, `-` for unavailable values, factual
caveats without result commentary. Internal timings belong in git-ignored
storage. Never change a crossover from reasoning or cross-session comparisons.

Target classification — what is public API versus internal tuning:

| Target | Classification |
| --- | --- |
| `poly` | Mixed. Public: the `schoolbook` multiply arms and the `divide`, `gcd`, `gcd/ext`, and `series` groups. The `karatsuba/*` arms call `internals::karatsuba_multiply` — internal tuning, excluded from the public record. Requires `internals` to build. |
| `prepared` | Mixed. Public: `construct-scratch` and `execute/auto` arms over `ConvolutionScratch` and `multiply_rows_into`. The `execute/karatsuba` and `execute/transform` arms and the `prepared_tuning/goldilocks` group use `internals` for crossover evidence and are excluded from the public record. Unsupported forced transforms are omitted rather than rerouted. Requires `internals` to build. |
| `hasse` | Public, all arms: `MultiplicityPlan` construction, scratch construction, and steady-state weighted evaluation across binary and prime fields, uniform and nonuniform multiplicities. |
| `competitors` | Public and dev-only comparison arms. The original ring operations compare `Gf64` against Goldilocks at matched element width, not identical field arithmetic. The matched Goldilocks product panel supplies identical coefficients and checks exact agreement across libraries before timing. The Goldilocks Hasse panel also checks outputs; multiplicities above one use explicitly composed baselines. Copyleft libraries stay outside the build. Confirm each comparator actually ran. |

The bench recipes pass `--all-features`, so targets requiring `internals`
build. `bench-save NAME` selects a target and saves Criterion's `before`
baseline; `bench NAME` compares against it. Public timing records do not
substitute for the internal A/B evidence needed to change a selector.

## Documentation and release

Rustdoc states contracts: what an operation computes, its invariants, its
panics, and who owns each buffer. Every public item and module carries a
one-line summary; `just doc` treats warnings as errors. Doc comments name the
layout when a packed buffer is read. Prose uses third person, present tense,
and plain words. Measured effects belong in `BENCHMARKS.md`, not guides or
comments. Public files never cite private planning material.

`README.md` covers scope, usage, features, build instructions, MSRV, and
license with the AI-authorship warning. `CHANGELOG.md` follows Keep a
Changelog 1.1.0 with a maintained topmost `Unreleased` section; breaking
entries carry migration guidance.

The manifest keeps `publish = false`. The `include` list follows the
ecosystem convention — `src/`, `tests/`, `benches/`, `examples/`, `README.md`,
`LICENSE`, `CHANGELOG.md` — so a packaged tree carries the library, its
tests and benches, and user-facing documents only; `AGENTS.md`,
`BENCHMARKS.md`, the command surface, and CI stay outside. Release
verification runs from a packaged copy against published dependencies; the
umbrella patch table is for local integration and is never edited to hide a
resolution failure.

Commit subjects stay near ten words, shaped `poly-ring: short verb phrase`.
What changed and why lives in the pull request and `CHANGELOG.md`. One crate
per pull request, validation green before it opens.
