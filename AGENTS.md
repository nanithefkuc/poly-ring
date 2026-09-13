# poly-ring

> Given polynomials over GF(2^m), compute in the ring — multiply, divide, gcd,
> evaluate, find roots, interpolate, and work modulo x^t, and never construct a
> code, a matrix, or a transform buffer. Field arithmetic comes from `fgf`;
> transform-domain evaluation/interpolation from `butterfly-fft`; matrices from
> `gfm`. This crate receives polynomials and returns polynomials.

## Non-negotiables

1. **Compose `fgf`, never re-host.** Every coefficient operation is
   `fgf::field::Elem` (scalar) or `fgf::ops::*` (packed), dispatched above the
   lane-bytes crossover. No hand-rolled field loop.
2. **Compose `butterfly-fft` for structured domains.** Subspace/coset
   multipoint eval and interpolation call `TransformPlan` + `monomial↔novel`;
   the crate reimplements only the arbitrary-point (Horner / subproduct-tree)
   path. No second additive-FFT.
3. **One ring, canonical representations.** `Polynomial<F>` stores `Vec<u8>`
   packed LE, always normalized; the zero polynomial is the empty buffer.
   `BivariatePolynomial<F>` keeps canonical rows with trailing zero `Y`
   rows dropped; `SparsePolynomial<F, N>` keeps terms sorted, merged, and
   zero-free. Conversions between them are explicit, never implicit.
4. **Characteristic-exact, not binary-first.** Hasse factors, `Y`
   substitutions, and root signs run through `binomial::<F>` and the
   field's negation. A parity mask is a characteristic-two optimization,
   never the general rule.
5. **No `unsafe`.** Forbidden at the crate root. The only SIMD is upstream.
6. **Steady-state zero allocation.** Every hot op has a `*_into` / scratch
   form; proven by `tests/zero_alloc.rs`.
7. **`inv(0) == 0` is inherited.** Division and root logic test pivots/roots
   with `is_zero()`; never infer "not a root" from a division result.
8. **Numbers live in `BENCHMARKS.md`.** Doc comments state the decision and
   the mechanism and point there.
9. **Do not land a performance change on reasoning alone.** A/B it, keep both
   twins compiled, record the ratio.
10. **Oracles stay independent.** An implementation is never its own test.

## Tooling

`just validate` is the pull-request gate; the shared recipe surface is
documented once in the umbrella's root `AGENTS.md`.

- **`TIERS = v3_gfni_crypto v3 v2 scalar`.** This crate owns no kernels — the
  tiers are `fgf`'s packed field ops and, through the `fft` feature,
  `butterfly-fft`'s butterflies. Above the lane-bytes crossover the polynomial
  paths route into them, so `just test-tiers` and `just cover` re-run pinned to
  each tier; the scalar run is the one that proves the portable twin.
- **`MIRI` is empty** — non-negotiable 4, no `unsafe` at the crate root. `just
  unsafe-check` reports that and skips. `COV_IGNORE` is empty too: every line
  counts toward the 95% gate.
- **Bench targets:** `poly`, `hasse`, `prepared`, `competitors` — `just bench-save poly`
  before the change, `just bench poly` after. `prepared` needs `internals`; the
  bench recipes pass `--all-features`, so it builds. `competitors` is the
  dev-only external panel (ring panel over 8-byte fields, Hasse panel over
  the shared Goldilocks prime including the labelled composed baseline for
  multiplicity above one).
- The cross-builds and `+1.89.0` check below are `just msrv` and manual `cargo
  build --target …`; `justfile` is a byte-identical vendored copy that must not
  be edited here (the umbrella's `just drift` check fails on it), and anything
  crate-specific goes in `crate.just`.

## Working here

- Edition 2024, MSRV 1.89. No toolchain pin; select `+1.89.0` for the MSRV
  job.
- Features: `default = ["std", "simd", "fft"]`; `simd` implies `std`; `fft`
  implies `std` and composes `butterfly-fft`; `parallel` is an off-by-default
  no-op placeholder; `internals` exposes this crate's unstable surface (never
  `fgf`'s — we do not enable it).
- `src/lib.rs` and every `mod.rs` hold declarations only — no function bodies,
  no `impl` blocks. Public items are re-exported at the crate root.
- Errors are hand-rolled in `src/error.rs`: small enums per failure domain,
  struct variants carrying the offending value and the limit, manual
  `Display`, `std::error::Error` under `std`. Every fallible public function
  documents `# Errors`.
- Test placement follows visibility: in-module `#[cfg(test)]` for private
  state, `tests/` for the public surface. Fixed-seed LCG only (`fgf`'s
  `noise(len, seed)` shape); no `rand`. Exact values, not predicates.
- The full check set:

  ```sh
  cargo fmt --all -- --check
  cargo clippy --all-targets --all-features -- -D warnings
  cargo clippy --all-targets --no-default-features -- -D warnings
  cargo test
  cargo test --all-features
  cargo test --no-default-features
  RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
  cargo build --target aarch64-unknown-linux-gnu --no-default-features
  cargo build --target wasm32-unknown-unknown --no-default-features
  cargo +1.89.0 build --all-features
  ```

- Benchmarks go through `criterion`; baselines are deliberately not committed.
  Measurement hygiene: interleave base/new, take the maximum of at least three
  runs, keep an unchanged 1.00x control, treat 16–128 byte buffers as noise.
- Commit subjects are at most ~10 words, shaped `poly-ring: short verb
  phrase`. What changed and why lives in the pull request and `CHANGELOG.md`.
