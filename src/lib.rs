//! The polynomial ring over binary and prime fields.
//!
//! > Given field elements and polynomials, compute in the ring — multiply,
//! > divide, differentiate, evaluate, interpolate, find roots, and work
//! > modulo x^t — and never construct a code, a matrix, a transform buffer,
//! > or a decoder. Field arithmetic comes from `fgf`; structured-domain
//! > evaluation/interpolation composes `butterfly-fft`. This crate receives
//! > polynomials and returns polynomials.
//!
//! # The object
//!
//! [`Polynomial<F>`] is the dense monomial-basis polynomial over a
//! `fgf` field: coefficients stored as `fgf`'s packed little-endian bytes, low
//! degree first, always normalized (no trailing zero coefficients), with the
//! zero polynomial represented by the empty buffer. [`BivariatePolynomial<F>`]
//! is the row-oriented dense bivariate form the interpolation engines
//! consume; [`SparsePolynomial<F, N>`] is the sparse multivariate form with
//! explicit [`MonomialOrder`] ranking. Conversions between the dense and
//! sparse forms are explicit and lossless.
//!
//! Field arithmetic is composed, never re-hosted: coefficient vectors run
//! through [`fgf::ops`] packed kernels above the measured lane-bytes
//! crossover and through scalar [`fgf::field::Elem`] arithmetic below it.
//! Structured-domain (additive subspace / affine coset) evaluation and
//! interpolation compose `butterfly-fft` transforms under the default-on
//! `fft` feature; the arbitrary-point Horner and subproduct-tree paths are
//! this crate's own. Hasse derivatives, jets, and the sparse multivariate
//! surface are exact over every characteristic — binomial factors are taken
//! in the field's characteristic, never as parity masks or factorials.
//!
//! # Layout
//!
//! - [`poly`] — the ring: construction, add/scale/shift/multiply
//!   (schoolbook, Karatsuba, AFFT), the dense bivariate object, sparse
//!   multivariate elements with monomial orders, division, gcd / extended
//!   gcd / truncated EEA (the key-equation primitive), and truncated
//!   power-series inversion.
//! - [`derivative`] and [`jet`] — prepared Hasse derivatives of one fixed
//!   order, and the truncated Taylor translation `f(a + T) mod T^s`.
//! - [`eval`] — Horner and subproduct-tree evaluation, Newton and Lagrange
//!   interpolation, multiplicity-weighted multipoint evaluation, Hermite
//!   reconstruction, [`eval::EvaluationDomain`] backend selection, and the
//!   `fft`-gated transform composition.
//! - [`roots`] — Chien search, equal-degree (Cantor–Zassenhaus) base-field
//!   roots, linearized/affine solving, and Roth–Ruckenstein / Alekhnovich
//!   power-series root lifting.
//! - [`cost`] — measured crossover constants and pure backend selectors.
//!
//! # Features
//!
//! | Feature | Effect |
//! | --- | --- |
//! | default (`std`, `simd`, `fft`) | full ring, transforms, Hasse plans, roots |
//! | `--no-default-features` | `no_std` core ring: gcd/EEA, division, Chien, Horner, Hasse plans, sparse multivariate arithmetic, power series; no `butterfly-fft` |
//! | `fft` without `std` | transform composition available in `no_std` builds |
//! | `parallel` | off-by-default placeholder for batch-axis parallelism |
//! | `internals` | unstable benchmarking surface, no compatibility promise |

//! ```
//! use fgf::{Gf16, gf16};
//! use poly_ring::Polynomial;
//!
//! let a = Polynomial::<Gf16>::from_coefficients(&[
//!     gf16::Elem::from_raw(1),
//!     gf16::Elem::from_raw(2),
//! ])
//! .unwrap();
//! let b = Polynomial::<Gf16>::from_coefficients(&[gf16::Elem::from_raw(5)]).unwrap();
//!
//! let product = a.multiply(&b).unwrap();
//! let (quotient, remainder) = product.div_rem(&b).unwrap();
//! assert_eq!(quotient, a);
//! assert!(remainder.is_zero());
//!
//! // Bézout cofactors: s·a + t·b == g.
//! let relation = a.gcd_ext(&b).unwrap();
//! assert_eq!(
//!     relation.a_cofactor.multiply(&a).unwrap()
//!         .add(&relation.b_cofactor.multiply(&b).unwrap()).unwrap(),
//!     relation.gcd
//! );
//!
//! // The key-equation primitive: stop the Euclidean algorithm the moment
//! // the remainder degree drops below the bound.
//! let step = poly_ring::truncated_eea(&a.shifted(8).unwrap(), &b, 4).unwrap();
//! assert!(step.remainder.degree().is_none_or(|d| d < 4));
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
#![warn(clippy::pedantic)]
#![allow(
    // Degree/length arithmetic moves through checked products; the
    // truncating casts that remain (field order u128 -> usize on capacities
    // already bounded by F::ORDER) are provably in range.
    clippy::cast_possible_truncation,
    // The crate is named after its object; Polynomial/EvaluationDomain-style
    // names read better than prefixed aliases.
    clippy::module_name_repetitions,
    // Ring identities read as equations; the trait method names that trip
    // this lint are the mathematical ones.
    clippy::similar_names,
    // `as_chunks::<F::BYTES>()` needs a const generic depending on a
    // generic parameter, which stable Rust rejects; the iterator form is
    // the only spelling that compiles for every field width.
    clippy::chunks_exact_to_as_chunks
)]

extern crate alloc;

pub mod cost;
pub mod derivative;
pub mod error;
pub mod eval;
pub mod jet;
pub mod poly;
pub mod roots;

mod geometry;

#[cfg(feature = "fft")]
pub use cost::product_crossover;
pub use cost::{
    BackendClass, BaseRootBackend, BaseRootCostKey, ProductBackend, ProductCostKey, RootBackend,
    RootCostKey, chien_equal_degree_crossover, select_base_roots, select_product, select_root,
};
pub use derivative::DerivativePlan;
pub use error::{
    ConfigError, DomainError, EvalError, HasseError, HermiteError, PolynomialError, ProductError,
    RootError,
};
pub use eval::{
    DomainScratch, EvaluationBackend, EvaluationDomain, HermitePlan,
    MODULE_INTERPOLATION_CROSSOVER, MULTIPOINT_EVAL_CROSSOVER, MultiplicityPlan,
    MultiplicityScratch, MultipointScratch, NewtonBasis, RemainderScratch, RemainderTree,
    evaluate_multipoint, evaluate_multipoint_into, interpolate_lagrange, interpolate_newton,
    interpolate_newton_into,
};
#[cfg(feature = "fft")]
pub use eval::{
    TransformScratch, evaluate_coset_into, evaluate_subspace, evaluate_subspace_into,
    interpolate_subspace, interpolate_subspace_into,
};
pub use jet::{JetPlan, JetScratch};
#[cfg(feature = "fft")]
pub use poly::{
    AFFT_BATCH4_CROSSOVER, AFFT_BATCH8_CROSSOVER, AFFT_BATCH16_CROSSOVER, AFFT_PRODUCT_CROSSOVER,
    SCALAR_AFFT_BATCH4_CROSSOVER, SCALAR_AFFT_BATCH8_CROSSOVER, SCALAR_AFFT_BATCH16_CROSSOVER,
    SCALAR_AFFT_PRODUCT_CROSSOVER, multiply_batch_truncated_with,
    substitute_y_affine_rows_truncated_into,
};
pub use poly::{
    BezoutRelation, BivariatePolynomial, ConvolutionScratch, KARATSUBA_CROSSOVER, MonomialOrder,
    MultiIndex, Polynomial, PolynomialField, SparsePolynomial, Term, TruncatedEea, WeightedTerm,
    binomial, binomial_odd, karatsuba_multiply, multiply_rows_into, series_divide, truncated_eea,
};
#[cfg(feature = "fft")]
pub use poly::{PolynomialProductScratch, ProductStrategy, multiply_batch_truncated};
#[cfg(feature = "internals")]
pub use poly::{ProductRoute, multiply_rows_route_into};
#[cfg(feature = "fft")]
pub use roots::{
    AffineRootFamily, AlekhnovichLimits, AlekhnovichScratch, DEFAULT_ROTH_RUCKENSTEIN_CROSSOVER,
    alekhnovich_roots, alekhnovich_roots_into,
};
pub use roots::{
    BaseFieldRoots, ChienScratch, FieldRootScratch, RothRuckensteinLimits, RothRuckensteinScratch,
    base_field_roots, base_field_roots_into, chien_roots, chien_roots_into, element_key,
    linearized_roots, roth_ruckenstein_roots, roth_ruckenstein_roots_into,
};
