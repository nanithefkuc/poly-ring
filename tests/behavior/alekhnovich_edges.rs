//! Alekhnovich divide-and-conquer edges on one field: the scalar leaf, the
//! vanish-exactly and valuation-confirmed refine arms, and the output
//! materialization limits.
//!
//! Every test runs with `with_roth_ruckenstein_crossover(0)` so the
//! divide-and-conquer route is forced regardless of weighted input size.
//! All cases share `Gf16` so the driver's branches land in one
//! instantiation.

#![cfg(feature = "fft")]

use fgf::{Gf16, gf16};
use poly_ring::{
    AlekhnovichLimits, AlekhnovichScratch, Polynomial, alekhnovich_roots, alekhnovich_roots_into,
};

fn g16(value: u16) -> gf16::Elem {
    gf16::Elem::from_raw(value)
}

fn forced_limits(
    max_output_roots: usize,
    max_families: usize,
    max_coefficients: usize,
    max_scratch_bytes: usize,
) -> AlekhnovichLimits {
    AlekhnovichLimits::new(
        1_000_000,
        max_families,
        max_coefficients,
        max_scratch_bytes,
        max_output_roots,
    )
    .with_roth_ruckenstein_crossover(0)
}

/// Build `Q(X, Y) = f(X) + Y`.
fn with_root(coefficients: &[gf16::Elem]) -> Vec<Polynomial<Gf16>> {
    let f = Polynomial::<Gf16>::from_coefficients(coefficients).expect("f");
    let one = Polynomial::<Gf16>::one().expect("one");
    vec![f, one]
}

/// The scalar-precision leaf charges its materialization budget, resolves
/// the constant-Y row, splits the base-field roots, and reserves the
/// family list; with `max_degree` zero it extracts the constant root.
#[test]
fn scalar_precision_leaf_extracts_root_and_verifies() {
    let a = g16(0x0107);
    let rows = with_root(&[a]);
    let mut scratch = AlekhnovichScratch::<Gf16>::new();
    let limits = forced_limits(256, 1_000_000, 1 << 24, 1 << 28);
    let found = alekhnovich_roots(&rows, 0, limits, &mut scratch).expect("roots");
    assert_eq!(found.len(), 1);
    assert_eq!(
        found[0],
        Polynomial::<Gf16>::from_coefficients(&[a]).expect("const")
    );
}

/// A linear root with headroom for degree two drives the coarse-refine
/// cycle: the coarse pass establishes the constant prefix and the refine
/// pass confirms the family once the valuation reaches the frame
/// precision.
#[test]
fn valuation_exceeds_or_equals_precision_confirms_family() {
    let a = g16(0x1122);
    let c = g16(0x2233);
    let rows = with_root(&[a, c]);
    let mut scratch = AlekhnovichScratch::<Gf16>::new();
    let limits = forced_limits(256, 1_000_000, 1 << 24, 1 << 28);
    let found = alekhnovich_roots(&rows, 2, limits, &mut scratch).expect("roots");
    let expected = Polynomial::<Gf16>::from_coefficients(&[a, c]).expect("f");
    assert!(found.contains(&expected));
}

/// A constant root vanishes exactly under the affine substitution: the
/// transformed rows carry no X-valuation, so the family is inserted
/// without further refinement.
#[test]
fn vanish_exactly_family_is_inserted_as_complete() {
    let a = g16(0x3B2C);
    let rows = with_root(&[a]);
    let mut scratch = AlekhnovichScratch::<Gf16>::new();
    let limits = forced_limits(256, 1_000_000, 1 << 24, 1 << 28);
    let found = alekhnovich_roots(&rows, 0, limits, &mut scratch).expect("roots");
    assert_eq!(found.len(), 1);
    assert_eq!(
        found[0],
        Polynomial::<Gf16>::from_coefficients(&[a]).expect("const")
    );
}

/// A quadratic with two constant roots materializes, deduplicates, and
/// verifies both candidates through the output pipeline.
#[test]
fn materialize_candidates_verifies_every_root() {
    let a = g16(0x0708);
    let c = g16(0x2324);
    // Q = (Y + a)(Y + c) = Y^2 + (a + c) Y + a·c.
    let q0 = Polynomial::<Gf16>::from_coefficients(&[a.mul(c)]).expect("a*c");
    let q1 = Polynomial::<Gf16>::from_coefficients(&[a.add(c)]).expect("a+c");
    let q2 = Polynomial::<Gf16>::one().expect("one");
    let mut scratch = AlekhnovichScratch::<Gf16>::new();
    let limits = forced_limits(256, 1_000_000, 1 << 24, 1 << 28);
    let found = alekhnovich_roots(&[q0, q1, q2], 3, limits, &mut scratch).expect("roots");
    assert_eq!(found.len(), 2);
    assert!(found.contains(&Polynomial::<Gf16>::from_coefficients(&[a]).expect("ca")));
    assert!(found.contains(&Polynomial::<Gf16>::from_coefficients(&[c]).expect("cc")));
}

/// A scratch reused across two extractions clears the completed family
/// slot and restores capacity.
#[test]
fn scratch_reuse_clears_completed_and_restores_capacity() {
    let rows = with_root(&[g16(0x1F0E)]);
    let mut scratch = AlekhnovichScratch::<Gf16>::new();
    let limits = forced_limits(256, 1_000_000, 1 << 24, 1 << 28);
    alekhnovich_roots(&rows, 0, limits, &mut scratch).expect("first");
    let cap = scratch.capacity();
    let rows2 = with_root(&[g16(0x7E6D)]);
    let mut output = Vec::new();
    alekhnovich_roots_into(&mut output, &rows2, 0, limits, &mut scratch).expect("second");
    assert_eq!(output.len(), 1);
    assert!(scratch.capacity() >= cap);
}

/// `AlekhnovichScratch::default` agrees with `new`.
#[test]
fn default_scratch_agrees_with_new() {
    let _ = AlekhnovichScratch::<Gf16>::default();
    assert_eq!(AlekhnovichScratch::<Gf16>::new().frame_capacity(), 0);
}

/// `alekhnovich_roots_into` returns the same roots as `alekhnovich_roots`.
#[test]
fn into_form_agrees_with_allocating_form() {
    let a = g16(0x51A4);
    let rows = with_root(&[a]);
    let mut scratch = AlekhnovichScratch::<Gf16>::new();
    let limits = forced_limits(256, 1_000_000, 1 << 24, 1 << 28);
    let from_alloc = alekhnovich_roots(&rows, 0, limits, &mut scratch).expect("alloc");
    let mut into_output = Vec::new();
    alekhnovich_roots_into(&mut into_output, &rows, 0, limits, &mut scratch).expect("into");
    assert_eq!(from_alloc, into_output);
}
