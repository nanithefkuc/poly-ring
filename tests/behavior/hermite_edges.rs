//! Hermite interpolation: accessors and zero-weight requests.
//!
//! Reconstruction fixtures live in `hermite.rs`; what needs covering here
//! is the prepared-layout surface: the caller-order accessors and the
//! entries a zero multiplicity contributes (nothing).

use fgf::field::Elem;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::HermitePlan;

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

/// Accessors report the caller-order layout the plan was built from.
#[test]
fn accessors_report_the_caller_order_layout() {
    let points = [m31(4), m31(9)];
    let plan = HermitePlan::<Mersenne31>::new(&points, &[2, 3]).expect("plan");
    assert_eq!(plan.points(), &points);
    assert_eq!(plan.multiplicities(), &[2, 3]);
    assert_eq!(plan.offsets(), &[0, 2, 5]);
    assert_eq!(plan.total_weight(), 5);
}

/// A zero-multiplicity entry contributes no jet and no modulus: the
/// request reconstructs through the positive entries alone.
#[test]
fn zero_weight_entries_contribute_nothing() {
    // Jets [1, 0] at 0 alone constrain value 1 and vanishing first
    // Hasse derivative: the minimal interpolant is the constant 1, with
    // the zero-weight point contributing no constraint at all.
    let plan = HermitePlan::<Gf8B>::new(&[b(0), b(1)], &[2, 0]).expect("plan");
    assert_eq!(plan.total_weight(), 2);
    assert_eq!(plan.multiplicities(), &[2, 0]);
    let reconstructed = plan.interpolate(&[b(1), b(0)]).expect("interpolate");
    let expected = poly_ring::Polynomial::<Gf8B>::from_coefficients(&[b(1)]).expect("one");
    assert_eq!(reconstructed, expected);
    assert_eq!(reconstructed.evaluate_hasse(b(0), 0), b(1));
    assert!(reconstructed.evaluate_hasse(b(0), 1).is_zero());

    // All-zero weights reconstruct the zero polynomial.
    let empty = HermitePlan::<Gf8B>::new(&[b(0), b(1)], &[0, 0]).expect("plan");
    assert_eq!(empty.total_weight(), 0);
    assert!(empty.interpolate(&[]).expect("interpolate").is_zero());
}
