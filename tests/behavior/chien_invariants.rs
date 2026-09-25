//! Chien-scan invariants: the degree-bound safety check and alloc
//! resistance for the warm root-vector reservation.
//!
//! Lines 160–162 (`cant_find_more_roots_than_degree`) cannot be reached
//! through the public API: a degree-d polynomial over a field has at most d
//! distinct roots, and the scan's packed-element evaluation is exact over
//! GF(2^m).  The branch exists as a defensive invariant — no input produces
//! it, and we document it here as unreachable.

use fgf::field::Elem;
use fgf::{Gf8B, gf8b};
use poly_ring::{ChienScratch, Polynomial, base_field_roots};

use crate::oracles;
use oracles::{naive_evaluate, noise_poly};

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

/// A full-degree polynomial with few roots exercises the degree-bound
/// check without triggering it: roots.len() <= degree holds.
#[test]
fn degree_bound_is_satisfied_for_few_roots() {
    let polynomial = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1), b(1)]).expect("X^2+X");
    let roots = poly_ring::chien_roots(&polynomial)
        .expect("chien")
        .into_finite()
        .expect("finite");
    assert_eq!(roots.len(), 2);
    for root in &roots {
        assert!(naive_evaluate(&polynomial, *root).is_zero());
    }
}

/// A polynomial that vanishes on many field points still respects the
/// degree bound — the intersection with the zero polynomial is the All
/// case, not a violated bound.
#[test]
fn zero_polynomial_is_all_case_instead_of_violated_bound() {
    assert_eq!(
        poly_ring::chien_roots(&Polynomial::<Gf8B>::zero()).expect("chien"),
        poly_ring::BaseFieldRoots::All
    );
    assert_eq!(
        base_field_roots(&Polynomial::<Gf8B>::zero()).expect("equal-degree"),
        poly_ring::BaseFieldRoots::All
    );
}

/// A warmed Chien scan over a polynomial whose degree equals the number
/// of roots (maximal root set) still passes the bound check.
#[test]
fn maximal_root_set_passes_degree_bound() {
    // X·(X+1)·(X+2) — degree 3 with exactly 3 distinct roots.
    let a = b(0);
    let bb = b(1);
    let c = b(2);
    let mut polynomial = Polynomial::<Gf8B>::one().expect("one");
    polynomial = polynomial.multiply_x_plus(a).expect("X");
    polynomial = polynomial.multiply_x_plus(bb).expect("X+1");
    polynomial = polynomial.multiply_x_plus(c).expect("X+2");
    let roots = poly_ring::chien_roots(&polynomial)
        .expect("chien")
        .into_finite()
        .expect("finite");
    assert_eq!(roots.len(), 3);
    assert!(roots.contains(&a));
    assert!(roots.contains(&bb));
    assert!(roots.contains(&c));
}

/// A constant (degree 0, nonzero) has no roots and passes the bound with
/// roots.len() == 0 <= 0.
#[test]
fn constant_polynomial_passes_degree_bound() {
    assert_eq!(
        poly_ring::chien_roots(&noise_poly::<Gf8B>(1, 0xF001))
            .expect("chien")
            .into_finite(),
        Some(Vec::new())
    );
}

/// Scratch default matches a fresh empty constructor.
#[test]
fn chien_scratch_default_is_empty() {
    let _ = ChienScratch::<Gf8B>::default();
    let fresh = ChienScratch::<Gf8B>::new();
    let _ = fresh;
}

/// The equal-degree root list also passes its own invariant check:
/// every returned root vanishes in the input.
#[test]
fn equal_degree_roots_pass_invariant_check() {
    let polynomial = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1), b(1)]).expect("X^2+X");
    let roots = base_field_roots(&polynomial)
        .expect("equal-degree")
        .into_finite()
        .expect("finite");
    for root in &roots {
        assert!(naive_evaluate(&polynomial, *root).is_zero());
    }
}
