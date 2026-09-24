//! Affine roots behavior and error boundaries.

use fgf::field::{Elem, Field};
use fgf::{Gf8B, Gf16, gf8b};
use poly_ring::{Polynomial, linearized_roots};

use crate::oracles;
use oracles::noise;

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

/// The zero map has the whole field as its kernel: 256 roots over Gf8B in
/// canonical key order, matching the zero-polynomial Chien scan.
#[test]
fn zero_map_kernel_is_the_whole_field() {
    // L(X) = X^2 + X^2 = 0 would normalize away; build the zero polynomial
    // directly with affine zero.
    let roots = linearized_roots(&Polynomial::<Gf8B>::zero(), b(0)).expect("all");
    assert_eq!(roots.len(), 256);
    let mut sorted = roots.clone();
    sorted.sort_by_key(|e| poly_ring::internals::element_key::<Gf8B>(*e));
    sorted.dedup();
    assert_eq!(roots, sorted);
    // Nonzero affine shift of the zero map is inconsistent: empty set.
    let empty = linearized_roots(&Polynomial::<Gf8B>::zero(), b(1)).expect("empty");
    assert!(empty.is_empty());
}

/// A rank-deficient map has a multi-dimensional kernel: L(X) = X^4 + X over
/// Gf16 has kernel dimension 2 (4 roots), verified against Chien.
#[test]
fn rank_deficient_map_enumerates_its_kernel() {
    // L(X) = X^4 + X: coefficients at degrees 1 and 4.
    let linearized = Polynomial::<Gf16>::from_coefficients(&[
        <Gf16 as Field>::Elem::ZERO,
        <Gf16 as Field>::Elem::ONE,
        <Gf16 as Field>::Elem::ZERO,
        <Gf16 as Field>::Elem::ZERO,
        <Gf16 as Field>::Elem::ONE,
    ])
    .expect("linearized");
    let roots = linearized_roots(&linearized, <Gf16 as Field>::Elem::ZERO).expect("roots");
    assert_eq!(roots.len(), 4);
    let ordinary = linearized.clone();
    let chien = poly_ring::chien_roots(&ordinary)
        .expect("chien")
        .into_finite()
        .expect("finite");
    assert_eq!(roots, chien);
    for root in &roots {
        assert!(ordinary.evaluate(*root).is_zero());
    }
}

/// An inconsistent affine shift lies outside the image: the solver returns
/// the empty set, matching the Chien scan of the shifted polynomial.
#[test]
fn inconsistent_shift_returns_empty_matching_chien() {
    // L(X) = X^2 + X over Gf8B has image of size 128; most affine shifts
    // are inconsistent. Find one by scanning.
    let linearized = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1), b(1)]).expect("L");
    let mut found_empty = false;
    for seed in 0..32 {
        let affine = noise::<Gf8B>(1, 0xF301 + seed)[0];
        let roots = linearized_roots(&linearized, affine).expect("roots");
        let mut shifted = linearized.clone();
        shifted.set_coefficient(0, affine).expect("shift");
        let chien = poly_ring::chien_roots(&shifted)
            .expect("chien")
            .into_finite()
            .expect("finite");
        assert_eq!(roots, chien);
        if roots.is_empty() {
            found_empty = true;
        }
    }
    assert!(found_empty, "expected an inconsistent shift in 32 draws");
}

/// A full-rank map is a bijection: exactly one root per affine shift.
#[test]
fn full_rank_map_is_bijective() {
    // L(X) = X over Gf8B: the identity map.
    let linearized = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1)]).expect("L");
    for seed in 0..8 {
        let affine = noise::<Gf8B>(1, 0xF302 + seed)[0];
        let roots = linearized_roots(&linearized, affine).expect("roots");
        assert_eq!(roots.len(), 1);
        // L(root) + affine = root + affine == 0 means root == affine.
        assert_eq!(roots[0], affine);
    }
}

/// Higher-degree linearized monomials solve exactly: X^8 + X^4 + X over
/// Gf16 matches Chien at several shifts.
#[test]
fn higher_degree_monomials_match_chien() {
    let one = <Gf16 as Field>::Elem::ONE;
    let zero = <Gf16 as Field>::Elem::ZERO;
    let linearized =
        Polynomial::<Gf16>::from_coefficients(&[zero, one, zero, zero, one, zero, zero, zero, one])
            .expect("L");
    for seed in 0..6 {
        let affine = noise::<Gf16>(1, 0xF303 + seed)[0];
        let roots = linearized_roots(&linearized, affine).expect("roots");
        let mut shifted = linearized.clone();
        shifted.set_coefficient(0, affine).expect("shift");
        let chien = poly_ring::chien_roots(&shifted)
            .expect("chien")
            .into_finite()
            .expect("finite");
        assert_eq!(roots, chien);
        for root in &roots {
            assert!(shifted.evaluate(*root).is_zero());
        }
    }
}
