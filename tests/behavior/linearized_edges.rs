//! Linearized-solver edges: the overflow guard in `enumerate_field`,
//! and verification against the Chien oracle over wider linearized
//! degrees.
//!
//! Lines 119–124 (error arm of `try_reserve_exact` in the main solver
//! path) cannot be reached with a simple allocator gate: every ordinary
//! Vec allocation (`basis`, `reduction`, `rows`) above the call is
//! non-fallible and aborts on null, so the gate that would refuse the
//! smaller roots allocation kills the binary first. The success arm is
//! exercised by the existing agreement suites; the error arm is
//! documented as a gated invariant.
//!
//! Lines 145–148 (`enumerate_field` overflow) cannot be reached for
//! the supported fields: every validated binary field has `F::ORDER <=
//! 2^16` which fits in `usize` on every tier target. The branch exists
//! as a defensive check.

use fgf::Field;
use fgf::field::Elem;
use fgf::{Gf8B, Gf16, gf8b};
use poly_ring::{Polynomial, RootError, linearized_roots};

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

/// The zero linearized polynomial with a nonzero affine constant has no
/// roots (the inconsistent-system path returns empty without allocating
/// the root vector).
#[test]
fn zero_linearized_nonzero_affine_returns_empty() {
    assert!(
        linearized_roots(&Polynomial::<Gf8B>::zero(), b(3))
            .expect("empty")
            .is_empty()
    );
}

/// The zero linearized polynomial with zero affine constant enumerates
/// every field element in canonical key order.
#[test]
fn zero_linearized_zero_affine_enumerates_full_field() {
    let all = linearized_roots(&Polynomial::<Gf8B>::zero(), b(0)).expect("all");
    assert_eq!(all.len(), Gf8B::ORDER as usize);
    let mut sorted = all.clone();
    sorted.sort_by_key(|root| poly_ring::internals::element_key::<Gf8B>(*root));
    sorted.dedup();
    assert_eq!(all, sorted);
    assert_eq!(
        poly_ring::chien_roots(&Polynomial::<Gf8B>::zero()).expect("chien"),
        poly_ring::BaseFieldRoots::All
    );
}

/// `X^2 + X`: kernel is the binary line; the solver returns the two
/// roots in canonical order and agrees with the Chien oracle.
#[test]
fn trace_kernel_roots_agree_and_verify() {
    let linearized = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1), b(1)]).expect("trace");
    let roots = linearized_roots(&linearized, b(0)).expect("roots");
    let scanned = poly_ring::chien_roots(&linearized)
        .expect("chien")
        .into_finite()
        .expect("finite");
    assert_eq!(roots, scanned);
    assert_eq!(roots.len(), 2);
    for root in &roots {
        let mut val = b(0);
        val = val.add(root.mul(b(1)));
        val = val.add(root.square().mul(b(1)));
        assert!(val.is_zero());
    }
}

/// `X^4 + X^2 + X` has a non-trivial GF(2)-kernel over Gf8B; the
/// solver enumerates every coset root and agrees with the Chien scan
/// on the ordinary polynomial view.
#[test]
fn degree_4_linearized_verified_against_chien() {
    let mut coefficients = vec![b(0); 5];
    coefficients[1] = b(1);
    coefficients[2] = b(1);
    coefficients[4] = b(1);
    let linearized = Polynomial::<Gf8B>::from_coefficients(&coefficients).expect("L");
    for affine in [b(0), b(3), b(0x1f), b(0x7a)] {
        let roots = linearized_roots(&linearized, affine).expect("roots");
        let shifted = {
            let mut s = linearized.clone();
            s.set_coefficient(0, affine).expect("const");
            s
        };
        let chien = poly_ring::chien_roots(&shifted)
            .expect("chien")
            .into_finite()
            .expect("finite");
        assert_eq!(roots.len(), chien.len());
        for root in &roots {
            assert!(chien.contains(root));
        }
    }
}

/// Over Gf16, `X^8 + X^4 + X^2 + X` is 2-linearized; its root set over
/// several affine cosets matches the Chien scan (`element_key` order).
#[test]
fn g16_varying_affine_constant_matches_chien() {
    use fgf::gf16;

    let mut coefficients = vec![gf16::Elem::from_raw(0); 9];
    coefficients[1] = gf16::Elem::from_raw(1);
    coefficients[2] = gf16::Elem::from_raw(1);
    coefficients[4] = gf16::Elem::from_raw(1);
    coefficients[8] = gf16::Elem::from_raw(1);
    let linearized = Polynomial::<Gf16>::from_coefficients(&coefficients).expect("L");
    for affine in [
        gf16::Elem::from_raw(0),
        gf16::Elem::from_raw(5),
        gf16::Elem::from_raw(0x13),
        gf16::Elem::from_raw(0x81),
    ] {
        let roots = linearized_roots(&linearized, affine).expect("roots");
        let shifted = {
            let mut s = linearized.clone();
            s.set_coefficient(0, affine).expect("const");
            s
        };
        let chien = poly_ring::chien_roots(&shifted)
            .expect("chien")
            .into_finite()
            .expect("finite");
        assert_eq!(roots, chien);
    }
}

/// Every returned root from an inconsistent affine coset is empty (the
/// solver's eliminator detects the contradiction).
#[test]
fn inconsistent_affine_coset_returns_empty() {
    // L(X) = X^2 + X; the affine constant 0x20 is outside its image.
    let linearized = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1), b(1)]).expect("trace");
    let roots = linearized_roots(&linearized, b(0x20)).expect("empty");
    assert!(roots.is_empty());
    let shifted = {
        let mut s = linearized.clone();
        s.set_coefficient(0, b(0x20)).expect("const");
        s
    };
    assert_eq!(
        poly_ring::chien_roots(&shifted)
            .expect("chien")
            .into_finite(),
        Some(Vec::new())
    );
}

/// A coefficient at a non-power-of-two degree is rejected with the
/// offending degree in the payload. Degree zero is also rejected.
#[test]
fn non_linearized_coefficient_names_the_offending_degree() {
    assert_eq!(
        linearized_roots(
            &Polynomial::<Gf8B>::from_coefficients(&[b(1)]).expect("const"),
            b(0)
        )
        .map(|_| ()),
        Err(RootError::NotLinearized { degree: 0 })
    );
    assert_eq!(
        linearized_roots(
            &Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1), b(0), b(1)]).expect("cubic"),
            b(0)
        )
        .map(|_| ()),
        Err(RootError::NotLinearized { degree: 3 })
    );
}
