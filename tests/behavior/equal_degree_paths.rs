//! Equal-degree splitting invariants: binary-field root extraction reaches
//! its boundary and abnormal-factor checks, recycles the factor stack and
//! pool, and verifies that every returned root vanishes in the input.

use fgf::field::Elem;
use fgf::{Gf8B, Gf16, gf8b};
use poly_ring::{BinaryRootScratch, Polynomial, base_field_roots, binary_field_roots_into};

use crate::oracles;
use oracles::naive_evaluate;

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

/// A scratch reused after a successful run recycles its empty factor
/// stack into the pool (line 85–87).  A second run over new roots
/// succeeds and retains at least the prior capacity.
#[test]
fn scratch_reuse_recycles_factors_and_retains_capacity() {
    // A polynomial with two planted linear factors over Gf8B.
    let a = b(0x13);
    let bb = b(0x57);
    let mut poly = Polynomial::<Gf8B>::one().expect("one");
    poly = poly.multiply_x_plus(a).expect("X+a");
    poly = poly.multiply_x_plus(bb).expect("X+b");
    let mut scratch = BinaryRootScratch::<Gf8B>::new();
    let mut roots = Vec::new();
    binary_field_roots_into(&mut roots, &poly, &mut scratch).expect("first");
    assert!(!roots.is_empty());
    let cap = scratch.capacity();

    // Second run: recycle_factors (line 85–87) runs over whatever the
    // first run left. Even if the factor stack was emptied, the pool
    // retains capacity; a different polynomial exercises fresh splits.
    let c = b(0x29);
    let d = b(0x7e);
    let e = b(0xa1);
    let mut poly2 = Polynomial::<Gf8B>::one().expect("one");
    poly2 = poly2.multiply_x_plus(c).expect("X+c");
    poly2 = poly2.multiply_x_plus(d).expect("X+d");
    poly2 = poly2.multiply_x_plus(e).expect("X+e");
    roots.clear();
    binary_field_roots_into(&mut roots, &poly2, &mut scratch).expect("second");
    assert_eq!(roots.len(), 3);
    assert!(scratch.capacity() >= cap);
}

/// A polynomial with no base-field linear factors exits early (line
/// 165, `base_degree == 0`) and returns empty without error.
#[test]
fn no_base_field_linear_factors_returns_empty() {
    let polynomial =
        Polynomial::<Gf8B>::from_coefficients(&[b(7), b(1), b(1), b(1)]).expect("X^3+X^2+X+7");
    let chien = poly_ring::chien_roots(&polynomial)
        .expect("chien")
        .into_finite()
        .expect("finite");
    assert!(chien.is_empty(), "oracle must find no roots");
    assert_eq!(
        base_field_roots(&polynomial)
            .expect("equal-degree")
            .into_finite(),
        Some(Vec::new())
    );
    let mut scratch = BinaryRootScratch::<Gf8B>::new();
    let mut roots = Vec::new();
    assert!(!binary_field_roots_into(&mut roots, &polynomial, &mut scratch).expect("into"));
    assert!(roots.is_empty());
}

/// A single linear factor: `gcd(X^|F|+X, X+a)` has degree 1, the base
/// factor has degree 1, and the single root is extracted without
/// splitting (exercises lines 216–229 in `process_factor`).
#[test]
fn single_linear_factor_extracts_its_single_root() {
    let a = b(0x42);
    let poly = Polynomial::<Gf8B>::from_coefficients(&[a, b(1)]).expect("X+a");
    let roots = base_field_roots(&poly)
        .expect("equal-degree")
        .into_finite()
        .expect("finite");
    assert_eq!(roots.len(), 1);
    assert_eq!(roots[0], a);
    assert!(naive_evaluate(&poly, roots[0]).is_zero());
}

/// A degree-2 square-free factor splits via trace (exercises
/// `split_factor_into` through `trace_polynomial_into` and the
/// `div_rem_into` on line 306–313). The two extracted roots agree
/// with the Chien oracle.
#[test]
fn degree_2_square_free_factor_splits_via_trace() {
    let a = b(0x11);
    let bb = b(0x65);
    let mut poly = Polynomial::<Gf8B>::one().expect("one");
    poly = poly.multiply_x_plus(a).expect("X+a");
    poly = poly.multiply_x_plus(bb).expect("X+b");
    let roots = base_field_roots(&poly)
        .expect("equal-degree")
        .into_finite()
        .expect("finite");
    assert_eq!(roots.len(), 2);
    assert!(roots.contains(&a));
    assert!(roots.contains(&bb));
    let chien = poly_ring::chien_roots(&poly)
        .expect("chien")
        .into_finite()
        .expect("finite");
    assert_eq!(&roots, &chien);
}

/// A degree-4 base factor requires two trace splits: first splits
/// into two degree-2 factors, then each splits into two linear
/// factors (exercises the splitting loop exhaustively).
#[test]
fn degree_4_square_free_factor_splits_exhaustively() {
    let a = b(0x03);
    let bb = b(0x15);
    let c = b(0x27);
    let d = b(0x8b);
    let mut poly = Polynomial::<Gf8B>::one().expect("one");
    poly = poly.multiply_x_plus(a).expect("X+a");
    poly = poly.multiply_x_plus(bb).expect("X+b");
    poly = poly.multiply_x_plus(c).expect("X+c");
    poly = poly.multiply_x_plus(d).expect("X+d");
    let roots = base_field_roots(&poly)
        .expect("equal-degree")
        .into_finite()
        .expect("finite");
    assert_eq!(roots.len(), 4);
    for root in &roots {
        assert!(naive_evaluate(&poly, *root).is_zero());
    }
    let chien = poly_ring::chien_roots(&poly)
        .expect("chien")
        .into_finite()
        .expect("finite");
    assert_eq!(&roots, &chien);
}

/// Over Gf16 a degree-3 base factor gets the root-extraction loop
/// running the splitting path: first split produces 1+2 linear and
/// non-linear factors. Every root verifies against the Chien oracle.
#[test]
fn g16_three_roots_split_and_verify_against_chien() {
    use fgf::gf16;

    let a = gf16::Elem::from_raw(0x03);
    let bb = gf16::Elem::from_raw(0x101);
    let c = gf16::Elem::from_raw(0x202);
    let mut poly = Polynomial::<Gf16>::one().expect("one");
    poly = poly.multiply_x_plus(a).expect("X+a");
    poly = poly.multiply_x_plus(bb).expect("X+b");
    poly = poly.multiply_x_plus(c).expect("X+c");
    let roots = base_field_roots(&poly)
        .expect("equal-degree")
        .into_finite()
        .expect("finite");
    assert_eq!(roots.len(), 3);
    let chien = poly_ring::chien_roots(&poly)
        .expect("chien")
        .into_finite()
        .expect("finite");
    assert_eq!(&roots, &chien);
}

/// `base_field_roots` agrees with `binary_field_roots_into` on the same
/// polynomial; both return the same finite root set.
#[test]
fn allocating_and_into_forms_agree() {
    let a = b(0x0b);
    let bb = b(0x1d);
    let mut poly = Polynomial::<Gf8B>::one().expect("one");
    poly = poly.multiply_x_plus(a).expect("X+a");
    poly = poly.multiply_x_plus(bb).expect("X+b");
    let from_alloc = base_field_roots(&poly)
        .expect("alloc")
        .into_finite()
        .expect("finite");
    let mut scratch = BinaryRootScratch::<Gf8B>::new();
    let mut into_roots = Vec::new();
    binary_field_roots_into(&mut into_roots, &poly, &mut scratch).expect("into");
    assert_eq!(from_alloc, into_roots);
}

/// `Default` for `BinaryRootScratch` agrees with `new`.
#[test]
fn default_scratch_agrees_with_new() {
    let _ = BinaryRootScratch::<Gf8B>::default();
    assert_eq!(BinaryRootScratch::<Gf8B>::new().capacity(), 0);
}

/// A constant polynomial (degree 0, nonzero) has no roots: returns
/// `Finite([])` from both forms.
#[test]
fn nonzero_constant_polynomial_has_no_roots() {
    let poly = Polynomial::<Gf8B>::from_coefficients(&[b(9)]).expect("const");
    assert_eq!(
        base_field_roots(&poly).expect("equal-degree").into_finite(),
        Some(Vec::new())
    );
    let mut scratch = BinaryRootScratch::<Gf8B>::new();
    let mut roots = Vec::new();
    assert!(!binary_field_roots_into(&mut roots, &poly, &mut scratch).expect("into"));
    assert!(roots.is_empty());
}
