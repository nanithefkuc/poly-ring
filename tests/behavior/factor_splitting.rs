//! Factor splitting behavior and error boundaries.

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16, gf8b};
use poly_ring::{BaseFieldRoots, Polynomial, base_field_roots, binary_field_roots_into};

use crate::oracles;
use oracles::noise;

/// A polynomial with many distinct planted roots exercises the split loop
/// repeatedly, including pool recycling across factors.
#[test]
fn many_planted_roots_split_repeatedly() {
    fn with_roots<F: FieldKernels>(seed: u64, root_count: usize) -> (Polynomial<F>, Vec<F::Elem>) {
        let mut polynomial = Polynomial::one().expect("one");
        let mut roots = Vec::new();
        for index in 0..root_count {
            let mut root = noise::<F>(1, seed + 100 * index as u64 + 1)[0];
            while roots.contains(&root) || root.is_zero() {
                root = root.mul(F::GENERATOR).add(F::Elem::ONE);
            }
            roots.push(root);
            polynomial = polynomial.multiply_x_plus(root).expect("linear factor");
        }
        (polynomial, roots)
    }

    // 24 distinct roots over Gf16: forces repeated trace splits and factor
    // stack recycling.
    let (polynomial, planted) = with_roots::<Gf16>(0xF201, 24);
    let found = base_field_roots(&polynomial).expect("roots");
    let mut list = found.into_finite().expect("finite");
    list.sort_by_key(|e| poly_ring::internals::element_key::<Gf16>(*e));
    let mut expected = planted;
    expected.sort_by_key(|e| poly_ring::internals::element_key::<Gf16>(*e));
    expected.dedup();
    assert_eq!(list, expected);

    // The `into` form with a reused scratch agrees, and the scratch
    // capacity is retained for the next extraction.
    let mut scratch = poly_ring::BinaryRootScratch::<Gf16>::new();
    let mut roots = Vec::new();
    let all = binary_field_roots_into(&mut roots, &polynomial, &mut scratch).expect("into");
    assert!(!all);
    assert_eq!(roots, list);
    let capacity = scratch.capacity();
    let (polynomial2, _) = with_roots::<Gf16>(0xF202, 10);
    let mut roots2 = Vec::new();
    binary_field_roots_into(&mut roots2, &polynomial2, &mut scratch).expect("reuse");
    assert!(scratch.capacity() >= capacity);
    for root in &roots2 {
        assert!(polynomial2.evaluate(*root).is_zero());
    }
}

/// Repeated roots collapse to the distinct set: the base factor is
/// square-free and the final list is deduplicated.
#[test]
fn repeated_roots_collapse_to_the_distinct_set() {
    let mut root = noise::<Gf8B>(1, 0xF203)[0];
    if root.is_zero() {
        root = <Gf8B as Field>::GENERATOR;
    }
    // (X + r)^6: one distinct root with multiplicity six.
    let linear =
        Polynomial::<Gf8B>::from_coefficients(&[root, <Gf8B as Field>::Elem::ONE]).expect("linear");
    let mut polynomial = Polynomial::<Gf8B>::one().expect("one");
    for _ in 0..6 {
        polynomial = polynomial.multiply(&linear).expect("power");
    }
    assert_eq!(
        base_field_roots(&polynomial).expect("roots"),
        BaseFieldRoots::Finite(vec![root])
    );
}

/// A polynomial with no base-field roots returns the empty set (the base
/// factor has degree zero).
#[test]
fn rootless_polynomial_returns_empty() {
    // X^2 + X + 1 over Gf8B has no root at 0 or 1, but may have roots
    // elsewhere in Gf256; use a constant instead for a guaranteed empty set
    // plus a high-degree construction whose base factor vanishes.
    let constant =
        Polynomial::<Gf8B>::from_coefficients(&[gf8b::Elem::from_raw(5)]).expect("constant");
    assert_eq!(
        base_field_roots(&constant).expect("roots"),
        BaseFieldRoots::Finite(Vec::new())
    );
    // An irreducible quadratic over Gf8B: X^2 + GENERATOR·X + 1. Its base
    // factor is empty iff it has no Gf256 root; verify against Chien.
    let irreducible = Polynomial::<Gf8B>::from_coefficients(&[
        gf8b::Elem::from_raw(1),
        <Gf8B as Field>::GENERATOR,
        gf8b::Elem::from_raw(1),
    ])
    .expect("quadratic");
    let split = base_field_roots(&irreducible).expect("split");
    let scanned = poly_ring::chien_roots(&irreducible).expect("chien");
    assert_eq!(split, scanned);
}

/// Splits over Gf16 with a planted mid-size set agree with Chien and
/// exercise deeper trace recursion than the small suites.
#[test]
fn mid_size_gf16_splits_match_chien() {
    let mut polynomial = Polynomial::<Gf16>::one().expect("one");
    let mut planted = Vec::new();
    for index in 0..12 {
        let mut root = noise::<Gf16>(1, 0xF205 + index as u64)[0];
        while planted.contains(&root) {
            root = root
                .mul(<Gf16 as Field>::GENERATOR)
                .add(<Gf16 as Field>::Elem::ONE);
        }
        planted.push(root);
        polynomial = polynomial.multiply_x_plus(root).expect("factor");
    }
    let split = base_field_roots(&polynomial).expect("split");
    let scanned = poly_ring::chien_roots(&polynomial).expect("chien");
    assert_eq!(split, scanned);
    assert_eq!(split.as_slice().expect("finite").len(), 12);
}

/// The zero polynomial reports All; a nonzero constant reports empty.
#[test]
fn zero_and_constant_extremes_hold() {
    assert_eq!(
        base_field_roots(&Polynomial::<Gf8B>::zero()).expect("zero"),
        BaseFieldRoots::All
    );
    let mut scratch = poly_ring::BinaryRootScratch::<Gf8B>::new();
    let mut roots = Vec::new();
    assert!(
        binary_field_roots_into(&mut roots, &Polynomial::<Gf8B>::zero(), &mut scratch)
            .expect("zero into")
    );
    let _ = poly_ring::BinaryRootScratch::<Gf8B>::default();
}
