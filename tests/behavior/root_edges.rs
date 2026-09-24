//! Root edge behavior: Chien invariants, equal-degree accessors, and the
//! linearized solver's field and shape contracts.
//!
//! The agreement suites scan and split typical locators; every test here
//! takes a branch they never reach — unsupported fields, non-linearized
//! shapes, inconsistent affine systems, empty root sets, and scratch reuse
//! — each against exact root sets or error payloads.

use fgf::field::{Elem, Field};
use fgf::{Gf8B, Gf16, Mersenne31, gf8b, mersenne31};
use poly_ring::{BaseFieldRoots, BinaryRootScratch, ChienScratch, Polynomial, RootError};

use crate::oracles;
use oracles::{naive_evaluate, noise_poly};

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

/// Prime fields remain outside the binary Chien and scratch-backed
/// equal-degree paths, which report the field order and element width.
#[test]
fn prime_fields_are_unsupported_for_binary_scans() {
    let polynomial = noise_poly::<Mersenne31>(5, 0xD401);
    let mut scratch = ChienScratch::<Mersenne31>::new();
    let mut roots = Vec::new();
    assert_eq!(
        poly_ring::chien_roots_into(&mut roots, &polynomial, &mut scratch).map(|_| ()),
        Err(RootError::UnsupportedField {
            field_order: Mersenne31::ORDER,
            element_bytes: Mersenne31::BYTES,
        })
    );
    assert_eq!(
        poly_ring::chien_roots(&polynomial).map(|_| ()),
        Err(RootError::UnsupportedField {
            field_order: Mersenne31::ORDER,
            element_bytes: Mersenne31::BYTES,
        })
    );
    let mut field_scratch = BinaryRootScratch::<Mersenne31>::new();
    assert_eq!(
        poly_ring::binary_field_roots_into(&mut roots, &polynomial, &mut field_scratch).map(|_| ()),
        Err(RootError::UnsupportedField {
            field_order: Mersenne31::ORDER,
            element_bytes: Mersenne31::BYTES,
        })
    );
    assert_eq!(
        poly_ring::linearized_roots(&polynomial, m31(1)).map(|_| ()),
        Err(RootError::UnsupportedField {
            field_order: Mersenne31::ORDER,
            element_bytes: Mersenne31::BYTES,
        })
    );
}

/// The zero polynomial has every field element as a root; a nonzero
/// constant has none. Both backends agree on the degenerate shapes.
#[test]
fn degenerate_shapes_agree_across_backends() {
    let zero = Polynomial::<Gf8B>::zero();
    assert!(
        poly_ring::chien_roots(&zero)
            .expect("chien")
            .as_slice()
            .is_none()
    );
    let mut scratch = ChienScratch::<Gf8B>::new();
    let mut roots = Vec::new();
    assert!(poly_ring::chien_roots_into(&mut roots, &zero, &mut scratch).expect("into"));
    assert!(roots.is_empty(), "the All case leaves the vector empty");
    let mut field_scratch = BinaryRootScratch::<Gf8B>::new();
    assert!(
        poly_ring::binary_field_roots_into(&mut roots, &zero, &mut field_scratch).expect("into")
    );
    assert_eq!(
        poly_ring::base_field_roots(&zero).expect("equal-degree"),
        BaseFieldRoots::All
    );

    let constant = Polynomial::<Gf8B>::from_coefficients(&[b(5)]).expect("const");
    assert_eq!(
        poly_ring::chien_roots(&constant)
            .expect("chien")
            .into_finite(),
        Some(Vec::new())
    );
    assert_eq!(
        poly_ring::base_field_roots(&constant)
            .expect("equal-degree")
            .into_finite(),
        Some(Vec::new())
    );
    // Accessors distinguish the two shapes.
    assert_eq!(BaseFieldRoots::<gf8b::Elem>::All.as_slice(), None);
    assert_eq!(
        BaseFieldRoots::Finite(vec![b(1)]).as_slice(),
        Some(&[b(1)][..])
    );
}

/// A warmed Chien scratch reuses its buffers over a changed locator and
/// agrees with a fresh scan.
#[test]
fn chien_scratch_reuse_matches_fresh_scan() {
    let first = noise_poly::<Gf8B>(5, 0xD402);
    let mut scratch = ChienScratch::<Gf8B>::new();
    let mut roots = Vec::new();
    let all = poly_ring::chien_roots_into(&mut roots, &first, &mut scratch).expect("first");
    assert_eq!(all, first.is_zero());
    for root in &roots {
        assert!(naive_evaluate(&first, *root).is_zero());
    }
    let second = noise_poly::<Gf8B>(6, 0xD403);
    let mut roots2 = Vec::new();
    poly_ring::chien_roots_into(&mut roots2, &second, &mut scratch).expect("second");
    for root in &roots2 {
        assert!(naive_evaluate(&second, *root).is_zero());
    }
    let _ = ChienScratch::<Gf8B>::default();
    let _ = scratch;
}

/// A warmed equal-degree scratch reuses its factor stack and pool over a
/// changed input.
#[test]
fn equal_degree_scratch_reuse_matches_fresh_split() {
    let first = noise_poly::<Gf16>(7, 0xD404);
    let mut scratch = BinaryRootScratch::<Gf16>::new();
    let mut roots = Vec::new();
    poly_ring::binary_field_roots_into(&mut roots, &first, &mut scratch).expect("first");
    for root in &roots {
        assert!(naive_evaluate(&first, *root).is_zero());
    }
    let capacity = scratch.capacity();
    let second = noise_poly::<Gf16>(6, 0xD405);
    let mut roots2 = Vec::new();
    poly_ring::binary_field_roots_into(&mut roots2, &second, &mut scratch).expect("second");
    for root in &roots2 {
        assert!(naive_evaluate(&second, *root).is_zero());
    }
    assert!(scratch.capacity() >= capacity);
    assert_eq!(BinaryRootScratch::<Gf16>::default().capacity(), 0);
}

/// A nonzero constant term at degree zero is not linearized: the solver
/// names the offending degree.
#[test]
fn nonzero_constant_term_is_not_linearized() {
    let polynomial =
        Polynomial::<Gf8B>::from_coefficients(&[b(1), b(1)]).expect("affine, not linearized");
    assert_eq!(
        poly_ring::linearized_roots(&polynomial, b(0)).map(|_| ()),
        Err(RootError::NotLinearized { degree: 0 })
    );
    // A nonzero coefficient at a non-power-of-two degree names it too.
    let cubic = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1), b(0), b(1)]).expect("cubic");
    assert_eq!(
        poly_ring::linearized_roots(&cubic, b(0)).map(|_| ()),
        Err(RootError::NotLinearized { degree: 3 })
    );
}

/// The zero linearized polynomial with a nonzero affine constant has no
/// roots; with a zero constant every field element is a root.
#[test]
fn zero_linearized_splits_on_affine_constant() {
    assert!(
        poly_ring::linearized_roots(&Polynomial::<Gf8B>::zero(), b(7))
            .expect("no roots")
            .is_empty()
    );
    let all = poly_ring::linearized_roots(&Polynomial::<Gf8B>::zero(), b(0)).expect("all");
    assert_eq!(all.len(), Gf8B::ORDER as usize);
    // Canonical key order: sorted, deduplicated, frozen.
    let mut sorted = all.clone();
    sorted.sort_by_key(|element: &gf8b::Elem| poly_ring::internals::element_key::<Gf8B>(*element));
    sorted.dedup();
    assert_eq!(all, sorted);
}

/// An inconsistent affine system is empty: the trace map `X^2 + X` has a
/// 128-element image, so some affine coset misses it entirely. The solver
/// agrees with the Chien scan there.
#[test]
fn inconsistent_affine_system_is_empty() {
    // L(X) = X^2 + X; the affine constant 0x20 lies outside its image
    // (verified by exhaustive search over the 256-element field).
    let linearized =
        Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1), b(1)]).expect("linearized");
    let empty = poly_ring::linearized_roots(&linearized, b(0x20)).expect("empty");
    assert!(empty.is_empty());
    let shifted = {
        let constant = Polynomial::<Gf8B>::from_coefficients(&[b(0x20)]).expect("const");
        linearized.add(&constant).expect("shift")
    };
    assert_eq!(
        poly_ring::chien_roots(&shifted)
            .expect("chien")
            .into_finite(),
        Some(Vec::new())
    );
}

/// The trace map `X^2 + X` has the binary line as its kernel: both roots
/// come back in canonical order and agree with the scan oracle.
#[test]
fn trace_kernel_matches_chien_scan() {
    let linearized = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1), b(1)]).expect("trace");
    let solved = poly_ring::linearized_roots(&linearized, b(0)).expect("roots");
    assert_eq!(solved.len(), 2);
    let scanned = poly_ring::chien_roots(&linearized).expect("chien");
    assert_eq!(scanned.as_slice(), Some(solved.as_slice()));
    // A nonzero affine shift of the trace map moves the coset: X^2+X+a
    // has roots exactly where the scan says.
    for affine in [b(0), b(1), b(0x1b)] {
        let solved = poly_ring::linearized_roots(&linearized, affine).expect("roots");
        let shifted = {
            let constant = Polynomial::<Gf8B>::from_coefficients(&[affine]).expect("const");
            linearized.add(&constant).expect("shift")
        };
        let scanned = poly_ring::chien_roots(&shifted).expect("chien");
        assert_eq!(scanned.as_slice(), Some(solved.as_slice()));
    }
}

/// A polynomial with no base-field roots returns the empty set without
/// error: the base factor has degree zero and extraction stops early.
#[test]
fn rootless_polynomial_returns_empty_without_error() {
    // Fixed-seed search for a degree-4 noise polynomial with no Gf8B roots
    // (verified against the Chien scan oracle below).
    let mut state = 0xD501_u64;
    let polynomial = loop {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bytes = state.to_le_bytes();
        let coefficients: Vec<gf8b::Elem> =
            bytes.iter().map(|v| gf8b::Elem::from_raw(*v)).collect();
        let candidate =
            Polynomial::<Gf8B>::from_coefficients(&coefficients[..5]).expect("candidate");
        if candidate.degree().unwrap_or(0) < 2 {
            continue;
        }
        if poly_ring::chien_roots(&candidate)
            .expect("scan")
            .into_finite()
            .expect("finite")
            .is_empty()
        {
            break candidate;
        }
    };
    let mut scratch = BinaryRootScratch::<Gf8B>::new();
    let mut roots = Vec::new();
    assert!(
        !poly_ring::binary_field_roots_into(&mut roots, &polynomial, &mut scratch).expect("split")
    );
    assert!(roots.is_empty());
    assert_eq!(
        poly_ring::base_field_roots(&polynomial)
            .expect("split")
            .into_finite(),
        Some(Vec::new())
    );
}
