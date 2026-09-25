//! Portable domains behavior and error boundaries.

#[cfg(feature = "fft")]
use fgf::Gf16;
use fgf::field::Field;
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Mersenne31};
use poly_ring::{DomainScratch, EvaluationBackend, EvaluationDomain};

use crate::oracles;
use oracles::{noise, noise_poly};

fn distinct_points<F: FieldKernels>(len: usize, seed: u64) -> Vec<F::Elem> {
    let mut points = Vec::new();
    let mut state = seed;
    while points.len() < len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let candidate = F::decode(&state.to_le_bytes()[..F::BYTES]);
        if !points.contains(&candidate) {
            points.push(candidate);
        }
    }
    points
}

/// Transform-free subspaces enumerate the binary span in index order and
/// evaluate through the arbitrary paths.
#[cfg(not(feature = "fft"))]
#[test]
fn transform_free_subspaces_enumerate_and_evaluate() {
    let one = <Gf8B as Field>::Elem::ONE;
    let generator = <Gf8B as Field>::GENERATOR;
    let basis = [one, generator];
    let domain = EvaluationDomain::<Gf8B>::additive_subspace_with_basis(4, &basis).expect("domain");
    assert_eq!(domain.len(), 4);
    assert!(!domain.is_empty());
    // Index order: point i is the XOR of basis[j] over set bits j of i.
    assert_eq!(domain.points()[0], <Gf8B as Field>::Elem::ZERO);
    assert_eq!(domain.points()[1], one);
    assert_eq!(domain.points()[2], generator);
    assert_eq!(domain.points()[3], one.add(generator));
    // Eight points exceed the Horner crossover: the tree backend serves.
    let big = EvaluationDomain::<Gf8B>::additive_subspace(32).expect("big");
    assert_eq!(big.len(), 32);
    assert_eq!(big.backend(), EvaluationBackend::SubproductTree);
    assert_eq!(big.points()[0], <Gf8B as Field>::Elem::ZERO);

    let polynomial = noise_poly::<Gf8B>(9, 0xE100);
    let mut scratch = DomainScratch::<Gf8B>::new();
    let values = domain
        .evaluate(&polynomial, &mut scratch)
        .expect("evaluate");
    for (point, value) in domain.points().iter().zip(&values) {
        assert_eq!(*value, polynomial.evaluate(*point));
    }
    // Interpolation round-trips through the arbitrary path.
    let interpolant = domain
        .interpolate(&values, &mut scratch)
        .expect("interpolate");
    for (point, value) in domain.points().iter().zip(&values) {
        assert_eq!(interpolant.evaluate(*point), *value);
    }
}

/// Transform-free cosets shift every subspace point by the coset shift.
#[cfg(not(feature = "fft"))]
#[test]
fn transform_free_cosets_shift_the_span() {
    let one = <Gf8B as Field>::Elem::ONE;
    let generator = <Gf8B as Field>::GENERATOR;
    let basis = [one, generator];
    let shift = generator.mul(generator);
    let coset = EvaluationDomain::<Gf8B>::affine_coset_with_basis(4, &basis, shift).expect("coset");
    let subspace =
        EvaluationDomain::<Gf8B>::additive_subspace_with_basis(4, &basis).expect("subspace");
    assert_eq!(coset.len(), 4);
    for (coset_point, subspace_point) in coset.points().iter().zip(subspace.points()) {
        assert_eq!(*coset_point, shift.add(*subspace_point));
    }
    let default = EvaluationDomain::<Gf8B>::affine_coset(4, shift).expect("default");
    assert_eq!(default.len(), 4);
    // Default bit-basis point 0 is the shift itself.
    assert_eq!(default.points()[0], shift);

    let polynomial = noise_poly::<Gf8B>(9, 0xE101);
    let mut scratch = DomainScratch::<Gf8B>::new();
    let values = coset.evaluate(&polynomial, &mut scratch).expect("evaluate");
    for (point, value) in coset.points().iter().zip(&values) {
        assert_eq!(*value, polynomial.evaluate(*point));
    }
}

/// Transform-free validation: short bases, dependent bases, and
/// over-capacity sizes each name their violation.
#[cfg(not(feature = "fft"))]
#[test]
fn transform_free_subspace_validation() {
    use poly_ring::DomainError;

    let short = noise::<Gf8B>(1, 0xE102);
    assert_eq!(
        EvaluationDomain::<Gf8B>::additive_subspace_with_basis(8, &short).map(|_| ()),
        Err(DomainError::NotSubspace { size: 8, limit: 8 })
    );
    let generator = <Gf8B as Field>::GENERATOR;
    assert_eq!(
        EvaluationDomain::<Gf8B>::additive_subspace_with_basis(4, &[generator, generator])
            .map(|_| ()),
        Err(DomainError::NotSubspace { size: 4, limit: 4 })
    );
    assert!(
        EvaluationDomain::<Gf8B>::additive_subspace(512).is_err()
            && EvaluationDomain::<Gf8B>::affine_coset(512, generator).is_err()
    );
}

/// With the transform, subspace and coset domains evaluate and interpolate
/// through it; the interpolation of a too-long value vector names both
/// lengths.
#[cfg(feature = "fft")]
#[test]
fn structured_domains_evaluate_and_reject_length_mismatch() {
    let subspace = EvaluationDomain::<Gf16>::additive_subspace(16).expect("subspace");
    assert_eq!(subspace.backend(), EvaluationBackend::ButterflyFftAdditive);
    assert_eq!(subspace.len(), 16);
    let shift = noise::<Gf16>(1, 0xE103)[0];
    let coset = EvaluationDomain::<Gf16>::affine_coset(16, shift).expect("coset");
    assert_eq!(coset.backend(), EvaluationBackend::ButterflyFftAffineCoset);

    let polynomial = noise_poly::<Gf16>(11, 0xE104);
    let mut scratch = DomainScratch::<Gf16>::new();
    for domain in [&subspace, &coset] {
        let values = domain
            .evaluate(&polynomial, &mut scratch)
            .expect("evaluate");
        assert_eq!(values.len(), 16);
        for (point, value) in domain.points().iter().zip(&values) {
            assert_eq!(*value, polynomial.evaluate(*point));
        }
        let interpolant = domain
            .interpolate(&values, &mut scratch)
            .expect("interpolate");
        for (point, value) in domain.points().iter().zip(&values) {
            assert_eq!(interpolant.evaluate(*point), *value);
        }
    }
    // Short value vectors surface the interpolation geometry failure.
    let short = noise::<Gf16>(3, 0xE105);
    assert!(subspace.interpolate(&short, &mut scratch).is_err());
    assert!(coset.interpolate(&short, &mut scratch).is_err());
    // Explicit-basis forms enumerate the same points as the defaults.
    let one = <Gf16 as Field>::Elem::ONE;
    let generator = <Gf16 as Field>::GENERATOR;
    let explicit = EvaluationDomain::<Gf16>::additive_subspace_with_basis(4, &[one, generator])
        .expect("basis");
    assert_eq!(explicit.len(), 4);
    let explicit_coset =
        EvaluationDomain::<Gf16>::affine_coset_with_basis(4, &[one, generator], shift)
            .expect("coset basis");
    for (coset_point, subspace_point) in explicit_coset.points().iter().zip(explicit.points()) {
        assert_eq!(*coset_point, shift.add(*subspace_point));
    }
}

/// Goldilocks/Mersenne coverage for the scalar multipoint and Newton paths.
#[test]
fn prime_field_scalar_paths_agree() {
    let points = distinct_points::<Mersenne31>(20, 0xE108);
    let polynomial = noise_poly::<Mersenne31>(13, 0xE109);
    let values = poly_ring::evaluate_multipoint(&polynomial, &points).expect("multipoint");
    for (point, value) in points.iter().zip(&values) {
        assert_eq!(*value, polynomial.evaluate(*point));
    }
    // Newton interpolation over Gf8B round-trips through its own points,
    // and Lagrange agrees with Newton on the same support.
    let sample_points = distinct_points::<Gf8B>(6, 0xE112);
    let sample = noise::<Gf8B>(6, 0xE113);
    let interpolant =
        poly_ring::interpolate_newton::<Gf8B>(&sample_points, &sample).expect("newton");
    for (point, value) in sample_points.iter().zip(&sample) {
        assert_eq!(interpolant.evaluate(*point), *value);
    }
    // Lagrange agrees with Newton on the same support.
    let lagrange =
        poly_ring::interpolate_lagrange::<Gf8B>(&sample_points, &sample).expect("lagrange");
    assert_eq!(interpolant, lagrange);
}
