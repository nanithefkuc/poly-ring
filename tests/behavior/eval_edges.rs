//! Evaluation surfaces: Newton basis errors and accessors, domain
//! backends and geometry rejections, and the subspace interpolation
//! round trip.
//!
//! Oracle agreement lives in `eval.rs`; every test here targets a branch
//! the agreement suite never takes: small-domain paths, explicit error
//! variants with their payloads, and the `fft`-gated interpolation entry
//! points.

use fgf::Gf8B;
#[cfg(feature = "fft")]
use fgf::Gf16;
#[cfg(feature = "fft")]
use fgf::Gf32;
use fgf::field::Field;
use fgf::kernel::FieldKernels;
use poly_ring::{
    ConfigError, DomainError, DomainScratch, EvalError, EvaluationBackend, EvaluationDomain,
    MultipointScratch, NewtonBasis, Polynomial, evaluate_multipoint_into, interpolate_lagrange,
};

fn noise<F: FieldKernels>(len: usize, seed: u64) -> Vec<F::Elem> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bytes = state.to_le_bytes();
            F::decode(&bytes[..F::BYTES])
        })
        .collect()
}

fn distinct_points<F: FieldKernels>(len: usize, seed: u64) -> Vec<F::Elem> {
    let mut points = Vec::new();
    let mut state = seed;
    while points.len() < len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bytes = state.to_le_bytes();
        let candidate = F::decode(&bytes[..F::BYTES]);
        if !points.contains(&candidate) {
            points.push(candidate);
        }
    }
    points
}

fn noise_poly<F: FieldKernels>(len: usize, seed: u64) -> Polynomial<F> {
    Polynomial::from_coefficients(&noise::<F>(len, seed)).expect("noise polynomial")
}

/// An empty support, an over-capacity support, and a repeated point are
/// three distinct rejections.
#[test]
fn newton_basis_construction_rejects_degenerate_supports() {
    assert_eq!(
        NewtonBasis::<Gf8B>::new(&[]).map(|_| ()),
        Err(EvalError::Domain(DomainError::Config(
            ConfigError::ZeroParameter {
                parameter: "interpolation support",
            }
        )))
    );

    let overfull = noise::<Gf8B>(300, 0xE001);
    assert_eq!(
        NewtonBasis::<Gf8B>::new(&overfull).map(|_| ()),
        Err(EvalError::Domain(DomainError::Config(
            ConfigError::FieldCapacityExceeded {
                points: 300,
                field_order: 256,
            }
        )))
    );

    let mut repeated = distinct_points::<Gf8B>(4, 0xE002);
    repeated[3] = repeated[1];
    assert_eq!(
        NewtonBasis::<Gf8B>::new(&repeated).map(|_| ()),
        Err(EvalError::Domain(DomainError::DuplicatePoint {
            first: 1,
            second: 3
        }))
    );
}

/// Accessors report the support the basis was built from.
#[test]
fn newton_basis_accessors_report_the_support() {
    let points = distinct_points::<Gf8B>(6, 0xE010);
    let basis = NewtonBasis::<Gf8B>::new(&points).expect("basis");
    assert_eq!(basis.points(), points.as_slice());
    assert_eq!(basis.len(), 6);
    assert!(!basis.is_empty());
    assert_eq!(basis.partials().len(), 6);
    // The partial products nest: N_{i+1} = N_i · (X + points[i]).
    for (index, partial) in basis.partials().iter().enumerate() {
        assert_eq!(partial.coefficient_count(), index + 1);
    }
}

/// A point/value length mismatch names both lengths.
#[test]
fn lagrange_rejects_mismatched_lengths() {
    let points = distinct_points::<Gf8B>(3, 0xE020);
    let values = noise::<Gf8B>(2, 0xE021);
    assert_eq!(
        interpolate_lagrange::<Gf8B>(&points, &values).map(|_| ()),
        Err(EvalError::Domain(DomainError::LengthMismatch {
            expected: 3,
            found: 2
        }))
    );
}

/// The default multipoint scratch evaluates exactly like a fresh one.
#[test]
fn default_multipoint_scratch_evaluates() {
    let mut scratch = MultipointScratch::<Gf8B>::default();
    let mut values = Vec::new();
    let polynomial = noise_poly::<Gf8B>(9, 0xE030);
    let points = distinct_points::<Gf8B>(5, 0xE031);
    evaluate_multipoint_into(&mut values, &polynomial, &points, &mut scratch).expect("evaluate");
    let mut horner = Vec::new();
    for point in &points {
        let mut value = <Gf8B as Field>::Elem::ZERO;
        for coefficient in polynomial.coefficients().rev() {
            value = value.mul(*point).add(coefficient);
        }
        horner.push(value);
    }
    assert_eq!(values, horner);
}

/// Arbitrary point sets are validated before any domain exists.
#[test]
fn arbitrary_domains_reject_empty_and_overfull_supports() {
    assert_eq!(
        EvaluationDomain::<Gf8B>::arbitrary(Vec::new()).map(|_| ()),
        Err(DomainError::Config(ConfigError::ZeroParameter {
            parameter: "evaluation-domain length",
        }))
    );
    let overfull = noise::<Gf8B>(300, 0xE040);
    assert_eq!(
        EvaluationDomain::<Gf8B>::arbitrary(overfull).map(|_| ()),
        Err(DomainError::Config(ConfigError::FieldCapacityExceeded {
            points: 300,
            field_order: 256,
        }))
    );
}

/// Subspace sizes are validated against the field before any basis runs.
#[test]
fn subspace_sizes_reject_non_subspaces_and_overcapacity() {
    // A short basis names the size and the power-of-two limit.
    let short = noise::<Gf8B>(1, 0xE050);
    assert_eq!(
        EvaluationDomain::<Gf8B>::additive_subspace_with_basis(8, &short).map(|_| ()),
        Err(DomainError::NotSubspace { size: 8, limit: 8 })
    );
    // A repeated basis vector is linearly dependent.
    let generator = <Gf8B as Field>::GENERATOR;
    assert_eq!(
        EvaluationDomain::<Gf8B>::additive_subspace_with_basis(4, &[generator, generator])
            .map(|_| ()),
        Err(DomainError::NotSubspace { size: 4, limit: 4 })
    );
    // 512 points cannot fit in a 256-element field.
    assert_eq!(
        EvaluationDomain::<Gf8B>::additive_subspace(512).map(|_| ()),
        Err(DomainError::Config(ConfigError::FieldCapacityExceeded {
            points: 512,
            field_order: 256,
        }))
    );
}

#[cfg(feature = "fft")]
#[test]
fn oversized_subspace_reports_the_plan_failure() {
    use butterfly_fft::kernel::ButterflyKernels;

    fn check<F: FieldKernels + ButterflyKernels>() {}
    check::<Gf32>();
    // 2^21 points fit in GF(2^32) but exceed the transform domain cap of
    // 2^20: the plan construction fails and the domain reports it.
    assert!(matches!(
        EvaluationDomain::<Gf32>::additive_subspace(1 << 21).map(|_| ()),
        Err(DomainError::TransformPlan(_))
    ));
}

/// Small arbitrary domains evaluate through Horner and interpolate through
/// Newton; the backend selector reports each route honestly.
#[cfg(feature = "fft")]
#[test]
fn small_domains_take_the_scalar_paths() {
    let mut scratch = DomainScratch::<Gf16>::new();
    let polynomial = noise_poly::<Gf16>(9, 0xE060);
    let points = distinct_points::<Gf16>(4, 0xE061);
    let domain = EvaluationDomain::arbitrary(points.clone()).expect("domain");
    assert!(!domain.is_empty());
    assert_eq!(domain.len(), 4);
    assert_eq!(domain.backend(), EvaluationBackend::Horner);

    // The Horner evaluations match per-point evaluation.
    let values = domain
        .evaluate(&polynomial, &mut scratch)
        .expect("evaluate");
    for (point, value) in points.iter().zip(&values) {
        let mut expected = <Gf16 as Field>::Elem::ZERO;
        for coefficient in polynomial.coefficients().rev() {
            expected = expected.mul(*point).add(coefficient);
        }
        assert_eq!(*value, expected);
    }

    // Four points sit below the Lagrange crossover: Newton interpolates,
    // and the round trip recovers the values.
    let interpolant = domain
        .interpolate(&values, &mut scratch)
        .expect("interpolate");
    for (point, value) in points.iter().zip(&values) {
        assert_eq!(interpolant.evaluate(*point), *value);
    }

    // A large arbitrary domain reports the tree backend.
    let many = distinct_points::<Gf16>(24, 0xE062);
    let big = EvaluationDomain::<Gf16>::arbitrary(many).expect("domain");
    assert_eq!(big.backend(), EvaluationBackend::SubproductTree);
}

/// Subspace and coset domains report their transform backends and expose
/// their plans; arbitrary domains hold no plan.
#[cfg(feature = "fft")]
#[test]
fn structured_domains_report_transform_backends() {
    let subspace = EvaluationDomain::<Gf16>::additive_subspace(16).expect("subspace");
    assert_eq!(subspace.backend(), EvaluationBackend::ButterflyFftAdditive);
    assert!(subspace.transform_plan().is_some());

    let shift = noise::<Gf16>(1, 0xE070)[0];
    let coset = EvaluationDomain::<Gf16>::affine_coset(16, shift).expect("coset");
    assert_eq!(coset.backend(), EvaluationBackend::ButterflyFftAffineCoset);
    assert!(coset.transform_plan().is_some());

    let points = distinct_points::<Gf16>(4, 0xE071);
    let domain = EvaluationDomain::<Gf16>::arbitrary(points).expect("domain");
    assert!(domain.transform_plan().is_none());
}

/// An explicit-basis coset enumerates the shifted span, verified by
/// independent Horner evaluation at every point.
#[cfg(feature = "fft")]
#[test]
fn explicit_basis_cosets_enumerate_the_shifted_span() {
    let mut scratch = DomainScratch::<Gf16>::new();
    let polynomial = noise_poly::<Gf16>(9, 0xE080);
    let one = <Gf16 as Field>::Elem::ONE;
    let generator = <Gf16 as Field>::GENERATOR;
    let basis = [one, generator];
    let shift = generator.mul(generator);
    let coset = EvaluationDomain::<Gf16>::affine_coset_with_basis(4, &basis, shift).expect("coset");
    assert_eq!(coset.len(), 4);
    let values = coset.evaluate(&polynomial, &mut scratch).expect("evaluate");
    for (point, value) in coset.points().iter().zip(&values) {
        let mut expected = <Gf16 as Field>::Elem::ZERO;
        for coefficient in polynomial.coefficients().rev() {
            expected = expected.mul(*point).add(coefficient);
        }
        assert_eq!(*value, expected);
    }
    // The coset is the shift added to every subspace point.
    let subspace =
        EvaluationDomain::<Gf16>::additive_subspace_with_basis(4, &basis).expect("subspace");
    for (coset_point, subspace_point) in coset.points().iter().zip(subspace.points()) {
        assert_eq!(*coset_point, shift.add(*subspace_point));
    }
}

/// Without the transform the same domain objects evaluate and
/// interpolate through the arbitrary paths.
#[cfg(not(feature = "fft"))]
#[test]
fn transform_free_domains_evaluate_and_interpolate() {
    let mut scratch = DomainScratch::<Gf8B>::new();
    let polynomial = noise_poly::<Gf8B>(9, 0xE090);
    let points = distinct_points::<Gf8B>(5, 0xE091);
    let domain = EvaluationDomain::arbitrary(points.clone()).expect("domain");
    assert_eq!(domain.len(), 5);
    assert!(!domain.is_empty());
    assert_eq!(domain.backend(), EvaluationBackend::Horner);
    let values = domain
        .evaluate(&polynomial, &mut scratch)
        .expect("evaluate");
    for (point, value) in points.iter().zip(&values) {
        assert_eq!(*value, polynomial.evaluate(*point));
    }
    let interpolant = domain
        .interpolate(&values, &mut scratch)
        .expect("interpolate");
    for (point, value) in points.iter().zip(&values) {
        assert_eq!(interpolant.evaluate(*point), *value);
    }
    let mut output = Polynomial::<Gf8B>::zero();
    domain
        .interpolate_into(&values, &mut scratch, &mut output)
        .expect("interpolate into");
    assert_eq!(output, interpolant);
    // A short value vector names both lengths.
    assert_eq!(
        domain
            .interpolate(&noise::<Gf8B>(2, 0xE092), &mut scratch)
            .map(|_| ()),
        Err(EvalError::Domain(DomainError::LengthMismatch {
            expected: 5,
            found: 2
        }))
    );
}

/// The subspace interpolant inverts the subspace evaluation, checked by
/// independent Horner evaluation at every plan point.
#[cfg(feature = "fft")]
#[test]
fn subspace_interpolation_inverts_subspace_evaluation() {
    use butterfly_fft::transform::TransformPlan;
    use poly_ring::{TransformScratch, evaluate_subspace, interpolate_subspace};

    let plan = TransformPlan::<Gf16>::new(16).expect("plan");
    let mut scratch = TransformScratch::default();
    let polynomial = noise_poly::<Gf16>(11, 0xE0A0);
    let values = evaluate_subspace(&polynomial, &plan, &mut scratch).expect("evaluate");
    let interpolant = interpolate_subspace(&plan, &values, &mut scratch).expect("interpolate");
    assert!(interpolant.degree().is_none_or(|degree| degree < 16));
    for (index, value) in values.iter().enumerate() {
        assert_eq!(interpolant.evaluate(plan.point_element(index)), *value);
    }

    // A short value vector is a geometry failure carrying the plan size.
    assert_eq!(
        interpolate_subspace(&plan, &values[..7], &mut scratch).map(|_| ()),
        Err(poly_ring::PolynomialError::Config(
            ConfigError::GeometryOverflow {
                context: "subspace interpolation values",
            }
        ))
    );
}

/// The default transform scratch evaluates exactly like a fresh one.
#[cfg(feature = "fft")]
#[test]
fn default_transform_scratch_evaluates() {
    use butterfly_fft::transform::TransformPlan;
    use poly_ring::{TransformScratch, evaluate_subspace_into};

    let plan = TransformPlan::<Gf8B>::new(8).expect("plan");
    let mut scratch = TransformScratch::default();
    let polynomial = noise_poly::<Gf8B>(8, 0xE0B0);
    let mut values = Vec::new();
    evaluate_subspace_into(&mut values, &polynomial, &plan, &mut scratch).expect("evaluate");
    for (value, index) in values.iter().zip(0..8) {
        assert_eq!(*value, polynomial.evaluate(plan.point_element(index)));
    }
}

/// The default domain scratch evaluates exactly like a fresh one.
#[test]
fn default_domain_scratch_evaluates() {
    let mut scratch = DomainScratch::<Gf8B>::default();
    let polynomial = noise_poly::<Gf8B>(9, 0xE0C0);
    let points = distinct_points::<Gf8B>(5, 0xE0C1);
    let domain = EvaluationDomain::arbitrary(points.clone()).expect("domain");
    let values = domain
        .evaluate(&polynomial, &mut scratch)
        .expect("evaluate");
    for (point, value) in points.iter().zip(&values) {
        assert_eq!(*value, polynomial.evaluate(*point));
    }
}
