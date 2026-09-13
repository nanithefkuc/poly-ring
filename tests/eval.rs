//! Evaluation, interpolation, and domains: three independent evaluators
//! agreeing.

#[cfg(feature = "fft")]
use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16};
use poly_ring::{
    DomainError, DomainScratch, EvaluationDomain, MultipointScratch, NewtonBasis, Polynomial,
    interpolate_lagrange, interpolate_newton,
};

mod oracles;
use oracles::{naive_evaluate, noise, noise_poly};

fn assert_multipoint<F: FieldKernels>() {
    for len in [1, 2, 5, 8, 16, 17, 31, 40] {
        let polynomial = noise_poly::<F>(len + 3, 0x3000 + len as u64);
        let points: Vec<F::Elem> = distinct_points::<F>(len, 0x3100 + len as u64);

        // The subproduct-tree result equals per-point Horner everywhere.
        let values = poly_ring::evaluate_multipoint(&polynomial, &points).expect("multipoint");
        let horner = polynomial.evaluate_many(&points).expect("horner");
        assert_eq!(values, horner);
        for (point, value) in points.iter().zip(&values) {
            assert_eq!(*value, naive_evaluate(&polynomial, *point));
        }
    }
}

fn assert_interpolation<F: FieldKernels>() {
    for len in [1, 2, 3, 7, 8, 9, 20, 33] {
        let points: Vec<F::Elem> = distinct_points::<F>(len, 0x3200 + len as u64);
        let values = noise::<F>(len, 0x3300 + len as u64);

        // Newton and Lagrange agree exactly (the differential the
        // ecosystem's duplicate Lagranges never had).
        let newton = interpolate_newton::<F>(&points, &values).expect("newton");
        let lagrange = interpolate_lagrange::<F>(&points, &values).expect("lagrange");
        assert_eq!(newton, lagrange);
        // Round trip: interpolate then evaluate returns the values, and the
        // degree stayed below the point count.
        assert!(newton.degree().is_none_or(|degree| degree < len));
        for (point, value) in points.iter().zip(&values) {
            assert_eq!(naive_evaluate(&newton, *point), *value);
        }
    }

    // Duplicate points name both indices (value and limit, U9).
    let points = distinct_points::<F>(4, 0x3400);
    let mut duplicated = points.clone();
    duplicated.push(points[1]);
    let values = noise::<F>(5, 0x3401);
    assert_eq!(
        interpolate_newton::<F>(&points, &[F::Elem::ZERO; 4]).map(|_| ()),
        Ok(())
    );
    assert_eq!(
        poly_ring::eval::interpolate_lagrange::<F>(&duplicated, &values).map(|_| ()),
        Err(poly_ring::EvalError::Domain(DomainError::DuplicatePoint {
            first: 1,
            second: 4
        }))
    );
}

fn assert_newton_basis_reuse<F: FieldKernels>() {
    let points: Vec<F::Elem> = distinct_points::<F>(12, 0x3500);
    let basis = NewtonBasis::<F>::new(&points).expect("basis");
    let mut output = Polynomial::<F>::zero();
    for seed in 0..3 {
        let values = noise::<F>(points.len(), 0x3600 + seed);
        basis
            .interpolate_into(&values, &mut output)
            .expect("interpolate");
        for (point, value) in points.iter().zip(&values) {
            assert_eq!(naive_evaluate(&output, *point), *value);
        }
    }
    // The vanishing polynomial is ∏(X + α_i): zero at every support point.
    for point in &points {
        assert!(basis.vanishing().evaluate(*point).is_zero());
    }
    // Value length mismatch names expected and found.
    assert_eq!(
        basis
            .interpolate_into(&noise::<F>(3, 0x3700), &mut output)
            .map(|_| ()),
        Err(poly_ring::EvalError::Domain(DomainError::LengthMismatch {
            expected: 12,
            found: 3
        }))
    );
}

#[cfg(feature = "fft")]
fn assert_domain_paths<F: ButterflyKernels>() {
    let mut scratch = DomainScratch::<F>::new();
    let polynomial = noise_poly::<F>(20, 0x3800);

    // Arbitrary domain over distinct points.
    let points = distinct_points::<F>(24, 0x3810);
    let domain = EvaluationDomain::arbitrary(points.clone()).expect("domain");
    let values = domain
        .evaluate(&polynomial, &mut scratch)
        .expect("evaluate");
    for (point, value) in points.iter().zip(&values) {
        assert_eq!(*value, naive_evaluate(&polynomial, *point));
    }
    let interpolant = domain
        .interpolate(&values, &mut scratch)
        .expect("interpolate");
    for (point, value) in points.iter().zip(&values) {
        assert_eq!(naive_evaluate(&interpolant, *point), *value);
    }

    // Subspace and coset domains produce identical evaluation results with
    // and without the transform: cross-check against plain Horner.
    for log_size in [2_u32, 4, 5] {
        let size = 1 << log_size;
        let subspace = EvaluationDomain::additive_subspace(size).expect("subspace");
        let values = subspace
            .evaluate(&polynomial, &mut scratch)
            .expect("evaluate");
        for (point, value) in subspace.points().iter().zip(&values) {
            assert_eq!(*value, naive_evaluate(&polynomial, *point));
        }
        // Interpolation round-trips over the same points.
        let interpolant = subspace
            .interpolate(&values, &mut scratch)
            .expect("interpolate");
        for (point, value) in subspace.points().iter().zip(&values) {
            assert_eq!(naive_evaluate(&interpolant, *point), *value);
        }

        let shift = noise::<F>(1, 0x3900 + log_size as u64)[0];
        let coset = EvaluationDomain::affine_coset(size, shift).expect("coset");
        let values = coset.evaluate(&polynomial, &mut scratch).expect("evaluate");
        for (point, value) in coset.points().iter().zip(&values) {
            assert_eq!(*value, naive_evaluate(&polynomial, *point));
        }
    }

    // A non-power-of-two size names the size and the limit.
    assert_eq!(
        EvaluationDomain::<F>::additive_subspace(24).map(|_| ()),
        Err(DomainError::NotSubspace {
            size: 24,
            limit: F::ORDER.min(usize::MAX as u128) as usize
        })
    );
    // A duplicate in an arbitrary support is rejected.
    let mut duplicated = points.clone();
    duplicated.push(points[0]);
    assert!(matches!(
        EvaluationDomain::<F>::arbitrary(duplicated),
        Err(DomainError::DuplicatePoint { .. })
    ));
    // Value length mismatch on interpolation.
    assert_eq!(
        domain
            .interpolate(&noise::<F>(2, 0x3950), &mut scratch)
            .map(|_| ()),
        Err(poly_ring::EvalError::Domain(DomainError::LengthMismatch {
            expected: domain.len(),
            found: 2
        }))
    );
}

fn distinct_points<F: FieldKernels>(len: usize, seed: u64) -> Vec<F::Elem> {
    let mut points = Vec::new();
    let mut state = seed;
    while points.len() < len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bytes = state.to_le_bytes();
        let candidate = F::read(&bytes[..F::BYTES]);
        if !points.contains(&candidate) && (len as u128) <= F::ORDER {
            points.push(candidate);
        }
    }
    points
}

#[test]
fn multipoint_matches_horner_across_fields() {
    assert_multipoint::<Gf8B>();
    assert_multipoint::<Gf8B>();
    assert_multipoint::<Gf16>();
}

#[test]
fn interpolation_routes_agree_across_fields() {
    assert_interpolation::<Gf8B>();
    assert_interpolation::<Gf8B>();
    assert_interpolation::<Gf16>();
}

#[test]
fn newton_basis_reuse_across_fields() {
    assert_newton_basis_reuse::<Gf8B>();
    assert_newton_basis_reuse::<Gf16>();
}

#[cfg(feature = "fft")]
#[test]
fn domain_paths_agree_across_fields() {
    assert_domain_paths::<Gf8B>();
    assert_domain_paths::<Gf8B>();
    assert_domain_paths::<Gf16>();
}

#[cfg(not(feature = "fft"))]
#[test]
fn domain_paths_agree_across_fields() {
    // Without the transform the same domain objects evaluate through the
    // arbitrary paths; the results are identical by construction.
    assert_domain_paths_nofft::<Gf8B>();
    assert_domain_paths_nofft::<Gf16>();
}

#[cfg(not(feature = "fft"))]
fn assert_domain_paths_nofft<F: FieldKernels>() {
    let mut scratch = DomainScratch::<F>::new();
    let polynomial = noise_poly::<F>(20, 0x3800);
    let points: Vec<F::Elem> = distinct_points::<F>(24, 0x3810);
    let domain = EvaluationDomain::arbitrary(points.clone()).expect("domain");
    let values = domain
        .evaluate(&polynomial, &mut scratch)
        .expect("evaluate");
    for (point, value) in points.iter().zip(&values) {
        assert_eq!(*value, naive_evaluate(&polynomial, *point));
    }
    let subspace = EvaluationDomain::additive_subspace(16).expect("subspace");
    let values = subspace
        .evaluate(&polynomial, &mut scratch)
        .expect("evaluate");
    for (point, value) in subspace.points().iter().zip(&values) {
        assert_eq!(*value, naive_evaluate(&polynomial, *point));
    }
}

#[cfg(feature = "fft")]
#[test]
fn subspace_transform_matches_plain_evaluation() {
    use butterfly_fft::core::transform::TransformPlan;
    use poly_ring::{TransformScratch, evaluate_subspace};

    for log_size in [2_u32, 4, 6] {
        let size = 1 << log_size;
        let plan = TransformPlan::<Gf16>::new(size).expect("plan");
        let mut scratch = TransformScratch::new();
        let polynomial = noise_poly::<Gf16>(size, 0x3A00 + log_size as u64);
        let values = evaluate_subspace(&polynomial, &plan, &mut scratch).expect("evaluate");
        for (value, index) in values.iter().zip(0..size) {
            assert_eq!(
                *value,
                naive_evaluate(&polynomial, plan.point_element(index))
            );
        }
        // The degree-overflow path reduces modulo the vanishing polynomial.
        let long = noise_poly::<Gf16>(size + 5, 0x3B00 + log_size as u64);
        let values = evaluate_subspace(&long, &plan, &mut scratch).expect("evaluate");
        for (value, index) in values.iter().zip(0..size) {
            assert_eq!(*value, naive_evaluate(&long, plan.point_element(index)));
        }
    }
}

#[test]
fn multipoint_scratch_reuse_is_exact() {
    let mut scratch = MultipointScratch::<Gf8B>::new();
    let mut values = Vec::new();
    let polynomial = noise_poly::<Gf8B>(30, 0x3C00);
    let points = distinct_points::<Gf8B>(40, 0x3C10);
    poly_ring::evaluate_multipoint_into(&polynomial, &points, &mut scratch, &mut values)
        .expect("warm-up");
    let warmed = values.clone();
    let other = noise_poly::<Gf8B>(25, 0x3C20);
    poly_ring::evaluate_multipoint_into(&other, &points, &mut scratch, &mut values)
        .expect("reuse");
    assert_eq!(values, other.evaluate_many(&points).expect("horner"));
    assert_ne!(values, warmed);
}
