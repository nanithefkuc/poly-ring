//! Transform composition branches: coset evaluation, long-polynomial
//! reduction, and interpolation error paths.
//!
//! The agreement suites evaluate short polynomials on subspaces; every test
//! here takes a branch they never reach — coset domains, inputs longer than
//! the plan (vanishing reduction), `into` reuse, and interpolation length
//! validation — each against independent Horner evaluation.

#[cfg(feature = "fft")]
use butterfly_fft::transform::TransformPlan;
#[cfg(feature = "fft")]
use fgf::field::Field;
#[cfg(feature = "fft")]
use fgf::{Gf8B, Gf16, gf8b};
#[cfg(feature = "fft")]
use poly_ring::{Polynomial, TransformScratch, evaluate_coset_into};

#[cfg(feature = "fft")]
use crate::oracles;
#[cfg(feature = "fft")]
use oracles::noise_poly;

#[cfg(feature = "fft")]
fn check_subspace_long_reduction() {
    let plan = TransformPlan::<Gf16>::new(16).expect("plan");
    let mut scratch = TransformScratch::new();
    // Longer than the plan: the vanishing reduction runs first, and the
    // values still match Horner at every plan point.
    let polynomial = noise_poly::<Gf16>(40, 0xF101);
    let values = poly_ring::evaluate_subspace(&polynomial, &plan, &mut scratch).expect("evaluate");
    assert_eq!(values.len(), 16);
    for (index, value) in values.iter().enumerate() {
        assert_eq!(*value, polynomial.evaluate(plan.point_element(index)));
    }
    // The `into` form reuses the scratch and the output vector.
    let mut reused = vec![<Gf16 as Field>::Elem::ZERO; 3];
    poly_ring::evaluate_subspace_into(&mut reused, &polynomial, &plan, &mut scratch).expect("into");
    assert_eq!(reused, values);
    let _ = TransformScratch::default();
}

/// A polynomial longer than the subspace is reduced by the vanishing
/// polynomial first; the `into` form reuses buffers and agrees.
#[cfg(feature = "fft")]
#[test]
fn long_polynomial_reduces_before_subspace_evaluation() {
    check_subspace_long_reduction();
}

/// Coset evaluation matches Horner at every shifted plan point, for short
/// and over-length inputs alike.
#[cfg(feature = "fft")]
#[test]
fn coset_evaluation_matches_horner_at_shifted_points() {
    use poly_ring::EvaluationDomain;

    let shift = <Gf16 as Field>::GENERATOR.pow(3);
    let domain = EvaluationDomain::<Gf16>::affine_coset(16, shift).expect("coset domain");
    let plan = domain.transform_plan().expect("plan");
    let coset_points = domain.points().to_vec();
    let mut scratch = TransformScratch::new();
    for (seed, len) in [(0xF102_u64, 7_usize), (0xF103, 40)] {
        let polynomial = noise_poly::<Gf16>(len, seed);
        let mut values = Vec::new();
        evaluate_coset_into(&mut values, &polynomial, plan, &mut scratch).expect("coset");
        assert_eq!(values.len(), 16);
        for (value, point) in values.iter().zip(&coset_points) {
            assert_eq!(*value, polynomial.evaluate(*point));
        }
    }
}

/// Coset domains evaluate and interpolate through the domain facade,
/// agreeing with the direct transform calls.
#[cfg(feature = "fft")]
#[test]
fn coset_domain_matches_direct_transform_calls() {
    use poly_ring::{DomainScratch, EvaluationDomain};

    let domain = EvaluationDomain::<Gf16>::affine_coset(16, <Gf16 as Field>::GENERATOR.pow(5))
        .expect("coset domain");
    assert!(domain.transform_plan().is_some());
    let polynomial = noise_poly::<Gf16>(10, 0xF104);
    let mut scratch = DomainScratch::new();
    let mut values = Vec::new();
    domain
        .evaluate_into(&polynomial, &mut scratch, &mut values)
        .expect("evaluate");
    assert_eq!(values.len(), 16);
    for (value, point) in values.iter().zip(domain.points()) {
        assert_eq!(*value, polynomial.evaluate(*point));
    }
    // Interpolation through the coset inverts evaluation below the size.
    let mut output = Polynomial::zero();
    domain
        .interpolate_into(&values, &mut scratch, &mut output)
        .expect("interpolate");
    assert!(output.degree().is_none_or(|degree| degree < 16));
    for (value, point) in values.iter().zip(domain.points()) {
        assert_eq!(output.evaluate(*point), *value);
    }
}

/// Subspace domains with an explicit basis evaluate through the transform,
/// and interpolation inverts evaluation.
#[cfg(feature = "fft")]
#[test]
fn explicit_basis_domain_evaluates_through_transform() {
    use poly_ring::{DomainScratch, EvaluationDomain};

    let basis: Vec<gf8b::Elem> = (0..3)
        .map(|bit| Gf8B::decode(&(1_u128 << bit).to_le_bytes()[..Gf8B::BYTES]))
        .collect();
    let domain = EvaluationDomain::<Gf8B>::additive_subspace_with_basis(8, &basis).expect("domain");
    assert!(domain.transform_plan().is_some());
    let polynomial = noise_poly::<Gf8B>(5, 0xF105);
    let mut scratch = DomainScratch::new();
    let mut values = Vec::new();
    domain
        .evaluate_into(&polynomial, &mut scratch, &mut values)
        .expect("evaluate");
    assert_eq!(values.len(), 8);
    for (value, point) in values.iter().zip(domain.points()) {
        assert_eq!(*value, polynomial.evaluate(*point));
    }
    // Short interpolant inverts: interpolate then re-evaluate is identity.
    let mut output = Polynomial::zero();
    domain
        .interpolate_into(&values, &mut scratch, &mut output)
        .expect("interpolate");
    let mut round_trip = Vec::new();
    domain
        .evaluate_into(&output, &mut scratch, &mut round_trip)
        .expect("re-evaluate");
    assert_eq!(round_trip, values);
}

/// Interpolation rejects a value vector that does not match the plan size,
/// on both the direct and facade paths.
#[cfg(feature = "fft")]
#[test]
fn interpolation_rejects_short_value_vectors() {
    use poly_ring::{ConfigError, DomainScratch, EvaluationDomain, PolynomialError};

    let plan = TransformPlan::<Gf16>::new(16).expect("plan");
    let mut scratch = TransformScratch::new();
    assert_eq!(
        poly_ring::interpolate_subspace(&plan, &[<Gf16 as Field>::Elem::ZERO; 7], &mut scratch)
            .map(|_| ()),
        Err(PolynomialError::Config(ConfigError::GeometryOverflow {
            context: "subspace interpolation values",
        }))
    );
    let domain = EvaluationDomain::<Gf16>::additive_subspace(16).expect("domain");
    let mut domain_scratch = DomainScratch::new();
    let mut output = Polynomial::zero();
    assert!(
        domain
            .interpolate_into(
                &[<Gf16 as Field>::Elem::ZERO; 7],
                &mut domain_scratch,
                &mut output
            )
            .is_err()
    );
}
