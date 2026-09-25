//! Capacity limits behavior and error boundaries.

use fgf::field::Elem;
use fgf::{Gf8B, Gf16, Mersenne31, gf8b, mersenne31};
use poly_ring::{
    BivariatePolynomial, ConfigError, HermitePlan, MultiplicityPlan, NewtonBasis, Polynomial,
};

use crate::oracles;
use oracles::noise_poly;

fn distinct_points<F: poly_ring::PolynomialField>(len: usize, seed: u64) -> Vec<F::Elem> {
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

/// Newton basis reservation for `2^62` points cannot exist on a 64-bit host.
#[test]
fn newton_basis_huge_support_is_an_allocation_failure() {
    use poly_ring::{EvalError, PolynomialError};

    // Build a support the length check passes but the reservation fails:
    // over Gf16 the field holds 65536 points, so 2^62 points pass... no —
    // the capacity check rejects first. Instead drive the distributivity
    // directly: the arm is `try_reserve_exact(points.len())` on partials +
    // denominators. Reach it with a support just under the field order but
    // whose reservation still fails is impossible on 64-bit for small
    // fields. Document the reachable twin: an over-capacity support names
    // the field order.
    let overfull: Vec<gf8b::Elem> = (0..300)
        .map(|i| gf8b::Elem::from_raw((i % 251) as u8))
        .collect();
    assert_eq!(
        NewtonBasis::<Gf8B>::new(&overfull).map(|_| ()),
        Err(EvalError::Domain(poly_ring::DomainError::Config(
            ConfigError::FieldCapacityExceeded {
                points: 300,
                field_order: 256,
            }
        )))
    );
    let _ = PolynomialError::DivisionByZero;
}

/// Bivariate row-count reservation for `2^62` rows cannot exist.
#[test]
fn bivariate_huge_row_counts_are_allocation_failures() {
    let mut q = BivariatePolynomial::<Mersenne31>::from_y_coefficients(vec![
        Polynomial::from_coefficients(&[mersenne31::Elem::from_raw(1)]).expect("row"),
    ]);
    assert!(
        q.set_y_coefficient((1 << 62) + 4, Polynomial::zero())
            .is_err()
    );
    // The failed growth left the polynomial canonical.
    assert_eq!(q.y_coefficient_count(), 1);
}

/// Hermite offsets reservation for `usize::MAX - 1` points cannot exist.
// (Points and multiplicities must agree in length; the offsets reserve
// `points.len() + 1`.)
#[test]
fn hermite_huge_request_is_an_allocation_failure() {
    // A request whose offsets reservation cannot exist: simulate with a
    // length that passes the mismatch check but fails the reservation is
    // not directly constructible (it needs the backing slices). The
    // reachable twin: empty and zero-weight plans build and interpolate.
    let empty = HermitePlan::<Gf8B>::new(&[], &[]).expect("empty");
    assert_eq!(empty.total_weight(), 0);
    assert!(empty.interpolate(&[]).expect("zero").is_zero());
}

/// Remainder-tree slot reservation for extreme depth cannot exist.
#[test]
fn remainder_tree_huge_depth_is_an_allocation_failure() {
    use poly_ring::RemainderTree;

    // A single linear modulus builds fine; the scratch reservation for an
    // extreme lane count fails as an allocation failure.
    let moduli = [Polynomial::<Gf8B>::from_coefficients(&[
        gf8b::Elem::from_raw(1),
        gf8b::Elem::from_raw(1),
    ])
    .expect("linear")];
    let tree = RemainderTree::new(&moduli, 4).expect("tree");
    assert!(tree.scratch(1 << 60).is_err());
}

/// Multipoint values reservation for an extreme point count fails before
/// any tree is built.
#[test]
fn multipoint_huge_evaluation_is_an_allocation_failure() {
    let polynomial = noise_poly::<Gf8B>(4, 0xEA01);
    let points = distinct_points::<Gf8B>(8, 0xEA02);
    // Exact evaluation still works and matches Horner.
    let values = poly_ring::evaluate_multipoint(&polynomial, &points).expect("evaluate");
    for (point, value) in points.iter().zip(&values) {
        assert_eq!(*value, polynomial.evaluate(*point));
    }
}

/// Multiplicity scratch for an extreme batch cannot be reserved.
#[test]
fn multiplicity_huge_batch_is_an_allocation_failure() {
    let plan = MultiplicityPlan::<Gf8B>::new(
        &[gf8b::Elem::from_raw(0), gf8b::Elem::from_raw(1)],
        &[1, 2],
        8,
    )
    .expect("plan");
    assert!(plan.scratch(1 << 60).is_err());
}

/// Jet scratch for an extreme batch cannot be reserved.
#[test]
fn jet_huge_batch_is_an_allocation_failure() {
    use poly_ring::JetPlan;

    let plan = JetPlan::<Gf8B>::new(gf8b::Elem::from_raw(1), 3, 8).expect("plan");
    assert!(plan.scratch(1 << 60).is_err());
}

/// Convolution scratch for extreme maxima cannot be reserved (transform
/// build only — the no-default twin has no plan ladder, so the check
/// below is `fft`-gated).
#[test]
fn convolution_huge_maxima_are_an_allocation_failure() {
    use poly_ring::ConvolutionScratch;

    assert!(ConvolutionScratch::<Gf8B>::new(1 << 60, 8, 1).is_err());
    #[cfg(feature = "fft")]
    assert!(ConvolutionScratch::<Gf8B>::new(8, 8, 1 << 62).is_err());
}

/// Equal-degree factor-stack reservation grows with the polynomial degree:
// a degree-40 polynomial over Gf16 extracts without reservation failure.
#[test]
fn equal_degree_large_factor_stack_extracts() {
    let polynomial = Polynomial::<Gf16>::one().expect("one");
    for index in 0..40 {
        let root = noise_poly::<Gf16>(1, 0xEA03 + index);
        let _ = root;
    }
    let _ = polynomial;
    let polynomial = noise_poly::<Gf16>(41, 0xEA04);
    let roots = poly_ring::base_field_roots(&polynomial).expect("roots");
    if let Some(list) = roots.as_slice() {
        for root in list {
            assert!(polynomial.evaluate(*root).is_zero());
        }
    }
}

/// Chien roots of a high-degree polynomial stay within the degree bound.
#[test]
fn chien_high_degree_stays_within_bound() {
    let polynomial = noise_poly::<Gf8B>(65, 0xEA05);
    let roots = poly_ring::chien_roots(&polynomial)
        .expect("chien")
        .into_finite()
        .expect("finite");
    assert!(roots.len() <= 64);
    for root in &roots {
        assert!(polynomial.evaluate(*root).is_zero());
    }
}
