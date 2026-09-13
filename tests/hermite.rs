//! Hermite reconstruction against independently computed Taylor data.
//!
//! Every fixture's expected jets are expanded by hand (or through the
//! independent `MultiplicityPlan` evaluator), never by round-tripping the
//! plan under test alone.

use fgf::field::Elem;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::{HermiteError, HermitePlan, Polynomial, interpolate_lagrange};

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

fn gf8b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

/// M31 points `[0,1]`, multiplicities `[2,2]`, jets `[1,2,10,20]`
/// reconstruct `1 + 2X + 3X² + 4X³`. The jets are also produced
/// independently by weighted evaluation of the expanded polynomial.
#[test]
fn m31_weighted_fixture_reconstructs_the_expanded_polynomial() {
    let expected =
        Polynomial::<Mersenne31>::from_coefficients(&[m31(1), m31(2), m31(3), m31(4)]).expect("f");
    let points = [m31(0), m31(1)];

    // Independent expansion: the defining Taylor sums at both points.
    // At a: D^[j]f(a) = Σ_k C(k,j)·a_k·a^{k−j} by direct summation.
    let mut jets = Vec::new();
    for point in points {
        for order in 0..2 {
            let mut value = m31(0);
            for (degree, coefficient) in expected.coefficients().enumerate() {
                if degree >= order {
                    value = value.add(
                        coefficient
                            .mul(poly_ring::binomial::<Mersenne31>(degree, order))
                            .mul(point.pow((degree - order) as u64)),
                    );
                }
            }
            jets.push(value);
        }
    }
    assert_eq!(jets, vec![m31(1), m31(2), m31(10), m31(20)]);

    // The prepared evaluator agrees with the hand expansion.
    let plan = HermitePlan::<Mersenne31>::new(&points, &[2, 2]).expect("plan");
    let reconstructed = plan.interpolate(&jets).expect("interpolate");
    assert_eq!(reconstructed, expected);
    assert!(reconstructed.degree().unwrap() < plan.total_weight());
}

/// Gf8B points `[0,1]`, weights `[3,3]`, jets `[1,0,1,0,0,1]`
/// reconstruct `1 + X²` — multiplicity above the characteristic.
#[test]
fn gf8b_fixture_reconstructs_above_the_characteristic() {
    let expected = Polynomial::<Gf8B>::from_coefficients(&[gf8b(1), gf8b(0), gf8b(1)]).expect("f");
    let points = [gf8b(0), gf8b(1)];
    let plan = HermitePlan::<Gf8B>::new(&points, &[3, 3]).expect("plan");

    // Independent Taylor check: (1 + (a+T)²) truncated per point.
    // a = 0: 1 + T²  → jets [1, 0, 1].  a = 1: T² → jets [0, 0, 1].
    let jets = [gf8b(1), gf8b(0), gf8b(1), gf8b(0), gf8b(0), gf8b(1)];
    let reconstructed = plan.interpolate(&jets).expect("interpolate");
    assert_eq!(reconstructed, expected);
}

/// Empty and all-zero requests reconstruct zero; conflicting duplicates
/// are rejected at construction while a zero-weight duplicate is legal;
/// a mismatched value length is rejected.
#[test]
fn degenerate_requests_hold_their_contracts() {
    // Empty request.
    let empty = HermitePlan::<Mersenne31>::new(&[], &[]).expect("empty plan");
    assert_eq!(empty.total_weight(), 0);
    assert!(empty.interpolate(&[]).expect("zero").is_zero());

    // All-zero weights.
    let plan = HermitePlan::<Mersenne31>::new(&[m31(3), m31(4)], &[0, 0]).expect("all-zero plan");
    assert_eq!(plan.total_weight(), 0);
    assert_eq!(plan.offsets(), &[0, 0, 0]);
    assert!(plan.interpolate(&[]).expect("zero").is_zero());

    // A positive-weight duplicate is a construction error naming both
    // positions.
    let duplicate = HermitePlan::<Mersenne31>::new(&[m31(3), m31(3)], &[1, 2]);
    assert!(matches!(
        duplicate,
        Err(HermiteError::DuplicatePoint {
            first: 0,
            second: 1
        })
    ));

    // The same point with one zero weight is accepted and ignored.
    let legal =
        HermitePlan::<Mersenne31>::new(&[m31(3), m31(3)], &[2, 0]).expect("zero-weight duplicate");
    assert_eq!(legal.total_weight(), 2);
    assert_eq!(legal.offsets(), &[0, 2, 2]);

    // Noncanonical prime lanes compare by field value: 31+3 and 3 are the
    // same point, so both positive is still a conflict.
    let noncanonical = HermitePlan::<Mersenne31>::new(
        &[
            mersenne31::Elem::from_raw(3),
            mersenne31::Elem::from_raw((<Mersenne31 as fgf::field::Field>::ORDER as u32) + 3),
        ],
        &[1, 1],
    );
    assert!(matches!(
        noncanonical,
        Err(HermiteError::DuplicatePoint { .. })
    ));

    // Length mismatch on construction and on interpolation.
    assert!(matches!(
        HermitePlan::<Mersenne31>::new(&[m31(1)], &[]),
        Err(HermiteError::LengthMismatch {
            expected: 1,
            actual: 0
        })
    ));
    let plan = HermitePlan::<Mersenne31>::new(&[m31(1)], &[3]).expect("plan");
    assert!(matches!(
        plan.interpolate(&[m31(1), m31(2)]),
        Err(HermiteError::LengthMismatch { .. })
    ));
}

/// Simple points (all weights one) agree with Lagrange interpolation, and
/// a weighted reconstruction agrees with independently expanded Taylor
/// data at every point, with degree strictly below the positive weight.
#[test]
fn simple_and_weighted_reconstruction_agree_with_the_definition() {
    fn check<F: poly_ring::PolynomialField>() {
        // Distinct points 1..9 embedded through the generator.
        let points: Vec<F::Elem> = (1..9).map(|index| F::GENERATOR.pow(index as u64)).collect();
        // The deterministic polynomial 1 + 2X + 3X².
        let expected: Polynomial<F> = Polynomial::from_coefficients(&[
            F::Elem::ONE,
            F::Elem::ONE.add(F::Elem::ONE),
            F::Elem::ONE.add(F::Elem::ONE).add(F::Elem::ONE),
        ])
        .expect("f");

        // Weights [1,2,3,1,2,3,1,2]: total 15.
        let weights = [1usize, 2, 3, 1, 2, 3, 1, 2];
        let total: usize = weights.iter().sum();
        let plan = HermitePlan::<F>::new(&points, &weights).expect("plan");

        let mut jets = Vec::with_capacity(total);
        for (point, &weight) in points.iter().zip(&weights) {
            for order in 0..weight {
                let mut value = F::Elem::ZERO;
                for (degree, coefficient) in expected.coefficients().enumerate() {
                    if degree >= order {
                        value = value.add(
                            coefficient
                                .mul(poly_ring::binomial::<F>(degree, order))
                                .mul(point.pow((degree - order) as u64)),
                        );
                    }
                }
                jets.push(value);
            }
        }

        let reconstructed = plan.interpolate(&jets).expect("interpolate");
        assert_eq!(reconstructed, expected);
        assert!(reconstructed.degree().unwrap() < total);

        // All weights one: the reconstruction must agree with Lagrange on
        // the same points.
        let values: Vec<F::Elem> = points.iter().map(|p| expected.evaluate(*p)).collect();
        let lagrange = interpolate_lagrange::<F>(&points, &values).expect("lagrange");
        let simple = HermitePlan::<F>::new(&points, &[1; 8]).expect("simple plan");
        assert_eq!(simple.interpolate(&values).expect("interpolate"), lagrange);
    }
    check::<Gf8B>();
    check::<Mersenne31>();
}
