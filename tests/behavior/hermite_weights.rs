//! Hermite weights behavior and error boundaries.

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16, Goldilocks, Mersenne31, QuadMersenne31, gf8b, mersenne31};
use poly_ring::{HermitePlan, Polynomial};

use crate::oracles;
use oracles::noise;

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn taylor_jets<F: FieldKernels>(
    expected: &Polynomial<F>,
    points: &[F::Elem],
    weights: &[usize],
) -> Vec<F::Elem> {
    let mut jets = Vec::new();
    for (point, &weight) in points.iter().zip(weights) {
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
    jets
}

fn check_round_trip<F: poly_ring::PolynomialField>(seed: u64, npoints: usize, max_weight: usize) {
    // The interpolant has degree below the total weight, so keep the noise
    // polynomial shorter than the weight sum.
    let weights: Vec<usize> = (0..npoints)
        .map(|i| 1 + (seed as usize + i) % max_weight)
        .collect();
    let total: usize = weights.iter().sum();
    let degree_total: usize = total.saturating_sub(1).max(2);
    let expected = Polynomial::<F>::from_coefficients(&noise::<F>(degree_total, seed)).expect("f");
    // Distinct deterministic points from the generator powers.
    let mut points = Vec::new();
    let mut candidate = F::Elem::ONE;
    while points.len() < npoints {
        candidate = candidate.mul(F::GENERATOR);
        if !points.contains(&candidate) {
            points.push(candidate);
        }
    }
    let plan = HermitePlan::<F>::new(&points, &weights).expect("plan");
    let jets = taylor_jets(&expected, &points, &weights);
    let reconstructed = plan.interpolate(&jets).expect("interpolate");
    assert_eq!(reconstructed, expected);
}

/// Single-entry interpolation exercises the loop body once with its scratch
/// build, translation, local assembly, and CRT blend.
#[test]
fn single_entry_interpolation_blends_one_weight() {
    let expected =
        Polynomial::<Mersenne31>::from_coefficients(&[m31(4), m31(2), m31(7)]).expect("f");
    let points = [m31(3)];
    let plan = HermitePlan::<Mersenne31>::new(&points, &[3]).expect("plan");
    let jets = taylor_jets(&expected, &points, &[3]);
    assert_eq!(plan.interpolate(&jets).expect("interpolate"), expected);
}

/// Many entries with mixed weights (including a zero in the middle)
/// exercise the slot/offset lockstep across the whole loop.
#[test]
fn mixed_weights_with_interior_zero_reconstruct() {
    // f = 1 + 2X + X^3 + 4X^5 over Gf16.
    let one = <Gf16 as Field>::Elem::ONE;
    let two = one.add(one);
    let four = two.add(two);
    let expected = Polynomial::<Gf16>::from_coefficients(&[
        one,
        two,
        <Gf16 as Field>::Elem::ZERO,
        one,
        <Gf16 as Field>::Elem::ZERO,
        four,
    ])
    .expect("f");
    let g = <Gf16 as Field>::GENERATOR;
    let points = [g, g.pow(3), g.pow(5), g.pow(7), g.pow(9)];
    let weights = [3_usize, 0, 2, 4, 1];
    let plan = HermitePlan::<Gf16>::new(&points, &weights).expect("plan");
    assert_eq!(plan.offsets(), &[0, 3, 3, 5, 9, 10]);
    let jets = taylor_jets(&expected, &points, &weights);
    // Interior zero weight contributes no jet entries.
    assert_eq!(jets.len(), 10);
    assert_eq!(plan.interpolate(&jets).expect("interpolate"), expected);
}

/// Binary fields above the characteristic blend CRT weights exactly.
#[test]
fn binary_fields_blend_crt_weights() {
    check_round_trip::<Gf8B>(0xF101, 4, 4);
    check_round_trip::<Gf16>(0xF102, 5, 3);
}

/// Prime fields blend CRT weights exactly, including the extension field.
#[test]
fn prime_fields_blend_crt_weights() {
    check_round_trip::<Mersenne31>(0xF103, 4, 3);
    check_round_trip::<Goldilocks>(0xF104, 3, 3);
    check_round_trip::<QuadMersenne31>(0xF105, 3, 2);
}

/// The interpolation result normalizes: a constant jet vector rebuilds the
/// constant polynomial with no trailing zeros.
#[test]
fn constant_jets_rebuild_the_constant() {
    let plan = HermitePlan::<Gf8B>::new(&[b(2), b(9)], &[2, 2]).expect("plan");
    let reconstructed = plan
        .interpolate(&[b(5), b(0), b(5), b(0)])
        .expect("interpolate");
    assert_eq!(
        reconstructed,
        Polynomial::<Gf8B>::from_coefficients(&[b(5)]).expect("constant")
    );
}
