//! Hermite substitution behavior and error boundaries.

#[cfg(feature = "fft")]
use fgf::Gf16;
use fgf::field::Field;
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::{BivariatePolynomial, HermitePlan, Polynomial};

use crate::oracles;
#[cfg(feature = "fft")]
use oracles::{noise, noise_poly};

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn rows<F: FieldKernels>(rows: &[&[F::Elem]]) -> BivariatePolynomial<F> {
    BivariatePolynomial::from_y_coefficients(
        rows.iter()
            .map(|row| Polynomial::from_coefficients(row).expect("row"))
            .collect(),
    )
}

/// Multi-entry Hermite interpolation blends every entry's CRT weight: three
/// points with mixed weights reconstruct exactly.
#[test]
fn three_entry_hermite_blends_every_weight() {
    // f = 2 + 5X + X^2 + 3X^4 over M31, points [1, 4, 6], weights [1, 3, 2].
    let expected =
        Polynomial::<Mersenne31>::from_coefficients(&[m31(2), m31(5), m31(1), m31(0), m31(3)])
            .expect("f");
    let points = [m31(1), m31(4), m31(6)];
    let weights = [1_usize, 3, 2];
    let plan = HermitePlan::<Mersenne31>::new(&points, &weights).expect("plan");
    assert_eq!(plan.offsets(), &[0, 1, 4, 6]);
    assert_eq!(plan.total_weight(), 6);
    // Independent jets by the defining Taylor sums.
    let mut jets = Vec::new();
    for (point, &weight) in points.iter().zip(&weights) {
        for order in 0..weight {
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
    let reconstructed = plan.interpolate(&jets).expect("interpolate");
    assert_eq!(reconstructed, expected);
    // The reconstruction's own jets are the input jets.
    for (point, &weight) in points.iter().zip(&weights) {
        for order in 0..weight {
            assert_eq!(
                reconstructed.evaluate_hasse(*point, order),
                jets[plan.offsets()[points.iter().position(|p| p == point).expect("point")]
                    + order]
            );
        }
    }
}

/// Hermite interpolation with a leading zero-weight entry skips the entry
/// in lockstep with the prepared tables.
#[test]
fn leading_zero_weight_entry_is_skipped_in_lockstep() {
    // f = 1 + 2X^2 over Gf8B: jets at 0 are [1,0,1], unused point first.
    let expected = Polynomial::<Gf8B>::from_coefficients(&[b(1), b(0), b(1)]).expect("f");
    let plan = HermitePlan::<Gf8B>::new(&[b(5), b(0)], &[0, 3]).expect("plan");
    assert_eq!(plan.offsets(), &[0, 0, 3]);
    let jets = [b(1), b(0), b(1)];
    let reconstructed = plan.interpolate(&jets).expect("interpolate");
    assert_eq!(reconstructed, expected);
}

/// Binary-field Hermite with weights above the characteristic blends the
/// CRT weights exactly.
#[test]
fn binary_hermite_above_the_characteristic_blends() {
    // f = 1 + X + X^3 over Gf8B, points [0, 1], weights [2, 4].
    let expected = Polynomial::<Gf8B>::from_coefficients(&[b(1), b(1), b(0), b(1)]).expect("f");
    let points = [b(0), b(1)];
    let weights = [2_usize, 4];
    let plan = HermitePlan::<Gf8B>::new(&points, &weights).expect("plan");
    // Independent jets: Horner-jet oracle, no binomials.
    fn horner_jet(
        coefficients: &[gf8b::Elem],
        point: gf8b::Elem,
        multiplicity: usize,
    ) -> Vec<gf8b::Elem> {
        let mut jet = vec![<Gf8B as Field>::Elem::ZERO; multiplicity];
        for &coefficient in coefficients.iter().rev() {
            for order in (1..multiplicity).rev() {
                jet[order] = point.mul(jet[order]).add(jet[order - 1]);
            }
            jet[0] = point.mul(jet[0]).add(coefficient);
        }
        jet
    }
    let coefficients: Vec<gf8b::Elem> = expected.coefficients().collect();
    let mut jets = Vec::new();
    for (point, &weight) in points.iter().zip(&weights) {
        jets.extend(horner_jet(&coefficients, *point, weight));
    }
    let reconstructed = plan.interpolate(&jets).expect("interpolate");
    assert_eq!(reconstructed, expected);
}

/// Fast affine substitution agrees with the scalar path on larger,
/// higher-degree inputs with nonzero shifts and truncation.
#[cfg(feature = "fft")]
#[test]
fn fast_affine_substitution_agrees_on_large_inputs() {
    use poly_ring::PolynomialProductScratch;

    fn check<F>()
    where
        F: FieldKernels + butterfly_fft::kernel::ButterflyKernels,
    {
        let prefix = Polynomial::<F>::from_coefficients(&noise::<F>(9, 0xE401)).expect("prefix");
        let q = BivariatePolynomial::from_y_coefficients(
            (0..6)
                .map(|j| noise_poly::<F>(7 + j * 3, 0xE402 + j as u64))
                .collect(),
        );
        let mut scratch = PolynomialProductScratch::<F>::new();
        for (tail_degree, coefficient_count) in [(0_usize, 24), (3, 24), (5, 13)] {
            let scalar = q
                .substitute_y_affine_truncated(&prefix, tail_degree, coefficient_count)
                .expect("scalar");
            let batched = q
                .substitute_y_affine_truncated_fast(
                    &prefix,
                    tail_degree,
                    coefficient_count,
                    &mut scratch,
                )
                .expect("batched");
            assert_eq!(
                scalar, batched,
                "tail {tail_degree} prec {coefficient_count}"
            );
        }
    }
    check::<Gf8B>();
    check::<Gf16>();
}

/// Fast affine substitution on zero rows and zero precision returns zero
/// without touching the product engine.
#[cfg(feature = "fft")]
#[test]
fn fast_affine_substitution_degenerate_inputs_are_zero() {
    use poly_ring::PolynomialProductScratch;

    let mut scratch = PolynomialProductScratch::<Gf16>::new();
    let prefix = noise_poly::<Gf16>(3, 0xE404);
    assert!(
        BivariatePolynomial::<Gf16>::zero()
            .substitute_y_affine_truncated_fast(&prefix, 1, 8, &mut scratch)
            .expect("zero")
            .is_zero()
    );
    let q = rows::<Gf16>(&[&[<Gf16 as Field>::Elem::ONE]]);
    assert!(
        q.substitute_y_affine_truncated_fast(&prefix, 1, 0, &mut scratch)
            .expect("zero precision")
            .is_zero()
    );
}
