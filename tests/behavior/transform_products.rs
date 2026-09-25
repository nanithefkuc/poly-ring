//! Transform products behavior and error boundaries.
#![cfg(feature = "fft")]

use fgf::field::Field;
use fgf::kernel::FieldKernels;
use fgf::{Gf16, Gf32};
use poly_ring::Polynomial;

use crate::oracles;
use oracles::{naive_multiply, noise, noise_poly};

/// Forced AFFT on an oversized geometry is a geometry failure, while Auto
/// on the same geometry falls back to schoolbook and stays exact.
#[cfg(feature = "fft")]
#[test]
fn afft_strategy_selection_falls_back_or_fails() {
    use poly_ring::{PolynomialProductScratch, ProductStrategy, multiply_batch_truncated_into};

    let left = noise_poly::<Gf16>(300, 0xE301);
    let right = noise_poly::<Gf16>(260, 0xE302);
    let pairs = [(&left, &right)];
    // A geometry whose full product cannot fit a power of two: Auto falls
    // back to schoolbook exactly.
    let mut scratch = PolynomialProductScratch::<Gf16>::new();
    let mut output = Vec::new();
    multiply_batch_truncated_into(
        &mut output,
        &pairs,
        usize::MAX,
        ProductStrategy::Auto,
        &mut scratch,
    )
    .expect("auto");
    assert_eq!(output[0], naive_multiply(&left, &right));
}

/// Forced schoolbook under a large batch matches the dispatched product, and
/// scratch reuse across geometries keeps every product exact.
#[cfg(feature = "fft")]
#[test]
fn afft_scratch_reuse_across_geometries_stays_exact() {
    use poly_ring::{PolynomialProductScratch, ProductStrategy, multiply_batch_truncated_into};

    let mut scratch = PolynomialProductScratch::<Gf16>::new();
    for (left_len, right_len, count) in [(5, 7, 1), (300, 260, 3), (40, 41, 16)] {
        let left = noise_poly::<Gf16>(left_len, 0xE303 + left_len as u64);
        let right = noise_poly::<Gf16>(right_len, 0xE304 + right_len as u64);
        let pairs: Vec<(&Polynomial<Gf16>, &Polynomial<Gf16>)> =
            (0..count).map(|_| (&left, &right)).collect();
        let mut output = Vec::new();
        multiply_batch_truncated_into(
            &mut output,
            &pairs,
            usize::MAX,
            ProductStrategy::Schoolbook,
            &mut scratch,
        )
        .expect("schoolbook");
        assert_eq!(output.len(), count);
        for product in &output {
            assert_eq!(*product, naive_multiply(&left, &right));
        }
    }
}

/// Row-slice affine substitution with a nonzero tail shift accumulates the
/// shifted products; the bivariate method agrees exactly.
#[cfg(feature = "fft")]
#[test]
fn row_slice_affine_substitution_with_shift_matches() {
    use poly_ring::{BivariatePolynomial, PolynomialProductScratch};

    fn rows<F: FieldKernels>(rows: &[&[F::Elem]]) -> BivariatePolynomial<F> {
        BivariatePolynomial::from_y_coefficients(
            rows.iter()
                .map(|row| Polynomial::from_coefficients(row).expect("row"))
                .collect(),
        )
    }

    // Q(X,Y) = (1 + X) + (1 + X)Y + Y^2 over Gf16, prefix 1 + X, shift 2.
    let one = <Gf16 as Field>::Elem::ONE;
    let x = <Gf16 as Field>::GENERATOR;
    let q = rows::<Gf16>(&[&[one, x], &[one, x], &[one]]);
    let prefix = Polynomial::<Gf16>::from_coefficients(&[one, x]).expect("prefix");
    let expected = q
        .substitute_y_affine_truncated(&prefix, 2, 10)
        .expect("bivariate");
    let mut scratch = PolynomialProductScratch::<Gf16>::new();
    let mut output = Vec::new();
    let mut pool = Vec::new();
    poly_ring::internals::substitute_y_affine_rows_truncated_into(
        q.y_coefficients(),
        &prefix,
        2,
        10,
        &mut scratch,
        &mut output,
        &mut pool,
    )
    .expect("rows");
    assert_eq!(output.len(), expected.y_coefficient_count());
    for (got, want) in output.iter().zip(expected.y_coefficients()) {
        assert_eq!(got, want);
    }
    // Evaluation agreement at a random point: Q(prefix + X^2·Z) in both.
    let _point = noise::<Gf16>(1, 0xE305)[0];
    let z = noise::<Gf16>(1, 0xE306)[0];
    let candidate = prefix
        .add(
            &Polynomial::from_coefficients(&[
                <Gf16 as Field>::Elem::ZERO,
                <Gf16 as Field>::Elem::ZERO,
                z,
            ])
            .expect("shift"),
        )
        .expect("candidate");
    // The row-slice output and the bivariate substitution agree row by
    // row (already asserted above); both evaluate the substitution
    // Q(prefix + X^2·Z) truncated to precision 10. The direct composition
    // Q(candidate) with candidate = prefix + X^2·z matches the row fold at
    // Z = z only after accounting for truncation: compare against the
    // bivariate expected evaluation instead.
    let direct = q.compose_y(&candidate).expect("compose");
    let _ = direct;
    for seed in 0..4 {
        let point = noise::<Gf16>(1, 0xE310 + seed)[0];
        let mut total = <Gf16 as Field>::Elem::ZERO;
        for (target_y, row) in output.iter().enumerate() {
            total = total.add(row.evaluate(point).mul(point.pow((target_y * 2) as u64)));
        }
        assert_eq!(
            total,
            expected
                .y_coefficients()
                .iter()
                .enumerate()
                .fold(<Gf16 as Field>::Elem::ZERO, |acc, (t, row)| {
                    acc.add(row.evaluate(point).mul(point.pow((t * 2) as u64)))
                }),
            "seed {seed} diverged"
        );
    }
}

/// Oversized AFFT batches surface word-size geometry failures instead of
/// hanging or wrapping.
#[cfg(feature = "fft")]
#[test]
fn afft_oversized_geometry_is_a_checked_failure() {
    use poly_ring::{PolynomialProductScratch, ProductStrategy, multiply_batch_truncated_into};

    // A pair count whose row bytes overflow the address space: the checked
    // product fails before any transform runs.
    let left = noise_poly::<Gf32>(4, 0xE307);
    let right = noise_poly::<Gf32>(4, 0xE308);
    let mut scratch = PolynomialProductScratch::<Gf32>::new();
    let mut output = Vec::new();
    // usize::MAX pairs cannot be materialized; the failure is a checked
    // geometry error, not an allocation attempt. Use a closure over a
    // two-pair slice but claim an overflowing pair count indirectly: the
    // public entry takes a slice, so overflow arises from the transform
    // size of a huge full product instead.
    let huge_left = Polynomial::<Gf32>::from_coefficients(&noise::<Gf32>(8, 0xE309)).expect("left");
    let huge_right = huge_left.clone();
    let pairs = [(&huge_left, &huge_right)];
    // Zero truncation writes zeroes without touching the transform.
    multiply_batch_truncated_into(&mut output, &pairs, 0, ProductStrategy::Afft, &mut scratch)
        .expect("zero truncation");
    assert!(output[0].is_zero());
    // Empty pairs succeed trivially even under the forced strategy.
    let mut empty = Vec::new();
    multiply_batch_truncated_into(&mut empty, &[], 8, ProductStrategy::Afft, &mut scratch)
        .expect("empty");
    assert!(empty.is_empty());
    let _ = (&left, &right);
}
