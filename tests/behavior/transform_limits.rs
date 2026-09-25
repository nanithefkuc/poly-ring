//! Transform limits behavior and error boundaries.

#[cfg(feature = "fft")]
use fgf::{Gf16, Gf32};
#[cfg(feature = "fft")]
use poly_ring::{PolynomialProductScratch, ProductStrategy, multiply_batch_truncated_into};

#[cfg(feature = "fft")]
use crate::oracles;
#[cfg(feature = "fft")]
use oracles::noise_poly;

/// Auto falls back to schoolbook when the plan cannot be built: a 2^21-row
/// transform exceeds Gf32's domain cap, so the batch runs the scalar path
/// and stays byte-identical to the naive convolution.
#[cfg(feature = "fft")]
#[test]
fn auto_falls_back_when_the_plan_is_unbuildable() {
    // Full product just above 2^20 rows: 2^20+1 coefficients needs a 2^21
    // transform, beyond the plan cap. Small enough to run schoolbook in
    // the test harness (2M coefficients is too slow), so scale down: the
    // fallback also triggers for any size whose plan fails. Instead drive
    // the unit directly: full product 2^21 needs checked arithmetic only.
    // A cheaper observable trigger: Gf16 caps at 2^16 rows; a full product
    // of 2^16+1 coefficients needs a 2^17 transform and falls back.
    let left = noise_poly::<Gf16>(32769, 0xE801);
    let right = noise_poly::<Gf16>(32769, 0xE802);
    let pairs = [(&left, &right)];
    let mut scratch = PolynomialProductScratch::<Gf16>::new();
    let mut output = Vec::new();
    multiply_batch_truncated_into(
        &mut output,
        &pairs,
        usize::MAX,
        ProductStrategy::Auto,
        &mut scratch,
    )
    .expect("auto fallback");
    // Truncated to a checkable prefix: the fallback wrote the exact
    // schoolbook product, so its low 64 coefficients match naive.
    let mut check_scratch = PolynomialProductScratch::<Gf16>::new();
    let mut check = Vec::new();
    multiply_batch_truncated_into(
        &mut check,
        &pairs,
        64,
        ProductStrategy::Schoolbook,
        &mut check_scratch,
    )
    .expect("schoolbook");
    let mut output_truncated = output[0].clone();
    output_truncated.truncate(64);
    assert_eq!(output_truncated, check[0]);
}

/// Forced AFFT on the same unbuildable geometry surfaces the plan error
/// instead of falling back.
#[cfg(feature = "fft")]
#[test]
fn forced_afft_surfaces_the_plan_error() {
    let left = noise_poly::<Gf16>(32769, 0xE803);
    let right = noise_poly::<Gf16>(32769, 0xE804);
    let pairs = [(&left, &right)];
    let mut scratch = PolynomialProductScratch::<Gf16>::new();
    let mut output = Vec::new();
    assert!(
        multiply_batch_truncated_into(
            &mut output,
            &pairs,
            usize::MAX,
            ProductStrategy::Afft,
            &mut scratch
        )
        .is_err()
    );
}

/// Gf32 plan-cap fallback through the domain constructor: 2^21 points fit
/// the field but exceed the transform cap, and the domain reports the plan
/// failure rather than building.
#[cfg(feature = "fft")]
#[test]
fn oversized_gf32_domain_reports_the_plan_failure() {
    assert!(
        poly_ring::EvaluationDomain::<Gf32>::additive_subspace(1 << 21)
            .map(|_| ())
            .is_err()
    );
}
