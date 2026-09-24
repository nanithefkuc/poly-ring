//! Prepared-product routes: Karatsuba-only fields, NTT lanes, and errors.
//!
//! The agreement suites run the default route on binary fields; every test
//! here takes a branch they never reach — the Karatsuba-only domain, the
//! prime-field NTT and embedded routes, scratch-capacity violations, and
//! the truncated/empty geometries — each against the naive scalar oracle
//! or an exact error payload.

use fgf::field::Field;
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf8D, Gf16};
#[cfg(feature = "fft")]
use fgf::{Goldilocks, Mersenne31, QuadMersenne31};
use poly_ring::{ConvolutionScratch, Polynomial};

use crate::oracles::{naive_multiply, noise, noise_poly};

#[cfg(feature = "fft")]
#[test]
fn wide_auto_batch_keeps_each_product_independent() {
    let left: Vec<_> = (0..17).map(|i| noise_poly::<Gf16>(5, 0xA400 + i)).collect();
    let right: Vec<_> = (0..17).map(|i| noise_poly::<Gf16>(3, 0xA500 + i)).collect();
    let pairs: Vec<_> = left.iter().zip(&right).collect();
    let expected: Vec<_> = pairs.iter().map(|(a, b)| naive_multiply(a, b)).collect();
    let mut scratch = poly_ring::PolynomialProductScratch::new();
    let mut output = Vec::new();
    poly_ring::multiply_batch_truncated_into(
        &mut output,
        &pairs,
        7,
        poly_ring::ProductStrategy::Auto,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(output, expected);
}

fn rows<F: FieldKernels>(polynomial: &Polynomial<F>, count: usize, batch: usize) -> Vec<u8> {
    assert_eq!(polynomial.coefficient_count(), count);
    let packed = polynomial.as_packed();
    let mut out = vec![0_u8; count * batch * F::BYTES];
    for degree in 0..count {
        for lane in 0..batch {
            let source = degree * F::BYTES;
            let destination = (degree * batch + lane) * F::BYTES;
            out[destination..destination + F::BYTES]
                .copy_from_slice(&packed[source..source + F::BYTES]);
        }
    }
    out
}

fn lane<F: FieldKernels>(
    output: &[u8],
    out_rows: usize,
    batch: usize,
    lane: usize,
) -> Polynomial<F> {
    let mut packed = vec![0_u8; out_rows * F::BYTES];
    for degree in 0..out_rows {
        let source = (degree * batch + lane) * F::BYTES;
        packed[degree * F::BYTES..(degree + 1) * F::BYTES]
            .copy_from_slice(&output[source..source + F::BYTES]);
    }
    Polynomial::from_packed(packed).expect("lane polynomial")
}

/// The Karatsuba-only domain (`Gf8D`) multiplies exactly: the naive scalar
/// oracle agrees lane by lane.
#[test]
fn karatsuba_only_domain_matches_naive_oracle() {
    let left = noise_poly::<Gf8D>(9, 0xB101);
    let right = noise_poly::<Gf8D>(7, 0xB102);
    let expected = {
        let mut product =
            vec![<Gf8D as Field>::Elem::ZERO; left.coefficient_count() + right.coefficient_count()];
        for (i, a) in left.coefficients().enumerate() {
            for (j, b) in right.coefficients().enumerate() {
                product[i + j] = product[i + j].add(a.mul(b));
            }
        }
        Polynomial::<Gf8D>::from_coefficients(&product).expect("naive")
    };
    let batch = 3;
    let mut scratch =
        ConvolutionScratch::<Gf8D>::new(left.coefficient_count(), right.coefficient_count(), batch)
            .expect("scratch");
    let full = left.coefficient_count() + right.coefficient_count() - 1;
    let mut output = vec![0_u8; full * batch * Gf8D::BYTES];
    poly_ring::multiply_rows_into(
        &mut output,
        &rows(&left, left.coefficient_count(), batch),
        left.coefficient_count(),
        &rows(&right, right.coefficient_count(), batch),
        right.coefficient_count(),
        batch,
        full,
        &mut scratch,
    )
    .expect("multiply");
    for lane_index in 0..batch {
        assert_eq!(lane::<Gf8D>(&output, full, batch, lane_index), expected);
    }
}

/// The Goldilocks NTT route agrees with the naive oracle on a
/// transform-sized product.
#[cfg(feature = "fft")]
#[test]
fn ntt_route_matches_naive_oracle() {
    let left = noise_poly::<Goldilocks>(17, 0xB103);
    let right = noise_poly::<Goldilocks>(13, 0xB104);
    let expected = {
        let mut product = vec![
            <Goldilocks as Field>::Elem::ZERO;
            left.coefficient_count() + right.coefficient_count()
        ];
        for (i, a) in left.coefficients().enumerate() {
            for (j, b) in right.coefficients().enumerate() {
                product[i + j] = product[i + j].add(a.mul(b));
            }
        }
        Polynomial::<Goldilocks>::from_coefficients(&product).expect("naive")
    };
    let batch = 2;
    let mut scratch = ConvolutionScratch::<Goldilocks>::new(
        left.coefficient_count(),
        right.coefficient_count(),
        batch,
    )
    .expect("scratch");
    let full = left.coefficient_count() + right.coefficient_count() - 1;
    let mut output = vec![0_u8; full * batch * Goldilocks::BYTES];
    poly_ring::multiply_rows_into(
        &mut output,
        &rows(&left, left.coefficient_count(), batch),
        left.coefficient_count(),
        &rows(&right, right.coefficient_count(), batch),
        right.coefficient_count(),
        batch,
        full,
        &mut scratch,
    )
    .expect("multiply");
    for lane_index in 0..batch {
        assert_eq!(
            lane::<Goldilocks>(&output, full, batch, lane_index),
            expected
        );
    }
}

/// The embedded Mersenne31 route agrees with the naive oracle, including
/// truncation to a shorter precision.
#[cfg(feature = "fft")]
#[test]
fn embedded_route_matches_naive_oracle_truncated() {
    let left = noise_poly::<Mersenne31>(11, 0xB105);
    let right = noise_poly::<Mersenne31>(9, 0xB106);
    let precision = 7;
    let expected = {
        let mut product = vec![
            <Mersenne31 as Field>::Elem::ZERO;
            left.coefficient_count() + right.coefficient_count()
        ];
        for (i, a) in left.coefficients().enumerate() {
            for (j, b) in right.coefficients().enumerate() {
                product[i + j] = product[i + j].add(a.mul(b));
            }
        }
        let mut truncated = Polynomial::<Mersenne31>::from_coefficients(&product).expect("naive");
        truncated.truncate(precision);
        truncated
    };
    let batch = 2;
    let mut scratch = ConvolutionScratch::<Mersenne31>::new(
        left.coefficient_count(),
        right.coefficient_count(),
        batch,
    )
    .expect("scratch");
    let mut output = vec![0_u8; precision * batch * Mersenne31::BYTES];
    poly_ring::multiply_rows_into(
        &mut output,
        &rows(&left, left.coefficient_count(), batch),
        left.coefficient_count(),
        &rows(&right, right.coefficient_count(), batch),
        right.coefficient_count(),
        batch,
        precision,
        &mut scratch,
    )
    .expect("multiply");
    for lane_index in 0..batch {
        assert_eq!(
            lane::<Mersenne31>(&output, precision, batch, lane_index),
            expected
        );
    }
}

/// QuadMersenne31 exercises the second NTT instantiation independently.
#[cfg(feature = "fft")]
#[test]
fn second_ntt_instantiation_matches_naive_oracle() {
    let left = noise_poly::<QuadMersenne31>(10, 0xB107);
    let right = noise_poly::<QuadMersenne31>(8, 0xB108);
    let expected = {
        let mut product = vec![
            <QuadMersenne31 as Field>::Elem::ZERO;
            left.coefficient_count() + right.coefficient_count()
        ];
        for (i, a) in left.coefficients().enumerate() {
            for (j, b) in right.coefficients().enumerate() {
                product[i + j] = product[i + j].add(a.mul(b));
            }
        }
        Polynomial::<QuadMersenne31>::from_coefficients(&product).expect("naive")
    };
    let batch = 1;
    let mut scratch = ConvolutionScratch::<QuadMersenne31>::new(
        left.coefficient_count(),
        right.coefficient_count(),
        batch,
    )
    .expect("scratch");
    let full = left.coefficient_count() + right.coefficient_count() - 1;
    let mut output = vec![0_u8; full * batch * QuadMersenne31::BYTES];
    poly_ring::multiply_rows_into(
        &mut output,
        &rows(&left, left.coefficient_count(), batch),
        left.coefficient_count(),
        &rows(&right, right.coefficient_count(), batch),
        right.coefficient_count(),
        batch,
        full,
        &mut scratch,
    )
    .expect("multiply");
    assert_eq!(lane::<QuadMersenne31>(&output, full, batch, 0), expected);
}

/// An undersized scratch reports the capacity violation with both counts.
#[test]
fn undersized_scratch_reports_capacity_violation() {
    let left = noise_poly::<Gf8B>(5, 0xB109);
    let right = noise_poly::<Gf8B>(4, 0xB10A);
    let batch = 1;
    let mut scratch = ConvolutionScratch::<Gf8B>::new(2, 2, batch).expect("small scratch");
    let full = left.coefficient_count() + right.coefficient_count() - 1;
    let mut output = vec![0_u8; full * batch * Gf8B::BYTES];
    assert_eq!(
        poly_ring::multiply_rows_into(
            &mut output,
            &rows(&left, left.coefficient_count(), batch),
            left.coefficient_count(),
            &rows(&right, right.coefficient_count(), batch),
            right.coefficient_count(),
            batch,
            full,
            &mut scratch,
        )
        .map(|_| ()),
        Err(poly_ring::ProductError::Config(
            poly_ring::ConfigError::ScratchTooSmall {
                context: "prepared operand coefficient capacity",
                required: 5,
                available: 2,
            }
        ))
    );
}

/// A short output buffer reports the byte-length mismatch exactly.
#[test]
fn short_output_reports_buffer_length() {
    let left = noise_poly::<Gf8B>(5, 0xB10B);
    let right = noise_poly::<Gf8B>(4, 0xB10C);
    let batch = 1;
    let mut scratch =
        ConvolutionScratch::<Gf8B>::new(left.coefficient_count(), right.coefficient_count(), batch)
            .expect("scratch");
    let full = left.coefficient_count() + right.coefficient_count() - 1;
    let mut output = vec![0_u8; full * batch * Gf8B::BYTES - 1];
    let error = poly_ring::multiply_rows_into(
        &mut output,
        &rows(&left, left.coefficient_count(), batch),
        left.coefficient_count(),
        &rows(&right, right.coefficient_count(), batch),
        right.coefficient_count(),
        batch,
        full,
        &mut scratch,
    )
    .expect_err("short output");
    assert!(matches!(
        error,
        poly_ring::ProductError::Config(poly_ring::ConfigError::BufferLength { .. })
    ));
}

/// Empty and batch-zero geometries succeed without touching the engine.
#[test]
fn empty_geometries_succeed_without_multiplying() {
    let batch = 1;
    let mut scratch = ConvolutionScratch::<Gf8B>::new(4, 4, batch).expect("scratch");
    let mut output = Vec::new();
    // Zero precision: no rows out, empty output.
    poly_ring::multiply_rows_into(&mut output, &[], 0, &[], 0, batch, 0, &mut scratch)
        .expect("empty");
    // Batch zero with empty buffers is a valid empty geometry.
    let mut scratch0 = ConvolutionScratch::<Gf8B>::new(4, 4, 0).expect("scratch");
    poly_ring::multiply_rows_into(&mut [], &[], 2, &[], 2, 0, 2, &mut scratch0)
        .expect("batch zero");
    // Explicitly caching a transform size is idempotent, even on the
    // Karatsuba-only domain where no plan exists.
    let _ = scratch.prepare_transform(8, batch);
    let _ = scratch.prepare_transform(8, batch);
    let _ = noise::<Gf16>(2, 0xB10D);
}

/// A `Gf8D` product whose operands both sit at the Karatsuba crossover
/// still routes through the fallback path — the transform engine has no
/// seat here — and every lane agrees with the naive scalar oracle.
#[test]
fn above_crossover_karatsuba_only_domain_matches_the_oracle() {
    let left = noise_poly::<Gf8D>(poly_ring::internals::KARATSUBA_CROSSOVER, 0xB10E);
    let right = noise_poly::<Gf8D>(poly_ring::internals::KARATSUBA_CROSSOVER + 3, 0xB10F);
    let batch = 2;
    let full = left.coefficient_count() + right.coefficient_count() - 1;
    let mut scratch =
        ConvolutionScratch::<Gf8D>::new(left.coefficient_count(), right.coefficient_count(), batch)
            .expect("scratch");
    let mut output = vec![0_u8; full * batch * Gf8D::BYTES];
    poly_ring::multiply_rows_into(
        &mut output,
        &rows(&left, left.coefficient_count(), batch),
        left.coefficient_count(),
        &rows(&right, right.coefficient_count(), batch),
        right.coefficient_count(),
        batch,
        full,
        &mut scratch,
    )
    .expect("multiply");
    for lane_index in 0..batch {
        assert_eq!(
            lane::<Gf8D>(&output, full, batch, lane_index),
            naive_multiply(&left, &right)
        );
    }
}

/// The `Auto` route falls back to schoolbook when the transform plan cannot
/// be built: a 300-coefficient Gf8B product exceeds the size-256 field cap,
/// so the naive oracle still agrees without any error.
#[cfg(feature = "fft")]
#[test]
fn auto_route_falls_back_past_plan_capacity() {
    let left = noise_poly::<Gf8B>(180, 0xB201);
    let right = noise_poly::<Gf8B>(160, 0xB202);
    let expected = {
        let mut product =
            vec![<Gf8B as Field>::Elem::ZERO; left.coefficient_count() + right.coefficient_count()];
        for (i, a) in left.coefficients().enumerate() {
            for (j, b) in right.coefficients().enumerate() {
                product[i + j] = product[i + j].add(a.mul(b));
            }
        }
        Polynomial::<Gf8B>::from_coefficients(&product).expect("naive")
    };
    let batch = 1;
    let mut scratch =
        ConvolutionScratch::<Gf8B>::new(left.coefficient_count(), right.coefficient_count(), batch)
            .expect("scratch");
    let full = left.coefficient_count() + right.coefficient_count() - 1;
    assert!(full > 256, "the full product must exceed the Gf8B plan cap");
    let mut output = vec![0_u8; full * batch * Gf8B::BYTES];
    poly_ring::multiply_rows_into(
        &mut output,
        &rows(&left, left.coefficient_count(), batch),
        left.coefficient_count(),
        &rows(&right, right.coefficient_count(), batch),
        right.coefficient_count(),
        batch,
        full,
        &mut scratch,
    )
    .expect("fallback multiply");
    assert_eq!(lane::<Gf8B>(&output, full, batch, 0), expected);
}

/// A forced transform route without a cached plan is an error carrying the
/// product size, never a silent fallback.
#[cfg(feature = "internals")]
#[test]
fn forced_transform_without_plan_is_an_error() {
    use poly_ring::internals::{ProductRoute, multiply_rows_route_into};

    // `Gf8D` has no transform route: any forced transform size misses the
    // plan cache, even with ample operand capacity.
    let left = noise_poly::<Gf8D>(5, 0xB203);
    let right = noise_poly::<Gf8D>(4, 0xB204);
    let batch = 1;
    let mut scratch =
        ConvolutionScratch::<Gf8D>::new(left.coefficient_count(), right.coefficient_count(), batch)
            .expect("scratch");
    let full = left.coefficient_count() + right.coefficient_count() - 1;
    let mut output = vec![0_u8; full * batch * Gf8D::BYTES];
    assert_eq!(
        multiply_rows_route_into(
            &mut output,
            &rows(&left, left.coefficient_count(), batch),
            left.coefficient_count(),
            &rows(&right, right.coefficient_count(), batch),
            right.coefficient_count(),
            batch,
            full,
            ProductRoute::Transform,
            &mut scratch,
        )
        .map(|_| ()),
        Err(poly_ring::ProductError::Config(
            poly_ring::ConfigError::ScratchTooSmall {
                context: "prepared transform plan",
                required: full,
                available: 0,
            }
        ))
    );
}

/// The `Auto` batch route falls back to schoolbook when the plan cannot be
/// built: twenty Gf8B pairs with a 339-coefficient full product exceed the
/// scalar batch-16 crossover (so the transform is selected) but need a
/// size-512 plan past the field cap (so construction fails). The batch
/// still matches the naive oracle exactly.
#[cfg(feature = "fft")]
#[test]
fn auto_batch_falls_back_past_plan_capacity() {
    use poly_ring::{PolynomialProductScratch, ProductStrategy, multiply_batch_truncated_into};

    let left = noise_poly::<Gf8B>(180, 0xB205);
    let right = noise_poly::<Gf8B>(160, 0xB206);
    let expected = {
        let mut product =
            vec![<Gf8B as Field>::Elem::ZERO; left.coefficient_count() + right.coefficient_count()];
        for (i, a) in left.coefficients().enumerate() {
            for (j, b) in right.coefficients().enumerate() {
                product[i + j] = product[i + j].add(a.mul(b));
            }
        }
        Polynomial::<Gf8B>::from_coefficients(&product).expect("naive")
    };
    let pairs = [(&left, &right); 20];
    let mut scratch = PolynomialProductScratch::<Gf8B>::new();
    let mut output = Vec::new();
    multiply_batch_truncated_into(
        &mut output,
        &pairs,
        usize::MAX,
        ProductStrategy::Auto,
        &mut scratch,
    )
    .expect("fallback batch");
    assert_eq!(output.len(), 20);
    for product in &output {
        assert_eq!(*product, expected);
    }
}

/// Prepared Goldilocks `Auto` agrees with the naive oracle around each
/// measured batch crossover.
#[cfg(feature = "fft")]
#[test]
fn goldilocks_auto_prepared_matches_oracle_across_crossover() {
    let shapes = [
        (255_usize, 255_usize, 1_usize),
        (256, 256, 1),
        (300, 260, 1),
        (127, 127, 4),
        (128, 128, 4),
        (160, 140, 4),
        (63, 63, 16),
        (64, 64, 16),
        (80, 72, 16),
    ];
    for (n, (left_len, right_len, batch)) in shapes.into_iter().enumerate() {
        let left = noise_poly::<Goldilocks>(left_len, 0xB300 + n as u64);
        let right = noise_poly::<Goldilocks>(right_len, 0xB400 + n as u64);
        let expected = naive_multiply(&left, &right);
        let full = left_len + right_len - 1;
        let mut scratch =
            ConvolutionScratch::<Goldilocks>::new(left_len, right_len, batch).expect("scratch");
        let mut output = vec![0_u8; full * batch * Goldilocks::BYTES];
        poly_ring::multiply_rows_into(
            &mut output,
            &rows(&left, left_len, batch),
            left_len,
            &rows(&right, right_len, batch),
            right_len,
            batch,
            full,
            &mut scratch,
        )
        .expect("prepared product");
        assert_eq!(
            lane::<Goldilocks>(&output, full, batch, 0),
            expected,
            "lane 0, shape {n}"
        );
        if batch > 1 {
            assert_eq!(
                lane::<Goldilocks>(&output, full, batch, batch - 1),
                expected,
                "last lane, shape {n}"
            );
        }
    }
}

/// Public Goldilocks multiplication agrees with the naive oracle around the
/// measured one-shot crossover.
#[cfg(feature = "fft")]
#[test]
fn goldilocks_multiply_matches_oracle_across_oneshot_crossover() {
    let crossover = poly_ring::cost::NTT_ONESHOT_PRODUCT_CROSSOVER;
    let shapes = [
        (crossover - 1, crossover - 1),
        (crossover, crossover),
        (crossover - 1, 4 * crossover),
        (crossover + 1, 2 * crossover),
    ];
    for (n, (left_len, right_len)) in shapes.into_iter().enumerate() {
        let left = noise_poly::<Goldilocks>(left_len, 0xB500 + n as u64);
        let right = noise_poly::<Goldilocks>(right_len, 0xB600 + n as u64);
        assert_eq!(
            left.multiply(&right).expect("multiply"),
            naive_multiply(&left, &right),
            "shape {n}"
        );
    }
}
