//! Prepared-convolution, ring, and series coverage edges.
//!
//! Each test asserts observable behavior: batched lane products against a
//! naive scalar oracle, ring algebra identities, series-division round
//! trips, and `Config` error variants with payloads — on branches the
//! agreement suites never take.

// `as_chunks::<F::BYTES>()` needs a const generic depending on a type
// parameter, which Rust rejects in const argument position.
#![allow(clippy::chunks_exact_to_as_chunks)]

#[cfg(all(feature = "fft", feature = "internals"))]
use fgf::Gf8D;
#[cfg(feature = "fft")]
use fgf::Gf16;
use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Goldilocks, Mersenne31};
use poly_ring::{
    ConfigError, ConvolutionScratch, Polynomial, PolynomialError, ProductError, multiply_rows_into,
    series_divide,
};

mod oracles;

// ---------------------------------------------------------------------------
// Local packing helpers: coefficient-major lane rows for the prepared engine.

fn pack<F: FieldKernels>(coefficients: &[F::Elem]) -> Vec<u8> {
    let mut rows = vec![0_u8; coefficients.len() * F::BYTES];
    for (degree, value) in coefficients.iter().enumerate() {
        F::encode(&mut rows[degree * F::BYTES..][..F::BYTES], *value);
    }
    rows
}

/// Pack `lanes` independent coefficient lists into coefficient-major rows.
fn pack_batch<F: FieldKernels>(lanes: &[Vec<F::Elem>]) -> Vec<u8> {
    let batch = lanes.len();
    let count = lanes[0].len();
    let mut rows = vec![0_u8; count * batch * F::BYTES];
    for (lane, coefficients) in lanes.iter().enumerate() {
        assert_eq!(coefficients.len(), count, "lanes share one row count");
        for (degree, value) in coefficients.iter().enumerate() {
            F::encode(
                &mut rows[(degree * batch + lane) * F::BYTES..][..F::BYTES],
                *value,
            );
        }
    }
    rows
}

fn lane_coefficient<F: FieldKernels>(
    rows: &[u8],
    batch: usize,
    degree: usize,
    lane: usize,
) -> F::Elem {
    F::decode(&rows[(degree * batch + lane) * F::BYTES..][..F::BYTES])
}

/// Naive per-coefficient convolution over scalar elements.
fn naive_product<F: FieldKernels>(left: &[F::Elem], right: &[F::Elem]) -> Vec<F::Elem> {
    let mut product = vec![F::Elem::ZERO; left.len() + right.len().saturating_sub(1)];
    for (left_index, a) in left.iter().enumerate() {
        for (right_index, b) in right.iter().enumerate() {
            let accumulated = product[left_index + right_index].add(a.mul(*b));
            product[left_index + right_index] = accumulated;
        }
    }
    product
}

/// Composition by its definition: Horner over the outer coefficients using
/// only public polynomial operations and the shared naive product.
fn naive_compose<F: FieldKernels>(outer: &Polynomial<F>, inner: &Polynomial<F>) -> Polynomial<F> {
    let mut result = Polynomial::zero();
    for coefficient in outer.coefficients().rev() {
        let scaled = oracles::naive_multiply(&result, inner);
        let mut coefficients: Vec<F::Elem> = scaled.coefficients().collect();
        match coefficients.first_mut() {
            Some(constant) => *constant = constant.add(coefficient),
            None => coefficients.push(coefficient),
        }
        result = Polynomial::from_coefficients(&coefficients).expect("composition accumulator");
    }
    result
}

// ---------------------------------------------------------------------------
// A one-off allocation floor for reserve-failure tests: allocations strictly
// smaller than the floor are refused, larger ones pass. One global allocator
// per test binary; the floor is thread-local and restored on scope exit.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct SmallAllocationFloor;
thread_local! {
    static FLOOR: Cell<usize> = const { Cell::new(0) };
}
unsafe impl GlobalAlloc for SmallAllocationFloor {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let refused = FLOOR
            .try_with(|floor| layout.size() < floor.get())
            .unwrap_or(false)
            && !std::thread::panicking();
        if refused {
            std::ptr::null_mut()
        } else {
            unsafe { System.alloc(layout) }
        }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static GLOBAL: SmallAllocationFloor = SmallAllocationFloor;
struct Restore;
impl Drop for Restore {
    fn drop(&mut self) {
        FLOOR.with(|floor| floor.set(0));
    }
}
fn with_floor<R>(bytes: usize, operation: impl FnOnce() -> R) -> R {
    FLOOR.with(|floor| floor.set(bytes));
    let _restore = Restore;
    operation()
}

// ---------------------------------------------------------------------------
// Prepared-engine products.

/// Batched lane products match the scalar oracle across batch widths and
/// operand lengths straddling the small power-of-two boundaries, at full and
/// truncated precisions.
#[test]
fn batched_lane_products_match_the_scalar_oracle() {
    let geometries = [
        (1_usize, 1_usize),
        (3, 4),
        (4, 5),
        (7, 8),
        (8, 8),
        (8, 9),
        (15, 16),
        (16, 16),
        (16, 17),
        (17, 9),
    ];
    for batch in [1_usize, 2, 5] {
        let mut scratch = ConvolutionScratch::<Gf8B>::new(17, 17, batch).expect("scratch");
        for (left_len, right_len) in geometries {
            let left: Vec<Vec<_>> = (0..batch)
                .map(|lane| oracles::noise::<Gf8B>(left_len, 0xC101 + lane as u64))
                .collect();
            let right: Vec<Vec<_>> = (0..batch)
                .map(|lane| oracles::noise::<Gf8B>(right_len, 0xC201 + lane as u64))
                .collect();
            let expected: Vec<Vec<_>> = (0..batch)
                .map(|lane| naive_product::<Gf8B>(&left[lane], &right[lane]))
                .collect();
            let full = left_len + right_len - 1;
            let packed_left = pack_batch::<Gf8B>(&left);
            let packed_right = pack_batch::<Gf8B>(&right);
            let mut precisions = vec![full, full - 1, 3, 1];
            precisions.sort_unstable();
            precisions.dedup();
            for precision in precisions {
                let out_rows = full.min(precision);
                let mut output = vec![0_u8; out_rows * batch * Gf8B::BYTES];
                multiply_rows_into::<Gf8B>(
                    &mut output,
                    &packed_left,
                    left_len,
                    &packed_right,
                    right_len,
                    batch,
                    precision,
                    &mut scratch,
                )
                .expect("batched product");
                for (lane, expected_lane) in expected.iter().enumerate() {
                    for (degree, expected_coefficient) in
                        expected_lane.iter().enumerate().take(out_rows)
                    {
                        assert_eq!(
                            lane_coefficient::<Gf8B>(&output, batch, degree, lane),
                            *expected_coefficient,
                            "lane {lane} coefficient {degree} at precision {precision}"
                        );
                    }
                }
            }
        }
    }
}

/// The same oracle agreement over a wide field, exercising the byte-lane
/// gather/scatter at an eight-byte element width.
#[test]
fn batched_goldilocks_products_match_the_scalar_oracle() {
    let mut scratch = ConvolutionScratch::<Goldilocks>::new(20, 18, 2).expect("scratch");
    for (left_len, right_len) in [(3_usize, 5_usize), (16, 17), (20, 18)] {
        let left: Vec<Vec<_>> = (0..2)
            .map(|lane| oracles::noise::<Goldilocks>(left_len, 0xC301 + lane as u64))
            .collect();
        let right: Vec<Vec<_>> = (0..2)
            .map(|lane| oracles::noise::<Goldilocks>(right_len, 0xC401 + lane as u64))
            .collect();
        let expected: Vec<Vec<_>> = (0..2)
            .map(|lane| naive_product::<Goldilocks>(&left[lane], &right[lane]))
            .collect();
        let full = left_len + right_len - 1;
        let packed_left = pack_batch::<Goldilocks>(&left);
        let packed_right = pack_batch::<Goldilocks>(&right);
        let mut output = vec![0_u8; full * 2 * Goldilocks::BYTES];
        multiply_rows_into::<Goldilocks>(
            &mut output,
            &packed_left,
            left_len,
            &packed_right,
            right_len,
            2,
            full,
            &mut scratch,
        )
        .expect("wide-field batched product");
        for (lane, expected_lane) in expected.iter().enumerate() {
            for (degree, expected_coefficient) in expected_lane.iter().enumerate() {
                assert_eq!(
                    lane_coefficient::<Goldilocks>(&output, 2, degree, lane),
                    *expected_coefficient,
                    "lane {lane} coefficient {degree}"
                );
            }
        }
    }
}

/// A product past the Karatsuba crossover matches the scalar oracle on the
/// fallback route.
#[test]
fn karatsuba_range_product_matches_the_scalar_oracle() {
    let left_len = 2100_usize;
    let right_len = 2048_usize;
    let left = oracles::noise::<Gf8B>(left_len, 0xC501);
    let right = oracles::noise::<Gf8B>(right_len, 0xC502);
    let expected = naive_product::<Gf8B>(&left, &right);
    let full = left_len + right_len - 1;
    let mut scratch = ConvolutionScratch::<Gf8B>::new(left_len, right_len, 1).expect("scratch");
    let mut output = vec![0_u8; full * Gf8B::BYTES];
    multiply_rows_into::<Gf8B>(
        &mut output,
        &pack::<Gf8B>(&left),
        left_len,
        &pack::<Gf8B>(&right),
        right_len,
        1,
        full,
        &mut scratch,
    )
    .expect("karatsuba-range product");
    for (degree, coefficient) in expected.iter().enumerate() {
        assert_eq!(
            lane_coefficient::<Gf8B>(&output, 1, degree, 0),
            *coefficient,
            "coefficient {degree}"
        );
    }
}

/// Empty geometries are valid and write no rows: an empty operand, a zero
/// precision, and a zero batch all succeed.
#[test]
fn empty_geometries_write_no_rows() {
    let mut scratch = ConvolutionScratch::<Gf8B>::new(4, 4, 2).expect("scratch");
    let lanes = [
        oracles::noise::<Gf8B>(4, 0xC601),
        oracles::noise::<Gf8B>(4, 0xC602),
    ];
    let operand = pack_batch::<Gf8B>(&lanes);
    let empty: &[u8] = &[];
    for (left_count, right_count, batch, precision) in [
        (0_usize, 4_usize, 2_usize, 8_usize),
        (4, 0, 2, 8),
        (4, 4, 2, 0),
        (0, 0, 0, 8),
    ] {
        let sliced_left = if left_count == 0 {
            empty
        } else {
            &operand[..left_count * batch * Gf8B::BYTES]
        };
        let sliced_right = if right_count == 0 {
            empty
        } else {
            &operand[..right_count * batch * Gf8B::BYTES]
        };
        multiply_rows_into::<Gf8B>(
            &mut [],
            sliced_left,
            left_count,
            sliced_right,
            right_count,
            batch,
            precision,
            &mut scratch,
        )
        .expect("empty geometry is valid");
    }
}

/// A row-count addition that overflows the address space fails as a geometry
/// error naming the product rows, before any capacity is consulted. A zero
/// batch keeps the operand buffers empty, so validation reaches the row
/// arithmetic.
#[test]
fn row_count_overflow_is_a_checked_failure() {
    let mut scratch = ConvolutionScratch::<Gf8B>::new(0, 0, 0).expect("scratch");
    let result = multiply_rows_into::<Gf8B>(&mut [], &[], usize::MAX, &[], 2, 0, 1, &mut scratch)
        .map(|_| ());
    assert_eq!(
        result,
        Err(ProductError::Config(ConfigError::GeometryOverflow {
            context: "prepared product rows",
        }))
    );
}

/// Mis-sized operand and output buffers, and capacities exceeded, each report
/// their own context with the exact expected and actual sizes.
#[test]
fn buffer_and_capacity_violations_report_their_contexts() {
    let left = oracles::noise::<Gf8B>(4, 0xC701);
    let right = oracles::noise::<Gf8B>(3, 0xC702);
    let full = 4 + 3 - 1;
    let packed_left = pack::<Gf8B>(&left);
    let packed_right = pack::<Gf8B>(&right);
    let mut exact_scratch = ConvolutionScratch::<Gf8B>::new(4, 3, 1).expect("scratch");
    let mut output = vec![0_u8; full * Gf8B::BYTES];

    // A left operand one row short.
    let result = multiply_rows_into::<Gf8B>(
        &mut output,
        &packed_left[..3 * Gf8B::BYTES],
        4,
        &packed_right,
        3,
        1,
        full,
        &mut exact_scratch,
    )
    .map(|_| ());
    assert_eq!(
        result,
        Err(ProductError::Config(ConfigError::BufferLength {
            context: "prepared left operand rows",
            expected: 4,
            actual: 3,
        }))
    );

    // An output one row too long.
    let mut long_output = vec![0_u8; (full + 1) * Gf8B::BYTES];
    let result = multiply_rows_into::<Gf8B>(
        &mut long_output,
        &packed_left,
        4,
        &packed_right,
        3,
        1,
        full,
        &mut exact_scratch,
    )
    .map(|_| ());
    assert_eq!(
        result,
        Err(ProductError::Config(ConfigError::BufferLength {
            context: "prepared output rows",
            expected: full,
            actual: full + 1,
        }))
    );

    // Operand coefficients beyond the prepared capacity.
    let mut small_scratch = ConvolutionScratch::<Gf8B>::new(3, 3, 1).expect("scratch");
    let result = multiply_rows_into::<Gf8B>(
        &mut output,
        &packed_left,
        4,
        &packed_right,
        3,
        1,
        full,
        &mut small_scratch,
    )
    .map(|_| ());
    assert_eq!(
        result,
        Err(ProductError::Config(ConfigError::ScratchTooSmall {
            context: "prepared operand coefficient capacity",
            required: 4,
            available: 3,
        }))
    );

    // A batch beyond the prepared lane capacity.
    let mut batched_left = vec![0_u8; 4 * 2 * Gf8B::BYTES];
    batched_left[..packed_left.len()].copy_from_slice(&packed_left);
    let mut batched_right = vec![0_u8; 3 * 2 * Gf8B::BYTES];
    batched_right[..packed_right.len()].copy_from_slice(&packed_right);
    let mut batched_output = vec![0_u8; full * 2 * Gf8B::BYTES];
    let result = multiply_rows_into::<Gf8B>(
        &mut batched_output,
        &batched_left,
        4,
        &batched_right,
        3,
        2,
        full,
        &mut exact_scratch,
    )
    .map(|_| ());
    assert_eq!(
        result,
        Err(ProductError::Config(ConfigError::ScratchTooSmall {
            context: "prepared lane capacity",
            required: 2,
            available: 1,
        }))
    );
}

/// A forced transform route on a field whose polynomials cannot carry the
/// requested transform size — beyond the plan cap for the field, or a field
/// with no transform route at all — is a capacity error, not a silent
/// fallback.
#[cfg(all(feature = "fft", feature = "internals"))]
#[test]
fn forced_transform_without_a_plan_is_an_error() {
    use poly_ring::internals::{ProductRoute, multiply_rows_route_into};

    // Gf8B caps its additive transform at the field order: a full product
    // of 559 coefficients needs a size-1024 transform, which has no plan.
    let left = oracles::noise::<Gf8B>(300, 0xC801);
    let right = oracles::noise::<Gf8B>(260, 0xC802);
    let full = 300 + 260 - 1;
    let mut scratch = ConvolutionScratch::<Gf8B>::new(300, 260, 1).expect("scratch");
    let mut output = vec![0_u8; full * Gf8B::BYTES];
    let result = multiply_rows_route_into::<Gf8B>(
        &mut output,
        &pack::<Gf8B>(&left),
        300,
        &pack::<Gf8B>(&right),
        260,
        1,
        full,
        ProductRoute::Transform,
        &mut scratch,
    )
    .map(|_| ());
    assert_eq!(
        result,
        Err(ProductError::Config(ConfigError::ScratchTooSmall {
            context: "prepared transform plan",
            required: full,
            available: 0,
        }))
    );

    // Gf8D caps its additive transform at the field order: a full product
    // of 289 coefficients needs a size-512 transform, which has no plan.
    let left = oracles::noise::<Gf8D>(150, 0xC803);
    let right = oracles::noise::<Gf8D>(140, 0xC804);
    let full = 150 + 140 - 1;
    let mut scratch = ConvolutionScratch::<Gf8D>::new(150, 140, 1).expect("scratch");
    let mut output = vec![0_u8; full * Gf8D::BYTES];
    let result = multiply_rows_route_into::<Gf8D>(
        &mut output,
        &pack::<Gf8D>(&left),
        150,
        &pack::<Gf8D>(&right),
        140,
        1,
        full,
        ProductRoute::Transform,
        &mut scratch,
    )
    .map(|_| ());
    assert_eq!(
        result,
        Err(ProductError::Config(ConfigError::ScratchTooSmall {
            context: "prepared transform plan",
            required: full,
            available: 0,
        }))
    );
}

/// `Auto` on a geometry whose transform size exceeds the field's domain —
/// a Gf16 product past 2^16 coefficients — finds no cached plan, rebuilds
/// nothing, and falls back to an exact lane-by-lane product.
#[cfg(feature = "fft")]
#[test]
fn auto_beyond_the_transform_cap_falls_back_exactly() {
    // Batch four puts the Auto crossover at 65_535 full coefficients on
    // packed backends (255 on scalar); a full size of 65_537 exceeds both
    // and needs a size-131_072 transform, past Gf16's 2^16-point domain.
    let batch = 4_usize;
    let left_len = 65_536_usize;
    let right_len = 2_usize;
    let full = left_len + right_len - 1;
    let mut left = vec![<Gf16 as Field>::Elem::ZERO; left_len];
    let endpoints = oracles::noise::<Gf16>(2, 0xC901);
    left[0] = endpoints[0];
    left[left_len - 1] = endpoints[1];
    let right = oracles::noise::<Gf16>(right_len, 0xC902);
    // (a0 + a1·X^{65535}) · (b0 + b1·X) has exactly four nonzero terms.
    let expected_terms = [
        (0_usize, endpoints[0].mul(right[0])),
        (1, endpoints[0].mul(right[1])),
        (left_len - 1, endpoints[1].mul(right[0])),
        (left_len, endpoints[1].mul(right[1])),
    ];
    let lanes: Vec<Vec<_>> = (0..batch).map(|_| left.clone()).collect();
    let packed_left = pack_batch::<Gf16>(&lanes);
    let right_lanes: Vec<Vec<_>> = (0..batch).map(|_| right.clone()).collect();
    let packed_right = pack_batch::<Gf16>(&right_lanes);
    let mut expected = vec![0_u8; full * batch * Gf16::BYTES];
    for lane in 0..batch {
        for (degree, value) in expected_terms {
            Gf16::encode(
                &mut expected[(degree * batch + lane) * Gf16::BYTES..][..Gf16::BYTES],
                value,
            );
        }
    }
    let mut scratch = ConvolutionScratch::<Gf16>::new(left_len, right_len, batch).expect("scratch");
    let mut output = vec![0_u8; full * batch * Gf16::BYTES];
    multiply_rows_into::<Gf16>(
        &mut output,
        &packed_left,
        left_len,
        &packed_right,
        right_len,
        batch,
        full,
        &mut scratch,
    )
    .expect("auto product beyond the transform cap");
    assert_eq!(output, expected, "the fallback product stays exact");
}

// ---------------------------------------------------------------------------
// Ring algebra identities.

/// Distributivity, squaring against multiplication, and additive cancellation
/// hold over both characteristics.
#[test]
fn ring_identities_hold_over_both_characteristics() {
    let a_len = 23_usize;
    let b_len = 17_usize;
    // Gf8B (characteristic two): the spread square path.
    let a = oracles::noise_poly::<Gf8B>(a_len, 0xCA01);
    let b = oracles::noise_poly::<Gf8B>(b_len, 0xCA02);
    let c = oracles::noise_poly::<Gf8B>(b_len, 0xCA03);
    let sum = b.add(&c).expect("sum");
    let distributed = a.multiply(&sum).expect("left product");
    let termwise = a
        .multiply(&b)
        .expect("first product")
        .add(&a.multiply(&c).expect("second product"))
        .expect("termwise sum");
    assert_eq!(distributed, termwise, "a·(b+c) = a·b + a·c");
    assert_eq!(
        a.square().expect("square"),
        a.multiply(&a).expect("product")
    );
    let restored = a.add(&b).expect("sum").sub(&b).expect("difference");
    assert_eq!(restored, a, "(a+b)−b = a");

    // Mersenne31 (odd characteristic): the general square path.
    let a = oracles::noise_poly::<Mersenne31>(a_len, 0xCA04);
    let b = oracles::noise_poly::<Mersenne31>(b_len, 0xCA05);
    let c = oracles::noise_poly::<Mersenne31>(b_len, 0xCA06);
    let sum = b.add(&c).expect("sum");
    let distributed = a.multiply(&sum).expect("left product");
    let termwise = a
        .multiply(&b)
        .expect("first product")
        .add(&a.multiply(&c).expect("second product"))
        .expect("termwise sum");
    assert_eq!(distributed, termwise, "a·(b+c) = a·b + a·c over M31");
    assert_eq!(
        a.square().expect("square"),
        a.multiply(&a).expect("product"),
        "squaring equals multiplication over M31"
    );
    let restored = a.add(&b).expect("sum").sub(&b).expect("difference");
    assert_eq!(restored, a, "(a+b)−b = a over M31");
}

/// Division by a power of X drops the valuation exactly and refuses a
/// valuation below the power.
#[test]
fn division_by_x_power_drops_the_valuation() {
    // X^7 + X^5 has valuation five.
    let coefficients = oracles::noise::<Gf8B>(8, 0xCB01);
    let mut sparse = vec![<Gf8B as Field>::Elem::ZERO; 8];
    sparse[5] = coefficients[5];
    sparse[7] = coefficients[7];
    let polynomial = Polynomial::<Gf8B>::from_coefficients(&sparse).expect("polynomial");
    let divided = polynomial.divide_by_x_power(5).expect("exact division");
    let expected = Polynomial::<Gf8B>::from_coefficients(&[
        coefficients[5],
        <Gf8B as Field>::Elem::ZERO,
        coefficients[7],
    ])
    .expect("expected");
    assert_eq!(divided, expected, "X^7 + X^5 divided by X^5 is 1 + X^2");
    assert_eq!(
        polynomial.divide_by_x_power(6),
        Err(PolynomialError::NonExactDivision),
        "a valuation of five cannot lose six powers of X"
    );
    let undisturbed = polynomial.divide_by_x_power(0).expect("power zero");
    assert_eq!(undisturbed, polynomial, "division by X^0 is the identity");
}

/// Composition with an affine or arbitrary inner polynomial matches the
/// Horner definition, including inner zero coefficients that skip the
/// constant-term update.
#[test]
fn composition_matches_the_naive_definition() {
    for seed in [0xCC01_u64, 0xCC02] {
        let outer_coefficients = oracles::noise::<Gf8B>(9, seed);
        // Zero interior coefficients exercise the skip branch.
        let mut outer = outer_coefficients;
        outer[1] = <Gf8B as Field>::Elem::ZERO;
        outer[4] = <Gf8B as Field>::Elem::ZERO;
        let outer = Polynomial::from_coefficients(&outer).expect("outer");
        let affine_values = oracles::noise::<Gf8B>(2, seed + 1);
        let inner = oracles::noise_poly::<Gf8B>(3, seed + 2);

        let expected_affine = naive_compose(
            &outer,
            &Polynomial::from_coefficients(&affine_values).expect("affine"),
        );
        let composed = outer
            .compose_linear(affine_values[0], affine_values[1])
            .expect("affine composition");
        assert_eq!(composed, expected_affine, "affine composition matches");

        let expected = naive_compose(&outer, &inner);
        assert_eq!(
            outer.compose(&inner).expect("composition"),
            expected,
            "general composition matches"
        );
    }
}

/// When the constant-term reserve inside a composition loop fails, both
/// composition forms report the coefficient context with the element width;
/// restored, the same compositions succeed.
#[test]
fn composition_reports_the_coefficient_reserve_failure() {
    // The reserve for one coefficient byte-width is the smallest allocation
    // either composition form makes along its loop; a floor of two element
    // widths refuses exactly that, after the two-element affine polynomial
    // itself has been packed.
    let outer = oracles::noise_poly::<Gf8B>(5, 0xCD01);
    let affine_values = oracles::noise::<Gf8B>(2, 0xCD02);
    let inner = oracles::noise_poly::<Gf8B>(3, 0xCD03);
    let floor = 2 * Gf8B::BYTES;
    let result = with_floor(floor, || {
        outer.compose_linear(affine_values[0], affine_values[1])
    });
    assert_eq!(
        result,
        Err(PolynomialError::Config(ConfigError::AllocationFailed {
            context: "polynomial coefficients",
            elements: Gf8B::BYTES,
            element_size: 1,
        }))
    );
    let result = with_floor(floor, || outer.compose(&inner));
    assert_eq!(
        result,
        Err(PolynomialError::Config(ConfigError::AllocationFailed {
            context: "polynomial coefficients",
            elements: Gf8B::BYTES,
            element_size: 1,
        }))
    );
    let expected_affine = naive_compose(
        &outer,
        &Polynomial::from_coefficients(&affine_values).expect("affine"),
    );
    assert_eq!(
        outer
            .compose_linear(affine_values[0], affine_values[1])
            .expect("restored affine composition"),
        expected_affine
    );
    assert_eq!(
        outer.compose(&inner).expect("restored composition"),
        naive_compose(&outer, &inner)
    );
}

// ---------------------------------------------------------------------------
// Truncated power series.

/// Newton inversion matches the linear solver over both characteristics,
/// across precisions that straddle the doubling steps; zero precision yields
/// the zero polynomial and a non-unit constant is refused.
#[test]
fn series_inversion_matches_the_linear_solver() {
    fn check_at_precisions<F: FieldKernels>(unit: &Polynomial<F>, field: &str) {
        let len = unit.coefficient_count();
        for precision in [0_usize, 1, 2, 3, 5, 8, 16, 17, len, len + 4] {
            let newton = unit
                .inverse_mod_x_power(precision)
                .unwrap_or_else(|error| panic!("{field} inversion at {precision}: {error}"));
            let linear = oracles::naive_series_inverse(unit, precision);
            assert_eq!(newton, linear, "{field} inversion at precision {precision}");
            if precision == 0 {
                assert!(newton.is_zero(), "{field} inversion mod x^0 is zero");
            }
        }
    }
    check_at_precisions(&oracles::noise_unit::<Gf8B>(11, 0xCE01), "Gf8B");
    check_at_precisions(&oracles::noise_unit::<Gf8B>(33, 0xCE02), "Gf8B");
    check_at_precisions(&oracles::noise_unit::<Mersenne31>(11, 0xCE03), "Mersenne31");
    check_at_precisions(&oracles::noise_unit::<Mersenne31>(33, 0xCE04), "Mersenne31");

    // A zero constant coefficient is refused by both solvers, and by series
    // division through the divisor.
    let mut coefficients = oracles::noise::<Gf8B>(6, 0xCE05);
    coefficients[0] = <Gf8B as Field>::Elem::ZERO;
    let zero_constant = Polynomial::from_coefficients(&coefficients).expect("series");
    assert_eq!(
        zero_constant.inverse_mod_x_power(4),
        Err(PolynomialError::ZeroConstantTerm {
            context: "truncated power-series inversion",
        })
    );
    let unit = oracles::noise_unit::<Gf8B>(4, 0xCE06);
    assert_eq!(
        series_divide(&zero_constant, &zero_constant, 4),
        Err(PolynomialError::ZeroConstantTerm {
            context: "truncated power-series inversion",
        })
    );
    assert_eq!(
        series_divide(&unit, &zero_constant, 4),
        Err(PolynomialError::ZeroConstantTerm {
            context: "truncated power-series inversion",
        })
    );

    // The zero polynomial is not invertible either.
    let zero = Polynomial::<Gf8B>::zero();
    assert_eq!(
        zero.inverse_mod_x_power(4),
        Err(PolynomialError::ZeroConstantTerm {
            context: "truncated power-series inversion",
        })
    );
}

/// Series division round-trips a truncated product: `(a·b) / b` returns `a`
/// modulo the precision, at precisions below, at, and beyond `a`'s degree.
#[test]
fn series_divide_round_trips_truncated_products() {
    for (len, seed) in [(9_usize, 0xCF01_u64), (20, 0xCF02)] {
        let a = oracles::noise_poly::<Gf8B>(len, seed);
        let b = oracles::noise_unit::<Gf8B>(len - 2, seed + 1);
        let product = oracles::naive_multiply(&a, &b);
        let a_coefficients: Vec<_> = a.coefficients().collect();
        for precision in [1_usize, 2, len - 1, len, len + 3] {
            let quotient = series_divide(&product, &b, precision)
                .unwrap_or_else(|e| panic!("division at precision {precision}: {e}"));
            for (degree, expected_coefficient) in a_coefficients.iter().enumerate().take(precision)
            {
                assert_eq!(
                    quotient.coefficient(degree),
                    *expected_coefficient,
                    "quotient coefficient {degree} at precision {precision}"
                );
            }
        }
    }
}

/// Reversal is an involution on nonzero polynomials and fixes zero.
#[test]
fn series_reversal_is_an_involution() {
    let polynomial = oracles::noise_poly::<Gf8B>(7, 0xD001);
    assert_eq!(
        polynomial.reverse().reverse(),
        polynomial,
        "reversal twice is the identity"
    );
    let coefficients = oracles::noise::<Mersenne31>(4, 0xD002);
    let reversed = Polynomial::<Mersenne31>::from_coefficients(&[
        coefficients[3],
        coefficients[2],
        coefficients[1],
        coefficients[0],
    ])
    .expect("expected");
    assert_eq!(
        Polynomial::from_coefficients(&coefficients)
            .expect("polynomial")
            .reverse(),
        reversed,
        "reversal reorders coefficients"
    );
    assert!(
        Polynomial::<Gf8B>::zero().reverse().is_zero(),
        "zero reverses to zero"
    );
}

/// Preparing an empty transform geometry is a no-op.
#[test]
fn empty_transform_preparation_succeeds() {
    let mut scratch = ConvolutionScratch::<Gf8B>::new(4, 4, 1).expect("scratch");
    scratch.prepare_transform(0, 1).expect("empty geometry");
}
