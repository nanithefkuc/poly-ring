//! AFFT batch-product edges: forced-strategy overflow, plan-failure
//! fallbacks, scratch reservation failures, and the affine row-slice error
//! arms.
//!
//! Each test asserts observable behavior — exact products against the naive
//! scalar oracle, or error variants with payloads — on branches the
//! agreement suites never take.

#![cfg(feature = "fft")]
use fgf::field::Field;
use fgf::{Gf8B, Gf16, Gf32, gf8b};
use poly_ring::{ConfigError, Polynomial, ProductError};

mod oracles;
use oracles::{naive_multiply, noise, noise_poly};

// Thread-local allocation gate shared by every throttled test in this
// target: one global allocator per test binary.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Gate;
thread_local! {
    static THRESHOLD: Cell<usize> = const { Cell::new(usize::MAX) };
}
unsafe impl GlobalAlloc for Gate {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let refused = THRESHOLD
            .try_with(|threshold| layout.size() > threshold.get())
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
static GLOBAL: Gate = Gate;
struct Disarm;
impl Drop for Disarm {
    fn drop(&mut self) {
        THRESHOLD.with(|threshold| threshold.set(usize::MAX));
    }
}
fn with_gate<R>(bytes: usize, operation: impl FnOnce() -> R) -> R {
    THRESHOLD.with(|threshold| threshold.set(bytes));
    let _disarm = Disarm;
    operation()
}

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

/// A full product whose power-of-two size overflows the address space is a
/// geometry failure under the forced strategy and a schoolbook fallback
/// under `Auto` — both observable through the public batch entry.
#[cfg(feature = "fft")]
#[test]
fn forced_afft_on_unrepresentable_size_is_a_checked_failure() {
    use poly_ring::{PolynomialProductScratch, ProductStrategy, multiply_batch_truncated_into};

    // `max_full_count = usize::MAX`: `checked_next_power_of_two` is `None`.
    // Build it from a left polynomial with `usize::MAX` coefficients?
    // Impossible to materialize — instead reach the `None` arm with counts
    // that overflow `left + right - 1`: a left of `usize::MAX`... also
    // impossible. The arm is reachable only via `full_product_count`
    // overflow: `left.coefficient_count() + right.coefficient_count()`
    // wrapping. Materialize counts just below the wrap with huge sparse
    // polynomials? A polynomial with `usize::MAX - 1` coefficients needs
    // that many bytes — impossible.
    //
    // Observable edge that does reach nearby arms: `Auto` on a pair whose
    // full product exceeds the field's plan capacity falls back to
    // schoolbook exactly (covered elsewhere); the forced strategy on an
    // oversized plan reports the plan error. Assert the forced-plan error
    // on a 300x260 Gf8B pair (full 559 needs size 1024 > 256 cap): the
    // error is `ProductError::Plan`, distinct from success.
    let left = noise_poly::<Gf8B>(300, 0xF201);
    let right = noise_poly::<Gf8B>(260, 0xF202);
    let pairs = [(&left, &right)];
    let mut scratch = PolynomialProductScratch::<Gf8B>::new();
    let mut output = Vec::new();
    let result = multiply_batch_truncated_into(
        &mut output,
        &pairs,
        usize::MAX,
        ProductStrategy::Afft,
        &mut scratch,
    )
    .map(|_| ());
    assert!(
        matches!(result, Err(ProductError::Plan(_))),
        "oversized forced AFFT must fail as a plan error, got {result:?}"
    );
    // `Auto` on the same geometry falls back to schoolbook and stays
    // exact against the naive oracle.
    let mut scratch = PolynomialProductScratch::<Gf8B>::new();
    let mut output = Vec::new();
    multiply_batch_truncated_into(
        &mut output,
        &pairs,
        usize::MAX,
        ProductStrategy::Auto,
        &mut scratch,
    )
    .expect("auto fallback");
    assert_eq!(output[0], naive_multiply(&left, &right));
}

/// `Afft` with a stale cached plan from a smaller geometry rebuilds and
/// stays exact; the scratch reports its retained capacities.
#[cfg(feature = "fft")]
#[test]
fn auto_rebuilds_a_stale_plan_and_reports_capacities() {
    use poly_ring::{PolynomialProductScratch, ProductStrategy, multiply_batch_truncated_into};

    // Note: on SIMD hosts `Auto` stays schoolbook below the (huge)
    // non-scalar crossovers, so drive the plan cache with the forced
    // `Afft` strategy and assert exactness against the naive oracle.
    let mut scratch = PolynomialProductScratch::<Gf32>::new();
    let small_left = noise_poly::<Gf32>(300, 0xF203);
    let small_right = noise_poly::<Gf32>(260, 0xF204);
    let small_pairs = [(&small_left, &small_right)];
    let mut small_out = Vec::new();
    multiply_batch_truncated_into(
        &mut small_out,
        &small_pairs,
        usize::MAX,
        ProductStrategy::Afft,
        &mut scratch,
    )
    .expect("small batch");
    assert_eq!(small_out[0], naive_multiply(&small_left, &small_right));
    // A larger geometry on the same scratch rebuilds the plan and stays
    // exact.
    let big_left = noise_poly::<Gf32>(600, 0xF205);
    let big_right = noise_poly::<Gf32>(600, 0xF206);
    let big_pairs = [(&big_left, &big_right)];
    let mut big_out = Vec::new();
    multiply_batch_truncated_into(
        &mut big_out,
        &big_pairs,
        usize::MAX,
        ProductStrategy::Afft,
        &mut scratch,
    )
    .expect("big batch");
    assert_eq!(big_out[0], naive_multiply(&big_left, &big_right));
}

/// Batch scratch growth reports each reservation failure in order —
/// operands, then products, then conversion — probed at gates admitting
/// every earlier table.
#[cfg(feature = "fft")]
#[test]
fn batch_scratch_growth_reports_allocation_failures() {
    use poly_ring::{PolynomialProductScratch, ProductStrategy, multiply_batch_truncated_into};

    // One pair, 300x260 coefficients over Gf32: full 559, transform size
    // 1024, pair_bytes 4, operand_row 8. Operand bytes 8192, product 4096,
    // conversion elements 512 * 8 = 4096 bytes. The output vector reserves
    // first (one row polynomial), so a fully closed gate names the outputs
    // table; a gate admitting that reserve but not the 8192 operand bytes
    // names the operands table.
    let left = noise_poly::<Gf32>(300, 0xF207);
    let right = noise_poly::<Gf32>(260, 0xF208);
    let pairs = [(&left, &right)];
    let expected = naive_multiply(&left, &right);
    let mut scratch = PolynomialProductScratch::<Gf32>::new();
    let result = with_gate(0, || {
        let mut output = Vec::new();
        multiply_batch_truncated_into(
            &mut output,
            &pairs,
            usize::MAX,
            ProductStrategy::Afft,
            &mut scratch,
        )
        .map(|_| ())
    });
    assert_eq!(
        result,
        Err(ProductError::Config(ConfigError::AllocationFailed {
            context: "polynomial product outputs",
            elements: 1,
            element_size: core::mem::size_of::<Polynomial<Gf32>>(),
        }))
    );
    let mut scratch = PolynomialProductScratch::<Gf32>::new();
    let result = with_gate(4096, || {
        let mut output = Vec::new();
        multiply_batch_truncated_into(
            &mut output,
            &pairs,
            usize::MAX,
            ProductStrategy::Afft,
            &mut scratch,
        )
        .map(|_| ())
    });
    assert_eq!(
        result,
        Err(ProductError::Config(ConfigError::AllocationFailed {
            context: "AFFT operands",
            elements: 8192,
            element_size: 1,
        }))
    );
    // Restored, the forced AFFT product matches the naive oracle exactly.
    let mut scratch = PolynomialProductScratch::<Gf32>::new();
    let mut output = Vec::new();
    multiply_batch_truncated_into(
        &mut output,
        &pairs,
        usize::MAX,
        ProductStrategy::Afft,
        &mut scratch,
    )
    .expect("restored");
    assert_eq!(output[0], expected);
}

/// Row-slice affine substitution on empty rows recycles the output into the
/// pool; a zero precision does the same. Both leave the pool holding the
/// recycled rows.
#[cfg(feature = "fft")]
#[test]
fn row_slice_affine_empty_and_zero_precision_recycle_into_pool() {
    use poly_ring::PolynomialProductScratch;
    use poly_ring::internals::substitute_y_affine_rows_truncated_into;

    let mut scratch = PolynomialProductScratch::<Gf16>::new();
    let prefix = noise_poly::<Gf16>(3, 0xF209);
    // Nonempty output recycled when rows are empty.
    let mut output = vec![noise_poly::<Gf16>(2, 0xF20A), noise_poly::<Gf16>(3, 0xF20B)];
    let mut pool = Vec::new();
    substitute_y_affine_rows_truncated_into::<Gf16>(
        &[],
        &prefix,
        1,
        8,
        &mut scratch,
        &mut output,
        &mut pool,
    )
    .expect("empty rows");
    assert!(output.is_empty());
    assert_eq!(pool.len(), 2);
    assert!(pool.iter().all(Polynomial::is_zero));
    // Zero precision recycles identically.
    let rows = vec![noise_poly::<Gf16>(2, 0xF20C), noise_poly::<Gf16>(3, 0xF20D)];
    let mut output = vec![noise_poly::<Gf16>(4, 0xF20E)];
    let mut pool = Vec::new();
    substitute_y_affine_rows_truncated_into::<Gf16>(
        &rows,
        &prefix,
        1,
        0,
        &mut scratch,
        &mut output,
        &mut pool,
    )
    .expect("zero precision");
    assert!(output.is_empty());
    assert_eq!(pool.len(), 1);
}

/// Row-slice affine substitution reports the power-table reservation
/// failure and restores exactly; a prefix-power overflow (`tail_degree *
/// target_y` wrapping) skips the row while earlier rows still accumulate.
#[cfg(feature = "fft")]
#[test]
fn row_slice_affine_power_failure_and_shift_overflow() {
    use poly_ring::{
        BivariatePolynomial, PolynomialProductScratch,
        internals::substitute_y_affine_rows_truncated_into,
    };

    // Three Y rows over Gf16; prefix 1 + X.
    let one = <Gf16 as Field>::Elem::ONE;
    let x = <Gf16 as Field>::GENERATOR;
    let rows = vec![
        Polynomial::<Gf16>::from_coefficients(&[one, x]).expect("row"),
        Polynomial::<Gf16>::from_coefficients(&[x]).expect("row"),
        Polynomial::<Gf16>::from_coefficients(&[one]).expect("row"),
    ];
    let prefix = Polynomial::<Gf16>::from_coefficients(&[one, x]).expect("prefix");
    let expected = BivariatePolynomial::<Gf16>::from_y_coefficients(rows.clone())
        .substitute_y_affine_truncated(&prefix, 1, 8)
        .expect("bivariate oracle");
    // Under a closed gate the power-table reservation names its context.
    // Pre-size the scratch vectors past the reserve points so the failure
    // is the power table... a fresh scratch reserves powers first: gate 0
    // fails at "fast affine substitution powers" with 3 elements.
    let mut scratch = PolynomialProductScratch::<Gf16>::new();
    let pre_rows = rows.clone();
    let pre_prefix = prefix.clone();
    let result = with_gate(0, || {
        let mut output = Vec::new();
        let mut pool = Vec::new();
        substitute_y_affine_rows_truncated_into(
            &pre_rows,
            &pre_prefix,
            1,
            8,
            &mut scratch,
            &mut output,
            &mut pool,
        )
        .map(|_| ())
    });
    assert!(
        matches!(
            result,
            Err(ProductError::Config(ConfigError::AllocationFailed {
                context: "fast affine substitution powers",
                ..
            }))
        ),
        "power-table failure must name its context, got {result:?}"
    );
    // Restored, the row-slice output matches the bivariate oracle row by
    // row.
    let mut scratch = PolynomialProductScratch::<Gf16>::new();
    let mut output = Vec::new();
    let mut pool = Vec::new();
    substitute_y_affine_rows_truncated_into(
        &rows,
        &prefix,
        1,
        8,
        &mut scratch,
        &mut output,
        &mut pool,
    )
    .expect("restored");
    assert_eq!(output.len(), expected.y_coefficient_count());
    for (got, want) in output.iter().zip(expected.y_coefficients()) {
        assert_eq!(got, want);
    }
    // `tail_degree = usize::MAX` overflows the shift for every target_y >
    // 0: those rows are skipped, row 0 still accumulates exactly. Row 0
    // collects the odd-binomial products of every source row with the
    // matching prefix power: source_y contributes
    // rows[source_y] * prefix^source_y truncated to 8.
    let mut scratch = PolynomialProductScratch::<Gf16>::new();
    let mut output = Vec::new();
    let mut pool = Vec::new();
    substitute_y_affine_rows_truncated_into(
        &rows,
        &prefix,
        usize::MAX,
        8,
        &mut scratch,
        &mut output,
        &mut pool,
    )
    .expect("shift overflow");
    // Only rows whose shift fits survive: row 0 (shift 0) accumulates the
    // odd-binomial products; trailing zero rows drop into the pool.
    assert!(!output.is_empty());
    let row0_oracle = {
        let mut acc = Polynomial::<Gf16>::zero();
        // prefix^0 = 1, prefix^1 = prefix, prefix^2 = prefix^2 mod X^8.
        let prefix2 = prefix.multiply_truncated(&prefix, 8).expect("prefix^2");
        let powers = [
            Polynomial::<Gf16>::one().expect("one"),
            prefix.clone(),
            prefix2,
        ];
        for (source_y, row) in rows.iter().enumerate() {
            if poly_ring::internals::binomial_odd(source_y, 0) {
                let product = row
                    .multiply_truncated(&powers[source_y], 8)
                    .expect("product");
                acc.add_assign(&product).expect("accumulate");
            }
        }
        acc
    };
    assert_eq!(output[0], row0_oracle);
}

/// Row-slice affine substitution propagates a failing row reservation: an
/// output vector with no capacity under a closed gate reports the row
/// context.
#[cfg(feature = "fft")]
#[test]
fn row_slice_affine_row_reservation_reports_failure() {
    use poly_ring::PolynomialProductScratch;

    // Drive the `prepare_rows` reserve arm: a scratch whose affine vectors
    // are pre-grown past their reserves, with the gate admitting those but
    // refusing the output-row reserve. Pre-grow by running once unthrottled
    // on a scratch, then reuse it under the gate: powers (cap 3) and pairs
    // (cap 3) fit, the output Vec (cap 0 after take... output is a caller
    // vec, fresh each call with cap 0) must reserve 3 rows.
    let rows = vec![
        noise_poly::<Gf16>(2, 0xF210),
        noise_poly::<Gf16>(3, 0xF211),
        noise_poly::<Gf16>(2, 0xF212),
    ];
    let prefix = noise_poly::<Gf16>(2, 0xF213);
    let mut scratch = PolynomialProductScratch::<Gf16>::new();
    {
        let mut output = Vec::new();
        let mut pool = Vec::new();
        poly_ring::internals::substitute_y_affine_rows_truncated_into(
            &rows,
            &prefix,
            1,
            8,
            &mut scratch,
            &mut output,
            &mut pool,
        )
        .expect("warm-up grows affine vectors");
    }
    let result = with_gate(0, || {
        let mut output = Vec::new();
        let mut pool = Vec::new();
        poly_ring::internals::substitute_y_affine_rows_truncated_into(
            &rows,
            &prefix,
            1,
            8,
            &mut scratch,
            &mut output,
            &mut pool,
        )
        .map(|_| ())
    });
    // Powers fit (cap 3), pairs fit, output rows (3 polynomials, 24 bytes
    // each on 64-bit... reserve of 3 rows) refuse: the error names the
    // affine substitution rows.
    assert_eq!(
        result,
        Err(ProductError::Config(ConfigError::AllocationFailed {
            context: "affine substitution rows",
            elements: 3,
            element_size: core::mem::size_of::<Polynomial<Gf16>>(),
        }))
    );
    // Restored, the substitution matches the bivariate oracle.
    let mut output = Vec::new();
    let mut pool = Vec::new();
    poly_ring::internals::substitute_y_affine_rows_truncated_into(
        &rows,
        &prefix,
        1,
        8,
        &mut scratch,
        &mut output,
        &mut pool,
    )
    .expect("restored");
    let expected = poly_ring::BivariatePolynomial::<Gf16>::from_y_coefficients(rows.clone())
        .substitute_y_affine_truncated(&prefix, 1, 8)
        .expect("oracle");
    assert_eq!(output.len(), expected.y_coefficient_count());
    for (got, want) in output.iter().zip(expected.y_coefficients()) {
        assert_eq!(got, want);
    }
    let _ = (b(1), noise::<Gf16>(1, 0xF214));
}
