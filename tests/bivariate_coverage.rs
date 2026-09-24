//! Edge and reservation-failure coverage for the bivariate surfaces.
//!
//! Bivariate row assignment and affine substitution are exercised at the
//! boundaries their error arms guard: partial field elements, row-table
//! growth under a failing allocator, and the per-row truncated product that
//! fails only after the prefix-power and staging tables fit.

use fgf::field::Field;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::{BivariatePolynomial, ConfigError, Polynomial, PolynomialError};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

fn row(coefficients: &[u8]) -> Polynomial<Gf8B> {
    let elements: Vec<_> = coefficients.iter().map(|&raw| b(raw)).collect();
    Polynomial::from_coefficients(&elements).expect("row polynomial")
}

// ── Test-only failing allocator ───────────────────────────────────────────────
//
// Thread-local gates, so parallel tests in this binary never charge each
// other: a size threshold and an exact-size allowance.

struct FailingAllocator;

thread_local! {
    static THRESHOLD: Cell<usize> = const { Cell::new(usize::MAX) };
}

unsafe impl GlobalAlloc for FailingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let refused = THRESHOLD
            .try_with(|gate| layout.size() > gate.get())
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
static GLOBAL: FailingAllocator = FailingAllocator;

/// Arms the threshold gate for one operation and disarms it however the
/// operation leaves.
fn with_threshold<R>(bytes: usize, operation: impl FnOnce() -> R) -> R {
    struct Disarm;
    impl Drop for Disarm {
        fn drop(&mut self) {
            THRESHOLD.with(|gate| gate.set(usize::MAX));
        }
    }
    THRESHOLD.with(|gate| gate.set(bytes));
    let _disarm = Disarm;
    operation()
}

/// Packed row assignment round-trips two rows, rejects a partial field
/// element before any growth happens, and reports the row-table reservation
/// failure with the requested count.
#[test]
fn packed_row_assignment_round_trips_and_reports_reservation_failure() {
    let mut bivariate = BivariatePolynomial::<Gf8B>::zero();
    let rows = [vec![1_u8, 2], vec![3_u8]];
    let packed: Vec<&[u8]> = rows.iter().map(|row| row.as_slice()).collect();
    bivariate
        .assign_y_coefficients_packed(packed.iter().copied())
        .expect("assignment");
    assert_eq!(bivariate.y_coefficient_count(), 2);
    for (degree, expected) in bivariate.y_coefficients().iter().zip(&rows) {
        let rendered: Vec<_> = degree.coefficients().collect();
        let wanted: Vec<_> = expected.iter().map(|&raw| b(raw)).collect();
        assert_eq!(rendered, wanted);
    }

    // Over GF(31) a two-byte row is a partial field element: both lengths
    // are reported.
    let mut truncated = BivariatePolynomial::<Mersenne31>::zero();
    let partial = vec![1_u8, 2];
    let error = truncated
        .assign_y_coefficients_packed(std::iter::once(partial.as_slice()))
        .expect_err("partial element");
    assert_eq!(
        error,
        PolynomialError::Config(ConfigError::BufferLength {
            context: "bivariate packed row",
            expected: 4,
            actual: 2,
        })
    );

    // The row table names itself when its growth cannot be reserved.
    let mut growing = BivariatePolynomial::<Gf8B>::zero();
    let error = with_threshold(0, || {
        growing.assign_y_coefficients_packed(packed.iter().copied())
    })
    .expect_err("closed gate");
    assert_eq!(
        error,
        PolynomialError::Config(ConfigError::AllocationFailed {
            context: "bivariate Y coefficients",
            elements: 2,
            element_size: std::mem::size_of::<Polynomial<Gf8B>>(),
        })
    );
    // The unused GF(31) row keeps the odd-characteristic instantiation.
    assert!(truncated.is_zero());
    let _ = m31(1);
    let _ = <Mersenne31 as Field>::BYTES;
}

/// Affine substitution of a constant prefix composes rows exactly, and the
/// per-row truncated product names its reservation failure once the
/// prefix-power and output tables fit.
#[test]
fn affine_substitution_composes_rows_and_reports_product_failure() {
    let low = row(&(0..200)
        .map(|index| (index % 251) as u8)
        .collect::<Vec<_>>());
    let high = row(&(0..200)
        .map(|index| (index % 241) as u8)
        .collect::<Vec<_>>());
    let bivariate =
        BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![low.clone(), high.clone()]);
    let prefix = Polynomial::<Gf8B>::constant(b(5)).expect("prefix");

    let substituted = bivariate
        .substitute_y_affine_truncated(&prefix, 4, 200)
        .expect("substitution");
    // (r0 + 5·r1) in row zero and X^4·r1 truncated in row one.
    let expected_low = low.add(&high.scaled(b(5))).expect("sum");
    let x_power = row(&[0, 0, 0, 0, 1]);
    let mut expected_high = high.multiply(&x_power).expect("shifted row");
    expected_high.truncate(200);
    assert_eq!(
        substituted.y_coefficients()[0]
            .coefficients()
            .collect::<Vec<_>>(),
        expected_low.coefficients().collect::<Vec<_>>()
    );
    assert_eq!(
        substituted.y_coefficients()[1]
            .coefficients()
            .collect::<Vec<_>>(),
        expected_high.coefficients().collect::<Vec<_>>()
    );

    // Prefix powers of a constant prefix are single coefficients and the
    // row/output tables are polynomial structs, so the first reservation
    // above the two-row table sizes is the per-row truncated product.
    let error = with_threshold(48, || {
        bivariate.substitute_y_affine_truncated(&prefix, 4, 200)
    })
    .expect_err("row product");
    assert_eq!(
        error,
        PolynomialError::Config(ConfigError::AllocationFailed {
            context: "polynomial coefficients",
            elements: 200,
            element_size: 1,
        })
    );

    // The zero bivariate substitutes to itself, and an empty precision
    // yields the zero polynomial as well.
    let zero = BivariatePolynomial::<Gf8B>::zero();
    assert!(
        zero.substitute_y_affine_truncated(&prefix, 4, 200)
            .expect("zero substitution")
            .is_zero()
    );
    assert!(
        bivariate
            .substitute_y_affine_truncated(&prefix, 4, 0)
            .expect("empty precision")
            .is_zero()
    );
}

/// The fast affine substitution agrees row for row with the portable one,
/// and its batched product reports a reservation failure once the
/// prefix-power, pair, and metadata tables fit.
#[cfg(feature = "fft")]
#[test]
fn fast_affine_substitution_matches_portable_and_reports_batch_failure() {
    use poly_ring::{PolynomialProductScratch, ProductError};
    let low = row(&(0..200)
        .map(|index| (index % 251) as u8)
        .collect::<Vec<_>>());
    let high = row(&(0..200)
        .map(|index| (index % 241) as u8)
        .collect::<Vec<_>>());
    let bivariate =
        BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![low.clone(), high.clone()]);
    let prefix = Polynomial::<Gf8B>::constant(b(5)).expect("prefix");
    let mut scratch = PolynomialProductScratch::new();

    let fast = bivariate
        .substitute_y_affine_truncated_fast(&prefix, 4, 200, &mut scratch)
        .expect("fast substitution");
    let portable = bivariate
        .substitute_y_affine_truncated(&prefix, 4, 200)
        .expect("portable substitution");
    for (fast_row, portable_row) in fast.y_coefficients().iter().zip(portable.y_coefficients()) {
        assert_eq!(
            fast_row.coefficients().collect::<Vec<_>>(),
            portable_row.coefficients().collect::<Vec<_>>()
        );
    }

    // The pair and metadata tables of two weighted rows fit below a
    // ninety-seven-byte gate; the batched product's two-hundred-byte
    // coefficient buffers do not.
    let error = with_threshold(96, || {
        bivariate.substitute_y_affine_truncated_fast(&prefix, 4, 200, &mut scratch)
    })
    .expect_err("batched product");
    assert_eq!(
        error,
        ProductError::Config(ConfigError::AllocationFailed {
            context: "polynomial coefficients",
            elements: 200,
            element_size: 1,
        })
    );

    // The warmed scratch still substitutes correctly once the gate opens.
    let recovered = bivariate
        .substitute_y_affine_truncated_fast(&prefix, 4, 200, &mut scratch)
        .expect("recovered substitution");
    assert_eq!(
        recovered.y_coefficients()[0]
            .coefficients()
            .collect::<Vec<_>>(),
        portable.y_coefficients()[0]
            .coefficients()
            .collect::<Vec<_>>()
    );
}
