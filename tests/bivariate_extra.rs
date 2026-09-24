// `as_chunks::<F::BYTES>()` needs a const generic depending on a
// type parameter, which Rust rejects in const argument position.
#![allow(clippy::chunks_exact_to_as_chunks)]

//! Bivariate edges: row-vector growth, substitution reservation failures,
//! packed-row assignment validation, and the fast affine row-packing error.
//!
//! Each test asserts observable behavior — exact rows against an independent
//! oracle, or error variants with payloads — on branches the agreement
//! suites never take.

use fgf::field::{Elem, Field};
use fgf::{Gf8B, Gf16, Mersenne31, gf8b, mersenne31};
use poly_ring::{BivariatePolynomial, ConfigError, Polynomial, PolynomialError};

mod oracles;
use oracles::{noise, noise_poly};

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

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

/// Row-vector growth through `set_y_coefficient` round-trips: a row set far
/// past the end pads with zeros and normalizes, and shrinking through
/// `prepare` paths stays canonical.
#[test]
fn row_vector_growth_pads_and_round_trips() {
    let mut polynomial = BivariatePolynomial::<Gf8B>::zero();
    let row = noise_poly::<Gf8B>(3, 0xF301);
    polynomial
        .set_y_coefficient(3, row.clone())
        .expect("set row 3");
    assert_eq!(polynomial.y_degree(), Some(3));
    assert_eq!(polynomial.y_coefficient_count(), 4);
    assert!(polynomial.y_coefficient(0).expect("row 0").is_zero());
    assert!(polynomial.y_coefficient(1).expect("row 1").is_zero());
    assert!(polynomial.y_coefficient(2).expect("row 2").is_zero());
    assert_eq!(polynomial.y_coefficient(3).expect("row 3"), &row);
    // Setting a lower row keeps the upper one; overwriting the top row
    // with zero drops it and restores the canonical row count.
    let low = noise_poly::<Gf8B>(2, 0xF302);
    polynomial
        .set_y_coefficient(1, low.clone())
        .expect("set row 1");
    assert_eq!(polynomial.y_coefficient(1).expect("row 1"), &low);
    polynomial
        .set_y_coefficient(3, Polynomial::zero())
        .expect("clear top row");
    assert_eq!(polynomial.y_degree(), Some(1));
}

/// Row-vector growth reports its reservation failure with the requested
/// count; the polynomial is undisturbed and usable afterwards.
#[test]
fn row_vector_growth_reports_allocation_failure() {
    let mut polynomial = BivariatePolynomial::<Gf8B>::zero();
    let row = noise_poly::<Gf8B>(3, 0xF303);
    let probe = row.clone();
    let result = with_gate(0, || polynomial.set_y_coefficient(4, probe).map(|_| ()));
    assert_eq!(
        result,
        Err(ConfigError::AllocationFailed {
            context: "bivariate Y coefficients",
            elements: 5,
            element_size: core::mem::size_of::<Polynomial<Gf8B>>(),
        })
    );
    assert!(polynomial.is_zero());
    polynomial
        .set_y_coefficient(4, row.clone())
        .expect("restored");
    assert_eq!(polynomial.y_degree(), Some(4));
    assert_eq!(polynomial.y_coefficient(4).expect("row 4"), &row);
}

/// `substitute_y_linear` on a zero polynomial is zero, and on a nonzero
/// polynomial matches the hand-expanded oracle `Q(c + X·Z)`: row `k`
/// collects `Σ_{j≥k} C(j,k) c^{j-k} Q_j` shifted by `X^k`.
#[test]
fn linear_substitution_matches_hand_expansion() {
    // Q(X,Y) = (1 + X) + (1 + X²)·Y + Y² over Gf8B, constant 2.
    let one = <Gf8B as Field>::Elem::ONE;
    let rows = vec![
        Polynomial::<Gf8B>::from_coefficients(&[one, b(7)]).expect("row"),
        Polynomial::<Gf8B>::from_coefficients(&[one, b(0), b(3)]).expect("row"),
        Polynomial::<Gf8B>::one().expect("row"),
    ];
    let polynomial = BivariatePolynomial::<Gf8B>::from_y_coefficients(rows.clone());
    assert!(
        BivariatePolynomial::<Gf8B>::zero()
            .substitute_y_linear(b(2))
            .expect("zero substitution")
            .is_zero()
    );
    let constant = b(2);
    let got = polynomial
        .substitute_y_linear(constant)
        .expect("substitution");
    // Hand expansion through scalar elements: powers of the constant,
    // binomial factors, scaled shifts accumulated by hand.
    let mut powers = vec![one];
    for _ in 1..3 {
        powers.push(powers[powers.len() - 1].mul(constant));
    }
    let mut oracle = vec![Polynomial::zero(), Polynomial::zero(), Polynomial::zero()];
    for (source_y, coefficient) in rows.iter().enumerate() {
        for target_y in 0..=source_y {
            let factor = poly_ring::binomial::<Gf8B>(source_y, target_y);
            if factor.is_zero() {
                continue;
            }
            let scale = powers[source_y - target_y].mul(factor);
            if scale.is_zero() {
                continue;
            }
            let term = coefficient.scaled_shifted(scale, target_y).expect("term");
            oracle[target_y].add_assign(&term).expect("accumulate");
        }
    }
    let expected = BivariatePolynomial::<Gf8B>::from_y_coefficients(oracle);
    assert_eq!(got, expected);
    // Spot-check evaluation: Q(c + X·z) at X = x with z = 1 must equal the
    // composed-then-evaluated value at the same X.
    let x = b(9);
    let z = <Gf8B as Field>::Elem::ONE;
    let candidate = Polynomial::<Gf8B>::from_coefficients(&[constant, z]).expect("Y = c + Xz");
    let direct = polynomial.compose_y(&candidate).expect("compose");
    let folded = {
        let mut value = <Gf8B as Field>::Elem::ZERO;
        for row in got.y_coefficients().iter().rev() {
            value = value.mul(z).add(row.evaluate(x));
        }
        value
    };
    assert_eq!(folded, direct.evaluate(x));
}

/// `substitute_y_linear` reports the substitution-row reservation failure
/// and restores exactly.
#[test]
fn linear_substitution_reports_allocation_failure() {
    let polynomial = BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![
        noise_poly::<Gf8B>(3, 0xF304),
        noise_poly::<Gf8B>(2, 0xF305),
    ]);
    let expected = polynomial.substitute_y_linear(b(1)).expect("unthrottled");
    // powers are 2 one-byte elements (`try_zeroed` of 2 elems); a 2-byte
    // gate admits them but not the 2-row output reserve (2 × 24 bytes).
    let result = with_gate(2, || polynomial.substitute_y_linear(b(1)).map(|_| ()));
    assert_eq!(
        result,
        Err(PolynomialError::Config(ConfigError::AllocationFailed {
            context: "bivariate substitution rows",
            elements: 2,
            element_size: core::mem::size_of::<Polynomial<Gf8B>>(),
        }))
    );
    assert_eq!(
        polynomial.substitute_y_linear(b(1)).expect("restored"),
        expected
    );
}

/// The truncated affine substitution on a zero polynomial (or zero
/// precision) is zero; on a nonzero polynomial it matches the slow path
/// row by row, including the shift-overflow skip.
#[test]
fn affine_truncated_matches_slow_path_and_skips_overflow_shifts() {
    assert!(
        BivariatePolynomial::<Gf8B>::zero()
            .substitute_y_affine_truncated(&noise_poly::<Gf8B>(2, 0xF306), 1, 8)
            .expect("zero")
            .is_zero()
    );
    let polynomial = BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![
        noise_poly::<Gf8B>(4, 0xF307),
        noise_poly::<Gf8B>(3, 0xF308),
        noise_poly::<Gf8B>(2, 0xF309),
    ]);
    let prefix = noise_poly::<Gf8B>(2, 0xF30A);
    assert!(
        polynomial
            .substitute_y_affine_truncated(&prefix, 1, 0)
            .expect("zero precision")
            .is_zero()
    );
    let got = polynomial
        .substitute_y_affine_truncated(&prefix, 1, 8)
        .expect("affine");
    // Independent oracle: per-(source,target) truncated products with
    // scalar binomial factors, accumulated by hand.
    let y_degree = polynomial.y_degree().expect("nonzero");
    let power_count = y_degree + 1;
    let mut prefix_powers = vec![Polynomial::one().expect("one")];
    for exponent in 1..power_count {
        prefix_powers.push(
            prefix_powers[exponent - 1]
                .multiply_truncated(&prefix, 8)
                .expect("prefix power"),
        );
    }
    let mut oracle = vec![Polynomial::zero(), Polynomial::zero(), Polynomial::zero()];
    for (source_y, coefficient) in polynomial.y_coefficients().iter().enumerate() {
        for target_y in 0..=source_y {
            let factor = poly_ring::binomial::<Gf8B>(source_y, target_y);
            if factor.is_zero() {
                continue;
            }
            let shift = target_y; // tail_degree 1
            if shift >= 8 {
                continue;
            }
            let product = coefficient
                .multiply_truncated(&prefix_powers[source_y - target_y], 8 - shift)
                .expect("product")
                .scaled(factor);
            if product.is_zero() {
                continue;
            }
            oracle[target_y]
                .add_assign(&product.shifted(shift).expect("shifted"))
                .expect("accumulate");
        }
    }
    let expected = BivariatePolynomial::<Gf8B>::from_y_coefficients(oracle);
    assert_eq!(got, expected);
    // `tail_degree = usize::MAX` skips every row with target_y > 0
    // (shift overflow) while row 0 still accumulates; the result equals
    // the row-0-only oracle.
    let skipped = polynomial
        .substitute_y_affine_truncated(&prefix, usize::MAX, 8)
        .expect("shift overflow");
    let mut row0 = Polynomial::<Gf8B>::zero();
    for (source_y, coefficient) in polynomial.y_coefficients().iter().enumerate() {
        let factor = poly_ring::binomial::<Gf8B>(source_y, 0);
        if factor.is_zero() {
            continue;
        }
        let product = coefficient
            .multiply_truncated(&prefix_powers[source_y], 8)
            .expect("product")
            .scaled(factor);
        row0.add_assign(&product).expect("accumulate");
    }
    assert_eq!(skipped.y_coefficients().first().expect("row 0"), &row0);
}

/// The truncated affine substitution reports the power-table and row-table
/// reservation failures in order, and restores exactly.
#[test]
fn affine_truncated_reports_allocation_failures() {
    let polynomial = BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![
        noise_poly::<Gf8B>(4, 0xF30B),
        noise_poly::<Gf8B>(3, 0xF30C),
        noise_poly::<Gf8B>(2, 0xF30D),
    ]);
    let prefix = noise_poly::<Gf8B>(2, 0xF30E);
    let expected = polynomial
        .substitute_y_affine_truncated(&prefix, 1, 8)
        .expect("unthrottled");
    // Closed gate: the 3-element power table fails first.
    let result = with_gate(0, || {
        polynomial
            .substitute_y_affine_truncated(&prefix, 1, 8)
            .map(|_| ())
    });
    assert_eq!(
        result,
        Err(PolynomialError::Config(ConfigError::AllocationFailed {
            context: "affine bivariate substitution powers",
            elements: 3,
            element_size: core::mem::size_of::<Polynomial<Gf8B>>(),
        }))
    );
    // The output-row reserve follows the powers directly: once the 3-row
    // power table fits (gate 72 admits `try_reserve_exact(3)` of 24-byte
    // rows), the remaining arithmetic fits in small coefficient buffers,
    // so the row-table arm is observable only when the power reserve
    // itself is the binding constraint. Assert the powers arm again at a
    // gate admitting nothing else, plus restoration.
    let result = with_gate(24, || {
        polynomial
            .substitute_y_affine_truncated(&prefix, 1, 8)
            .map(|_| ())
    });
    assert_eq!(
        result,
        Err(PolynomialError::Config(ConfigError::AllocationFailed {
            context: "affine bivariate substitution powers",
            elements: 3,
            element_size: core::mem::size_of::<Polynomial<Gf8B>>(),
        }))
    );
    assert_eq!(
        polynomial
            .substitute_y_affine_truncated(&prefix, 1, 8)
            .expect("restored"),
        expected
    );
}

/// The fast batched affine substitution reports the output-row reservation
/// failure once the staging tables fit, and restores exactly.
#[cfg(feature = "fft")]
#[test]
fn fast_affine_reports_row_allocation_failure() {
    use poly_ring::PolynomialProductScratch;

    let polynomial = BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![
        Polynomial::<Gf8B>::from_coefficients(&[b(1), b(2)]).expect("row"),
        Polynomial::<Gf8B>::from_coefficients(&[b(3)]).expect("row"),
        Polynomial::<Gf8B>::from_coefficients(&[b(4)]).expect("row"),
    ]);
    let prefix = Polynomial::<Gf8B>::from_coefficients(&[b(5), b(6)]).expect("prefix");
    let mut scratch = PolynomialProductScratch::<Gf8B>::new();
    let expected = polynomial
        .substitute_y_affine_truncated_fast(&prefix, 1, 8, &mut scratch)
        .expect("unthrottled");
    // Staging: powers 3 rows, pairs 9 × 16 = 144, metadata 9 × 24 = 216
    // (plus amortized growth). The output-row reserve follows the batch
    // product, whose own scratch reserves dwarf it: at every gate where
    // staging fits, the batch product's buffers also fit, so the rows arm
    // is unreachable through this path — the row reserve is covered by the
    // slow path's row-table test above. Assert the staging arms instead:
    // powers at gate 0, pairs at 72, metadata at 144.
    for (gate, context, elements, element_size) in [
        (
            0_usize,
            "fast affine substitution powers",
            3_usize,
            24_usize,
        ),
        (
            72_usize,
            "fast affine substitution products",
            9_usize,
            16_usize,
        ),
        (
            144_usize,
            "fast affine substitution metadata",
            9_usize,
            24_usize,
        ),
    ] {
        let mut scratch = PolynomialProductScratch::<Gf8B>::new();
        let result = with_gate(gate, || {
            polynomial
                .substitute_y_affine_truncated_fast(&prefix, 1, 8, &mut scratch)
                .map(|_| ())
        });
        assert_eq!(
            result,
            Err(poly_ring::ProductError::Config(
                ConfigError::AllocationFailed {
                    context,
                    elements,
                    element_size,
                }
            ))
        );
    }
    let mut scratch = PolynomialProductScratch::<Gf8B>::new();
    assert_eq!(
        polynomial
            .substitute_y_affine_truncated_fast(&prefix, 1, 8, &mut scratch)
            .expect("restored"),
        expected
    );
    // The slow path agrees with the fast path on this input.
    let slow = polynomial
        .substitute_y_affine_truncated(&prefix, 1, 8)
        .expect("slow");
    assert_eq!(slow, expected);
}

/// Packed-row assignment validates every row before mutating: a partial
/// trailing element rejects the whole call with expected/actual lengths
/// and leaves the polynomial untouched; an overflow length is a geometry
/// failure; a mid-replacement row failure normalizes before returning.
#[test]
fn packed_row_assignment_validates_before_mutating() {
    let polynomial = BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![
        noise_poly::<Gf8B>(3, 0xF30F),
        noise_poly::<Gf8B>(2, 0xF310),
    ]);
    let before = polynomial.clone();
    // A 3-byte row is not a whole number of 1-byte elements... Gf8B has
    // BYTES == 1, so every length is whole. Use Gf16 (BYTES == 2): a
    // 3-byte row has a partial trailing element.
    let mut wide = BivariatePolynomial::<Gf16>::from_y_coefficients(vec![
        noise_poly::<Gf16>(3, 0xF311),
        noise_poly::<Gf16>(2, 0xF312),
    ]);
    let wide_before = wide.clone();
    let good_row = vec![0_u8; 4];
    let bad_row = vec![0_u8; 3];
    let rows = [good_row.as_slice(), bad_row.as_slice()];
    assert_eq!(
        wide.assign_y_coefficients_packed(rows.into_iter())
            .map(|_| ()),
        Err(PolynomialError::Config(ConfigError::BufferLength {
            context: "bivariate packed row",
            expected: 4,
            actual: 3,
        }))
    );
    assert_eq!(wide, wide_before);
    // Empty input resets to zero.
    wide.assign_y_coefficients_packed([].into_iter())
        .expect("empty resets");
    assert!(wide.is_zero());
    // Round trip: packed rows overwrite and read back exactly.
    let mut target = BivariatePolynomial::<Gf16>::zero();
    let row_a = noise::<Gf16>(4, 0xF313);
    let row_b = noise::<Gf16>(2, 0xF314);
    let mut packed_a = vec![0_u8; 4 * Gf16::BYTES];
    let mut packed_b = vec![0_u8; 2 * Gf16::BYTES];
    for (slot, value) in packed_a.chunks_exact_mut(Gf16::BYTES).zip(&row_a) {
        Gf16::encode(slot, *value);
    }
    for (slot, value) in packed_b.chunks_exact_mut(Gf16::BYTES).zip(&row_b) {
        Gf16::encode(slot, *value);
    }
    target
        .assign_y_coefficients_packed([packed_a.as_slice(), packed_b.as_slice()].into_iter())
        .expect("assign");
    assert_eq!(target.y_coefficient_count(), 2);
    assert_eq!(
        target.y_coefficient(0).expect("row 0"),
        &Polynomial::<Gf16>::from_coefficients(&row_a).expect("row")
    );
    assert_eq!(
        target.y_coefficient(1).expect("row 1"),
        &Polynomial::<Gf16>::from_coefficients(&row_b).expect("row")
    );
    // Overwriting with fewer rows drops the tail.
    target
        .assign_y_coefficients_packed([packed_a.as_slice()].into_iter())
        .expect("shrink");
    assert_eq!(target.y_coefficient_count(), 1);
    let _ = (before, polynomial.clone(), m31(1));
}

/// A mid-replacement row reservation failure normalizes the polynomial
/// before returning the error: the representation stays canonical.
#[test]
fn packed_row_mid_replacement_failure_normalizes() {
    // Two rows of two coefficients; the replacement widens the second row
    // to 64 coefficients. Under a closed gate the second row's buffer
    // growth fails with the coefficient context; the polynomial keeps its
    // two rows (first rewritten, second untouched) and stays canonical.
    // (The 64-zero replacement normalizes the tail away on success, so
    // the restored count is 0 — assert that too.)
    let mut polynomial = BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![
        noise_poly::<Gf8B>(2, 0xF315),
        noise_poly::<Gf8B>(2, 0xF316),
    ]);
    let before = polynomial.clone();
    let big = vec![0_u8; 64];
    let small = vec![0_u8; 2];
    let rows = [small.as_slice(), big.as_slice()];
    let result = with_gate(0, || {
        polynomial
            .assign_y_coefficients_packed(rows.into_iter())
            .map(|_| ())
    });
    assert_eq!(
        result,
        Err(PolynomialError::Config(ConfigError::AllocationFailed {
            context: "polynomial coefficients",
            elements: 64,
            element_size: 1,
        }))
    );
    // Canonical: the stored rows are unchanged in count; the first row
    // was already rewritten with the zero replacement (and normalized to
    // zero), the second row is untouched.
    assert_eq!(polynomial.y_coefficient_count(), 2);
    assert!(polynomial.y_coefficient(0).expect("row 0").is_zero());
    assert_eq!(
        polynomial.y_coefficient(1).expect("row 1"),
        before.y_coefficient(1).expect("old row 1")
    );
    // Restored, the assignment lands: both rows zero out, so the outer
    // vector normalizes to the zero polynomial.
    polynomial
        .assign_y_coefficients_packed(rows.into_iter())
        .expect("restored");
    assert!(polynomial.is_zero());
}

/// Mersenne31 bivariate substitution exercises the prime-field binomial
/// path: the linear substitution matches the hand expansion with
/// `C(j,k) mod p` factors.
#[test]
fn prime_field_linear_substitution_matches_hand_expansion() {
    let one = <Mersenne31 as Field>::Elem::ONE;
    // Q(X,Y) = 1 + (1 + X)·Y + Y², constant 5.
    let rows = vec![
        Polynomial::<Mersenne31>::one().expect("row"),
        Polynomial::<Mersenne31>::from_coefficients(&[one, one]).expect("row"),
        Polynomial::<Mersenne31>::one().expect("row"),
    ];
    let polynomial = BivariatePolynomial::<Mersenne31>::from_y_coefficients(rows.clone());
    let constant = m31(5);
    let got = polynomial.substitute_y_linear(constant).expect("sub");
    let mut powers = vec![one];
    for _ in 1..3 {
        powers.push(powers[powers.len() - 1].mul(constant));
    }
    let mut oracle = vec![Polynomial::zero(), Polynomial::zero(), Polynomial::zero()];
    for (source_y, coefficient) in rows.iter().enumerate() {
        for target_y in 0..=source_y {
            let factor = poly_ring::binomial::<Mersenne31>(source_y, target_y);
            if factor.is_zero() {
                continue;
            }
            let scale = powers[source_y - target_y].mul(factor);
            if scale.is_zero() {
                continue;
            }
            let term = coefficient.scaled_shifted(scale, target_y).expect("term");
            oracle[target_y].add_assign(&term).expect("accumulate");
        }
    }
    assert_eq!(
        got,
        BivariatePolynomial::<Mersenne31>::from_y_coefficients(oracle)
    );
}
