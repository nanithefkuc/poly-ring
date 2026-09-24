//! Dense bivariate edges: row growth, geometry accessors, the linear
//! and affine substitutions at degenerate inputs, and truncation and
//! valuation.
//!
//! The agreement suite in `bivariate.rs` covers the common paths; every
//! test here takes a branch it never reaches, each against exact rows or
//! an independent evaluation check.

use fgf::field::Field;
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::{BivariatePolynomial, ConfigError, Polynomial, PolynomialError};

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

/// Setting a row beyond the stored count grows the vector with zero rows,
#[test]
fn row_assignment_grows_and_rejects_overflow() {
    let mut q = rows::<Mersenne31>(&[&[m31(1)]]);
    q.set_y_coefficient(3, Polynomial::from_coefficients(&[m31(7)]).expect("row"))
        .expect("grow");
    assert_eq!(q.y_coefficient_count(), 4);
    assert!(q.y_coefficient(1).expect("row").is_zero());
    assert!(q.y_coefficient(2).expect("row").is_zero());
    assert_eq!(q.y_coefficient(3).expect("row").coefficient(0), m31(7));

    assert_eq!(
        q.set_y_coefficient(usize::MAX, Polynomial::zero())
            .map(|_| ()),
        Err(ConfigError::GeometryOverflow {
            context: "bivariate Y coefficient count",
        })
    );
    // A representable but unreservable row count fails without allocating:
    // the reservation for `2^62` rows cannot exist on a 64-bit host.
    assert!(matches!(
        q.set_y_coefficient((1 << 62) + 4, Polynomial::zero()),
        Err(ConfigError::AllocationFailed { .. })
    ));
    // The failed growth left the polynomial canonical.
    assert_eq!(q.y_coefficient_count(), 4);
}

/// The weighted degree mirrors the leading term, and the zero polynomial
/// has neither.
#[test]
fn weighted_degree_mirrors_the_leading_term() {
    let q = rows::<Mersenne31>(&[&[m31(1), m31(0), m31(1)], &[m31(1)]]);
    assert_eq!(q.weighted_degree(2).expect("degree"), Some(2));
    assert_eq!(q.weighted_degree(4).expect("degree"), Some(4));
    assert_eq!(
        BivariatePolynomial::<Mersenne31>::zero()
            .weighted_degree(2)
            .expect("empty"),
        None
    );
    assert_eq!(BivariatePolynomial::<Mersenne31>::zero().y_degree(), None);
    assert_eq!(
        BivariatePolynomial::<Mersenne31>::default().y_coefficient_count(),
        0
    );
}

/// A `Y` order at or above the row count differentiates to zero without
/// reading any row.
#[test]
fn hasse_discrepancy_beyond_the_rows_is_zero() {
    let q = rows::<Mersenne31>(&[&[m31(1), m31(2)]]);
    assert_eq!(q.hasse_discrepancy(m31(3), m31(5), 0, 5), m31(0));
}

/// Products with zero vanish, and every row times `X + constant` scales
/// the evaluation by `x + constant`.
#[test]
fn zero_products_and_linear_row_products_hold() {
    let q = rows::<Gf8B>(&[&[b(1), b(2)], &[b(3)]]);
    assert!(
        q.multiply(&BivariatePolynomial::zero())
            .expect("mul")
            .is_zero()
    );
    assert!(
        BivariatePolynomial::zero()
            .multiply(&q)
            .expect("mul")
            .is_zero()
    );

    // An internal zero row on the right is skipped, not convolved.
    let sparse = rows::<Gf8B>(&[&[b(1)], &[], &[b(1)]]);
    let product = q.multiply(&sparse).expect("product");
    let x = <Gf8B as Field>::GENERATOR;
    let y = x.mul(x);
    assert_eq!(
        product.evaluate(x, y),
        q.evaluate(x, y).mul(sparse.evaluate(x, y))
    );

    let scaled = q.multiply_x_plus(b(5)).expect("linear");
    assert_eq!(scaled.evaluate(x, y), q.evaluate(x, y).mul(x.add(b(5))));
}

/// Substituting into the zero polynomial, or truncating to zero rows, yields zero.
#[test]
fn degenerate_substitutions_yield_zero() {
    let zero = BivariatePolynomial::<Gf8B>::zero();
    assert!(zero.substitute_y_linear(b(1)).expect("linear").is_zero());
    let q = rows::<Gf8B>(&[&[b(1)], &[b(2)]]);
    let prefix = Polynomial::from_coefficients(&[b(1)]).expect("prefix");
    assert!(
        q.substitute_y_affine_truncated(&prefix, 1, 0)
            .expect("truncated")
            .is_zero()
    );
    assert!(
        zero.substitute_y_affine_truncated(&prefix, 1, 4)
            .expect("truncated")
            .is_zero()
    );
}

/// Substituting `Y = 0 + X·Z` into `X²·Y²` drops every vanishing middle
/// term and leaves `X⁴·Z²` in the top row.
#[test]
fn linear_substitution_at_zero_constant_drops_vanishing_terms() {
    let q = rows::<Mersenne31>(&[&[], &[], &[m31(0), m31(0), m31(1)]]);
    let substituted = q.substitute_y_linear(m31(0)).expect("substitution");
    assert_eq!(substituted.y_coefficient_count(), 3);
    assert!(substituted.y_coefficient(0).expect("row").is_zero());
    assert!(substituted.y_coefficient(1).expect("row").is_zero());
    let top = substituted.y_coefficient(2).expect("row");
    assert_eq!(top.degree(), Some(4));
    assert_eq!(top.coefficient(4), m31(1));
}

/// An overflowing tail shift drops its rows: with `tail_degree` at the
/// address-space edge only the unshifted `Y^0` contributions survive.
#[test]
fn affine_substitution_drops_overflowing_shifts() {
    // Q = (1 + X) + Y over M31, prefix 1 + X, four output rows.
    let q = rows::<Mersenne31>(&[&[m31(1), m31(1)], &[m31(1)]]);
    let prefix = Polynomial::from_coefficients(&[m31(1), m31(1)]).expect("prefix");
    let result = q
        .substitute_y_affine_truncated(&prefix, usize::MAX, 4)
        .expect("substitution");
    // Only target row 0 survives: (1 + X) + (1 + X), truncated to 4 rows.
    let expected = Polynomial::from_coefficients(&[m31(2), m31(2)]).expect("expected");
    assert_eq!(result.y_coefficient_count(), 1);
    assert_eq!(result.y_coefficient(0).expect("row"), &expected);
}

/// A row with no surviving low coefficients contributes nothing: with a
/// one-row precision every `X` multiple truncates away and the result is
/// zero.
#[test]
fn affine_substitution_drops_fully_truncated_products() {
    let q = rows::<Mersenne31>(&[&[m31(0), m31(1)], &[m31(0), m31(1)]]);
    let prefix = Polynomial::from_coefficients(&[m31(1)]).expect("prefix");
    let result = q
        .substitute_y_affine_truncated(&prefix, 0, 1)
        .expect("substitution");
    assert!(result.is_zero());
}

/// Truncation drops high `X` coefficients row-wise, and the valuation is
/// the minimum across the nonzero rows.
#[test]
fn truncation_and_valuation_follow_the_rows() {
    let q = rows::<Mersenne31>(&[&[m31(1), m31(2), m31(3)], &[m31(4), m31(5)]]);
    let truncated = q.truncated_x(2);
    assert_eq!(
        truncated.y_coefficient(0).expect("row").coefficient_count(),
        2
    );
    assert_eq!(
        truncated.y_coefficient(1).expect("row").coefficient_count(),
        2
    );
    assert_eq!(q.x_valuation(), Some(0));
    assert_eq!(
        BivariatePolynomial::<Mersenne31>::zero().x_valuation(),
        None
    );

    let shifted = rows::<Mersenne31>(&[&[m31(0), m31(0), m31(1)], &[m31(0), m31(1)]]);
    assert_eq!(shifted.x_valuation(), Some(1));
    let divided = shifted.divide_by_x_power(1).expect("divide");
    assert_eq!(
        divided.y_coefficient(0).expect("row").coefficient(1),
        m31(1)
    );
    assert_eq!(
        shifted.divide_by_x_power(2).map(|_| ()),
        Err(PolynomialError::NonExactDivision)
    );
}
