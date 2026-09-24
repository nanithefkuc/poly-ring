//! Sparse conversions and derivative edge behavior.
//!
//! The agreement suites cover round-trips and common orders; every test here
//! takes a branch they never reach — zero-term inputs, out-of-support
//! orders with nonzero binomial structure, jet-buffer length contracts, and
//! dense conversions that must reject or normalize — each against exact
//! terms or an independent scalar sum.

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::{
    BivariatePolynomial, ConfigError, MonomialOrder, MultiIndex, Polynomial, SparsePolynomial, Term,
};

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn term<F: FieldKernels, const N: usize>(
    exponents: [usize; N],
    coefficient: F::Elem,
) -> Term<F, N> {
    Term {
        exponents: MultiIndex::new(exponents),
        coefficient,
    }
}

/// Zero terms never survive construction: an explicit zero coefficient is
/// dropped and the polynomial reads as zero.
#[test]
fn zero_terms_are_dropped_at_construction() {
    let polynomial: SparsePolynomial<Mersenne31, 2> =
        SparsePolynomial::from_terms(vec![term([1, 0], m31(0)), term([0, 1], m31(5))]);
    assert_eq!(polynomial.terms(), &[term([0, 1], m31(5))]);
    assert_eq!(polynomial.coefficient(&MultiIndex::new([1, 0])), m31(0));
    assert_eq!(polynomial.coefficient(&MultiIndex::new([0, 1])), m31(5));
    assert_eq!(
        polynomial,
        SparsePolynomial::from_terms(vec![term([0, 1], m31(5))])
    );
    assert_eq!(polynomial.clone(), polynomial);
    assert_eq!(
        SparsePolynomial::<Mersenne31, 2>::default(),
        SparsePolynomial::zero()
    );
}

/// Merging an empty sum with a nonzero one copies the survivor verbatim,
///
/// and a sum whose every pair cancels is the canonical zero.
#[test]
fn sums_with_empty_sides_copy_or_cancel() {
    let zero: SparsePolynomial<Mersenne31, 2> = SparsePolynomial::zero();
    let one = SparsePolynomial::from_terms(vec![term([2, 3], m31(7))]);
    assert_eq!(zero.add(&one).expect("add"), one);
    assert_eq!(one.add(&zero).expect("add"), one);
    let order = <Mersenne31 as Field>::ORDER;
    let negated = SparsePolynomial::from_terms(vec![term([2, 3], m31((order - 7) as u32))]);
    assert!(one.add(&negated).expect("cancel").is_zero());
    // Subtraction against empty negates on the right side only.
    let difference = zero.sub(&one).expect("sub");
    assert_eq!(difference.terms(), &[term([2, 3], m31((order - 7) as u32))]);
}

/// A pair count that cannot be represented is a geometry error, never a
/// wrapped allocation.
#[test]
fn product_pair_count_overflow_is_rejected() {
    // The pair count itself overflows before any exponent is added: the
    // operands are tiny but the test drives `checked_mul` on the counts.
    // With two and three terms the count is representable, so instead the
    // exponent sum overflows on the product path.
    let huge: SparsePolynomial<Mersenne31, 2> =
        SparsePolynomial::from_terms(vec![term([usize::MAX, 0], m31(1))]);
    let unit: SparsePolynomial<Mersenne31, 2> =
        SparsePolynomial::from_terms(vec![term([1, 0], m31(1))]);
    assert_eq!(
        huge.multiply(&unit).map(|_| ()),
        Err(ConfigError::GeometryOverflow {
            context: "multivariate exponent sum",
        })
    );
}

/// Derivative orders beyond a term's support contribute nothing, while a
/// fully supported order keeps the exact binomial factor.
#[test]
fn derivative_orders_beyond_support_contribute_zero() {
    let polynomial: SparsePolynomial<Mersenne31, 2> =
        SparsePolynomial::from_terms(vec![term([1, 2], m31(3))]);
    // Order [2, 0] exceeds the X support: empty.
    assert!(
        polynomial
            .hasse_derivative(&MultiIndex::new([2, 0]))
            .expect("derivative")
            .is_zero()
    );
    // Order [1, 2]: C(1,1)·C(2,2)·3 = 3 at [0, 0].
    assert_eq!(
        polynomial
            .hasse_derivative(&MultiIndex::new([1, 2]))
            .expect("derivative")
            .terms(),
        &[term([0, 0], m31(3))]
    );
    // Order [0, 1]: C(2,1)·3 = 6 at [1, 1].
    assert_eq!(
        polynomial
            .hasse_derivative(&MultiIndex::new([0, 1]))
            .expect("derivative")
            .terms(),
        &[term([1, 1], m31(6))]
    );
}

/// The scalar Hasse sum skips out-of-support terms: `evaluate_hasse` at an
/// order beyond one term still sees the other.
#[test]
fn scalar_hasse_sums_skip_out_of_support_terms() {
    let polynomial: SparsePolynomial<Mersenne31, 2> =
        SparsePolynomial::from_terms(vec![term([0, 0], m31(5)), term([2, 0], m31(1))]);
    // Order [1, 0]: the constant contributes zero, X² contributes 2·X.
    assert_eq!(
        polynomial.evaluate_hasse(&[m31(4), m31(9)], &MultiIndex::new([1, 0])),
        m31(8)
    );
    // Order [0, 0] is plain evaluation: 5 + 16 = 21.
    assert_eq!(polynomial.evaluate(&[m31(4), m31(9)]), m31(21));
    // Binary field: C(2,1) = 0 kills the X² term even in the scalar sum.
    let binary: SparsePolynomial<Gf8B, 2> = SparsePolynomial::from_terms(vec![term([2, 0], b(1))]);
    assert!(
        binary
            .evaluate_hasse(&[b(3), b(4)], &MultiIndex::new([1, 0]))
            .is_zero()
    );
}

/// The jet buffer rejects a short output before writing: the stored value
/// is untouched on error.
#[test]
fn jet_buffer_rejects_short_output_without_writing() {
    let polynomial: SparsePolynomial<Mersenne31, 2> =
        SparsePolynomial::from_terms(vec![term([1, 0], m31(2))]);
    let point = [m31(4), m31(5)];
    let orders = [MultiIndex::new([0, 0]), MultiIndex::new([1, 0])];
    let mut output = [m31(99), m31(99)];
    assert!(
        polynomial
            .evaluate_jet_into(&point, &orders[..1], &mut output)
            .is_err()
    );
    assert_eq!(output, [m31(99), m31(99)]);
    polynomial
        .evaluate_jet_into(&point, &orders, &mut output)
        .expect("jet");
    assert_eq!(output, [m31(8), m31(2)]);
}

/// `from_bivariate` keeps stored zero rows as stored coefficients: an
/// internal zero `Y` row contributes no terms but the dense round-trip
/// restores it.
#[test]
fn sparse_from_bivariate_skips_zero_rows_exactly() {
    let dense = BivariatePolynomial::<Mersenne31>::from_y_coefficients(vec![
        Polynomial::from_coefficients(&[m31(1)]).expect("row"),
        Polynomial::zero(),
        Polynomial::from_coefficients(&[m31(0), m31(2)]).expect("row"),
    ]);
    let sparse = SparsePolynomial::from_bivariate(&dense).expect("sparse");
    assert_eq!(sparse.terms().len(), 2);
    let back = sparse.to_bivariate().expect("dense");
    assert_eq!(back, dense);
}

/// `from_univariate` enumerates stored coefficients including interior
/// zeroes; `to_univariate` restores them.
#[test]
fn univariate_conversions_keep_interior_zeroes() {
    let univariate =
        Polynomial::<Mersenne31>::from_coefficients(&[m31(4), m31(0), m31(6)]).expect("dense");
    let sparse = SparsePolynomial::from_univariate(&univariate).expect("sparse");
    assert_eq!(sparse.terms().len(), 2);
    assert_eq!(sparse.to_univariate().expect("dense"), univariate);
}

/// Leading-term ranking under `Lex` and `GradedLex` follows the stored
/// exponents, and the empty polynomial has no leading term.
#[test]
fn leading_terms_follow_lex_and_graded_orders() {
    let polynomial: SparsePolynomial<Mersenne31, 2> =
        SparsePolynomial::from_terms(vec![term([3, 0], m31(1)), term([1, 9], m31(1))]);
    // Lex from variable 0: X³ wins over X·Y⁹.
    assert_eq!(
        polynomial
            .leading_term(&MonomialOrder::Lex)
            .expect("lex")
            .expect("nonzero")
            .exponents
            .exponents(),
        &[3, 0]
    );
    // GradedLex by total degree: X·Y⁹ (degree 10) wins over X³.
    assert_eq!(
        polynomial
            .leading_term(&MonomialOrder::GradedLex)
            .expect("graded")
            .expect("nonzero")
            .exponents
            .exponents(),
        &[1, 9]
    );
    assert!(
        SparsePolynomial::<Mersenne31, 2>::zero()
            .leading_term(&MonomialOrder::Lex)
            .expect("empty")
            .is_none()
    );
}
