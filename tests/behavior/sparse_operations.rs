//! Sparse operations behavior and error boundaries.

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::{
    BivariatePolynomial, ConfigError, MonomialOrder, MultiIndex, Polynomial, SparsePolynomial, Term,
};

use crate::oracles;
use oracles::{noise, noise_poly};

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

/// Left-tail and right-tail copies: sums where one side exhausts first keep
/// every remaining term with the right sign.
#[test]
fn merge_tails_copy_with_sign() {
    // Left has an extra high term; right an extra low term.
    let left = SparsePolynomial::<Mersenne31, 2>::from_terms(vec![
        term([5, 0], m31(1)),
        term([9, 9], m31(2)),
    ]);
    let right = SparsePolynomial::<Mersenne31, 2>::from_terms(vec![
        term([0, 0], m31(3)),
        term([5, 0], m31(4)),
    ]);
    let sum = left.add(&right).expect("add");
    assert_eq!(
        sum.terms(),
        &[
            term([0, 0], m31(3)),
            term([5, 0], m31(5)),
            term([9, 9], m31(2)),
        ]
    );
    // Subtraction negates the right tail: [0,0] becomes -3, [9,9] stays.
    let difference = left.sub(&right).expect("sub");
    let order = <Mersenne31 as Field>::ORDER;
    assert_eq!(
        difference.terms(),
        &[
            term([0, 0], m31((order - 3) as u32)),
            term([5, 0], m31((order - 3) as u32)),
            term([9, 9], m31(2)),
        ]
    );
    // Right-only tail under subtraction is negated element by element.
    let nearly_empty = SparsePolynomial::<Mersenne31, 2>::from_terms(vec![term([1, 1], m31(1))]);
    let tail = SparsePolynomial::<Mersenne31, 2>::from_terms(vec![
        term([0, 0], m31(1)),
        term([7, 7], m31(2)),
    ]);
    let difference = nearly_empty.sub(&tail).expect("sub");
    assert_eq!(
        difference.terms(),
        &[
            term([0, 0], m31((order - 1) as u32)),
            term([1, 1], m31(1)),
            term([7, 7], m31((order - 2) as u32)),
        ]
    );
}

/// Jet-output validation names both byte counts and writes nothing on
/// failure; the empty request succeeds.
#[test]
fn jet_output_validation_reports_byte_counts() {
    let p = SparsePolynomial::<Mersenne31, 2>::from_terms(vec![
        term([2, 1], m31(3)),
        term([0, 0], m31(1)),
    ]);
    let point = [m31(2), m31(3)];
    let orders = [MultiIndex::new([1, 0]), MultiIndex::new([0, 1])];
    let unit = core::mem::size_of::<mersenne31::Elem>();
    // One slot short: expected 2 units, actual 1.
    let mut short = [m31(0); 1];
    assert_eq!(
        p.evaluate_jet_into(&point, &orders, &mut short).map(|_| ()),
        Err(ConfigError::BufferLength {
            context: "multivariate jet output",
            expected: 2 * unit,
            actual: unit,
        })
    );
    assert_eq!(short, [m31(0)]);
    // Exact evaluation matches the defining scalar sums.
    let mut output = [m31(0); 2];
    p.evaluate_jet_into(&point, &orders, &mut output)
        .expect("jet");
    assert_eq!(output[0], p.evaluate_hasse(&point, &orders[0]));
    assert_eq!(output[1], p.evaluate_hasse(&point, &orders[1]));
    // Hasse evaluation beyond a term's support contributes zero per term.
    assert_eq!(p.evaluate_hasse(&point, &MultiIndex::new([9, 9])), m31(0));
    // Empty requests succeed without touching anything.
    let mut empty: [mersenne31::Elem; 0] = [];
    p.evaluate_jet_into(&point, &[], &mut empty).expect("empty");
}

/// Weighted and total-degree overflows are checked errors, and the empty
/// polynomial has no leading term under any order.
#[test]
fn degree_overflows_are_checked_and_empty_has_no_leading_term() {
    let p = SparsePolynomial::<Mersenne31, 2>::from_terms(vec![
        term([2, 0], m31(1)),
        term([0, 1], m31(1)),
    ]);
    assert!(
        p.leading_term(&MonomialOrder::Weighted([usize::MAX, 1]))
            .is_err()
            || p.leading_term(&MonomialOrder::Weighted([usize::MAX, 0]))
                .is_err()
    );
    assert!(
        SparsePolynomial::<Mersenne31, 2>::zero()
            .leading_term(&MonomialOrder::Lex)
            .expect("empty")
            .is_none()
    );
    assert!(
        SparsePolynomial::<Mersenne31, 2>::zero()
            .leading_term(&MonomialOrder::GradedLex)
            .expect("empty")
            .is_none()
    );
    // A cancelling sum is the canonical zero.
    let negated = p
        .scaled(m31((<Mersenne31 as Field>::ORDER - 1) as u32))
        .expect("negate");
    assert!(p.add(&negated).expect("cancel").is_zero());
}

/// Dense conversions count terms exactly and round-trip losslessly,
// including an interior zero row that contributes no terms.
#[test]
fn dense_conversions_count_terms_and_round_trip() {
    // Bivariate with an interior zero row: only rows 0 and 2 contribute.
    let bivar = BivariatePolynomial::from_y_coefficients(vec![
        Polynomial::<Gf8B>::from_coefficients(&[b(1), b(2)]).expect("row"),
        Polynomial::<Gf8B>::zero(),
        Polynomial::<Gf8B>::from_coefficients(&[b(3)]).expect("row"),
    ]);
    let sparse = SparsePolynomial::<Gf8B, 2>::from_bivariate(&bivar).expect("sparse");
    assert_eq!(sparse.terms().len(), 3);
    assert_eq!(sparse.to_bivariate().expect("dense"), bivar);
    // Univariate round trip on a longer polynomial.
    let dense = noise_poly::<Gf8B>(11, 0xE601);
    let sparse1 = SparsePolynomial::<Gf8B, 1>::from_univariate(&dense).expect("sparse");
    assert_eq!(sparse1.to_univariate().expect("dense"), dense);
    // Zero conversions stay zero in both arities.
    assert!(
        SparsePolynomial::<Gf8B, 2>::from_bivariate(&BivariatePolynomial::zero())
            .expect("zero")
            .is_zero()
    );
}

/// Hasse derivatives drop zero-factor terms: `C(2,1) = 0` over Gf8B removes
/// the term, while M31 keeps `2·X`.
#[test]
fn hasse_derivatives_drop_zero_factor_terms() {
    let square = SparsePolynomial::<Gf8B, 2>::from_terms(vec![term([2, 0], b(1))]);
    assert!(
        square
            .hasse_derivative(&MultiIndex::new([1, 0]))
            .expect("derivative")
            .is_zero()
    );
    let square_m31 = SparsePolynomial::<Mersenne31, 2>::from_terms(vec![term([2, 0], m31(1))]);
    assert_eq!(
        square_m31
            .hasse_derivative(&MultiIndex::new([1, 0]))
            .expect("derivative")
            .terms(),
        &[term([1, 0], m31(2))]
    );
    // Beyond-support orders vanish without building anything.
    assert!(
        square_m31
            .hasse_derivative(&MultiIndex::new([3, 0]))
            .expect("beyond")
            .is_zero()
    );
    // Coefficient lookup: hit returns the value, miss returns zero.
    assert_eq!(square_m31.coefficient(&MultiIndex::new([2, 0])), m31(1));
    assert!(square_m31.coefficient(&MultiIndex::new([1, 1])).is_zero());
}

/// Sparse products over M31 carry real characteristic-exact coefficients.
#[test]
fn sparse_products_carry_exact_coefficients() {
    // (2·X0)(3·X0) = 6·X0², never a bare XOR.
    let left = SparsePolynomial::<Mersenne31, 2>::from_terms(vec![term([1, 0], m31(2))]);
    let right = SparsePolynomial::<Mersenne31, 2>::from_terms(vec![term([1, 0], m31(3))]);
    assert_eq!(
        left.multiply(&right).expect("product").terms(),
        &[term([2, 0], m31(6))]
    );
    // from_terms merges duplicates by field addition: 2 + 3 = 5.
    let merged = SparsePolynomial::<Mersenne31, 2>::from_terms(vec![
        term([1, 0], m31(2)),
        term([1, 0], m31(3)),
        term([0, 0], m31(0)),
    ]);
    assert_eq!(merged.terms(), &[term([1, 0], m31(5))]);
    // Evaluation is the defining sum at the point.
    assert_eq!(merged.evaluate(&[m31(4), m31(9)]), m31(20));
    let _ = noise::<Gf8B>(1, 0xE602);
}
