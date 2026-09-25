//! Sparse multivariate arithmetic: sums, differences, scaling,
//! products, and Hasse derivatives against hand-expanded values.
//!
//! Normalization, ordering, and dense conversions live in
//! `multivariate.rs`; every operation here asserts exact coefficients,
//! so a sign or factor error cannot hide.

use fgf::field::Field;
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::{
    BivariatePolynomial, ConfigError, MonomialOrder, MultiIndex, SparsePolynomial, Term,
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

fn poly2<F: FieldKernels>(terms: Vec<Term<F, 2>>) -> SparsePolynomial<F, 2> {
    SparsePolynomial::from_terms(terms)
}

/// (X0 + 2 X1) + (3 X0 + X1) = 4 X0 + 3 X1, merged in storage order,
#[test]
fn sums_merge_in_storage_order_and_cancel() {
    let left = poly2::<Mersenne31>(vec![term([1, 0], m31(1)), term([0, 1], m31(2))]);
    let right = poly2::<Mersenne31>(vec![
        term([1, 0], m31(3)),
        term([0, 0], m31(9)),
        term([0, 1], m31(1)),
    ]);
    let sum = left.add(&right).expect("add");
    assert_eq!(
        sum.terms(),
        &[
            term([0, 0], m31(9)),
            term([0, 1], m31(3)),
            term([1, 0], m31(4)),
        ]
    );

    // Cancellation merges by the whole exponent vector: only the shared
    // support cancels.
    let negated = left
        .scaled(m31((<Mersenne31 as Field>::ORDER - 1) as u32))
        .expect("negate");
    assert!(left.add(&negated).expect("cancel").is_zero());
}

/// Subtraction scales by the additive inverse: `X0 − (X0 + X1) = −X1`,
/// and over M31 that negation is `p − 1`, never a bare XOR.
#[test]
fn differences_negate_before_merging() {
    let left = poly2::<Mersenne31>(vec![term([1, 0], m31(1))]);
    let right = poly2::<Mersenne31>(vec![term([1, 0], m31(1)), term([0, 1], m31(1))]);
    let difference = left.sub(&right).expect("sub");
    let order = <Mersenne31 as Field>::ORDER;
    assert_eq!(difference.terms(), &[term([0, 1], m31((order - 1) as u32))]);

    // A zero scale yields the canonical zero polynomial.
    assert!(left.scaled(m31(0)).expect("zero scale").is_zero());
    // Scaling preserves storage order: the sorted input stays sorted,
    // with every coefficient tripled and nothing re-merged.
    let scaled = right.scaled(m31(3)).expect("scaled");
    assert_eq!(
        scaled.terms(),
        &[term([0, 1], m31(3)), term([1, 0], m31(3))]
    );
}

/// (1 + X0)(1 + X1) expands by term-pair convolution, and zero annihilates.
#[test]
fn products_convolve_term_pairs() {
    let left = poly2::<Mersenne31>(vec![term([0, 0], m31(1)), term([1, 0], m31(1))]);
    let right = poly2::<Mersenne31>(vec![term([0, 0], m31(1)), term([0, 1], m31(1))]);
    let product = left.multiply(&right).expect("multiply");
    assert_eq!(
        product.terms(),
        &[
            term([0, 0], m31(1)),
            term([0, 1], m31(1)),
            term([1, 0], m31(1)),
            term([1, 1], m31(1)),
        ]
    );
    assert!(
        left.multiply(&SparsePolynomial::zero())
            .expect("zero product")
            .is_zero()
    );
    assert!(
        SparsePolynomial::<Mersenne31, 2>::zero()
            .multiply(&right)
            .expect("zero product")
            .is_zero()
    );

    // An exponent sum that cannot be represented is an error, never a
    // wrapped degree: `usize::MAX + 1` overflows the addition.
    let huge = poly2::<Mersenne31>(vec![term([usize::MAX, 0], m31(1))]);
    let unit = poly2::<Mersenne31>(vec![term([1, 0], m31(1))]);
    assert_eq!(
        huge.multiply(&unit).map(|_| ()),
        Err(ConfigError::GeometryOverflow {
            context: "multivariate exponent sum",
        })
    );
}

/// `D^[(1,0)](X0²·X1 + 3·X2) = 2·X0·X1` over M31: the `C(2,1)` factor
/// survives prime arithmetic and vanishes in characteristic two.
#[test]
fn hasse_derivatives_carry_the_binomial_factor() {
    let polynomial: SparsePolynomial<Mersenne31, 3> =
        SparsePolynomial::from_terms(vec![term([2, 1, 0], m31(1)), term([0, 0, 1], m31(3))]);
    let derivative = polynomial
        .hasse_derivative(&MultiIndex::new([1, 0, 0]))
        .expect("derivative");
    assert_eq!(derivative.terms(), &[term([1, 1, 0], m31(2))]);

    // An order beyond every term's support differentiates to zero.
    assert!(
        polynomial
            .hasse_derivative(&MultiIndex::new([3, 0, 0]))
            .expect("empty")
            .is_zero()
    );

    // In characteristic two `C(2,1) = 0` drops the term entirely.
    let binary: SparsePolynomial<Gf8B, 3> =
        SparsePolynomial::from_terms(vec![term([2, 1, 0], b(1)), term([0, 0, 1], b(3))]);
    assert!(
        binary
            .hasse_derivative(&MultiIndex::new([1, 0, 0]))
            .expect("binary")
            .is_zero()
    );
}

/// The empty sparse polynomial converts to the empty dense bivariate.
#[test]
fn empty_and_missing_monomials_read_as_zero() {
    let zero: SparsePolynomial<Mersenne31, 2> = SparsePolynomial::zero();
    assert_eq!(
        zero.to_bivariate().expect("dense"),
        BivariatePolynomial::zero()
    );
    let polynomial = poly2::<Mersenne31>(vec![term([1, 0], m31(4))]);
    assert_eq!(polynomial.coefficient(&MultiIndex::new([0, 1])), m31(0));
    // A weighted comparison that cannot be represented is an error:
    // ranking the second term against the first multiplies the maximal
    // weight by two and overflows the degree arithmetic.
    let quadratic = poly2::<Mersenne31>(vec![term([0, 1], m31(4)), term([2, 0], m31(4))]);
    assert_eq!(
        quadratic
            .leading_term(&MonomialOrder::Weighted([usize::MAX, 0]))
            .map(|_| ()),
        Err(ConfigError::GeometryOverflow {
            context: "weighted monomial degree",
        })
    );
}
