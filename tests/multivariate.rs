//! The sparse multivariate surface against scalar monomial-sum oracles.
//!
//! Cancellation, characteristic behavior, the ordered jet buffer, the
//! monomial orders, and the dense conversions each defend one contract of
//! the new representation.

use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::{
    BivariatePolynomial, ConfigError, MonomialOrder, MultiIndex, Polynomial, SparsePolynomial, Term,
};

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

fn gf8b(value: u8) -> gf8b::Elem {
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

/// Terms `+X0·X2` and `−X0·X2` cancel completely, while `X0·X1` survives:
/// cancellation merges by the whole exponent vector, so distinct variable
/// coordinates cannot collapse into it.
#[test]
fn duplicate_terms_cancel_across_distinct_coordinates() {
    // The negated coefficients are p − value: over M31 that is
    // 2^31 − 1 − value, not small-integer arithmetic.
    let minus_two = m31((<Mersenne31 as fgf::field::Field>::ORDER - 2) as u32);
    let minus_seven = m31((<Mersenne31 as fgf::field::Field>::ORDER - 7) as u32);
    let cancelled: SparsePolynomial<Mersenne31, 3> = SparsePolynomial::from_terms(vec![
        term::<Mersenne31, 3>([1, 0, 1], m31(2)),
        term::<Mersenne31, 3>([1, 0, 1], minus_two),
        term::<Mersenne31, 3>([1, 1, 0], m31(5)),
    ]);
    assert_eq!(cancelled.terms().len(), 1);
    assert_eq!(cancelled.terms()[0].exponents.exponents(), &[1, 1, 0]);
    assert_eq!(cancelled.terms()[0].coefficient, m31(5));

    let none: SparsePolynomial<Mersenne31, 3> = SparsePolynomial::from_terms(vec![
        term::<Mersenne31, 3>([1, 0, 1], m31(7)),
        term::<Mersenne31, 3>([1, 0, 1], minus_seven),
    ]);
    assert!(none.is_zero());
}

/// Noncanonical prime lanes in duplicate terms merge by field value, not
/// by raw encoding: values `5` and `p + 5` sum to `10`.
#[test]
fn noncanonical_duplicates_compare_by_field_value() {
    let raw = SparsePolynomial::<Mersenne31, 1>::from_terms(vec![
        term::<Mersenne31, 1>([0], mersenne31::Elem::from_raw(5)),
        term::<Mersenne31, 1>(
            [0],
            mersenne31::Elem::from_raw((<Mersenne31 as fgf::field::Field>::ORDER as u32) + 5),
        ),
    ]);
    assert_eq!(raw.terms().len(), 1);
    // The merged coefficient is the field sum of the two values.
    assert_eq!(raw.terms()[0].coefficient, m31(10));
    assert_eq!(raw.coefficient(&MultiIndex::new([0])), m31(10));
}

/// `P = X0²·X1 + 3·X2` over M31 evaluates to `24` at `(2,3,4)` and its
/// Hasse derivative of order `[1,1,0]` is `4`; in characteristic two that
/// derivative vanishes.
#[test]
fn evaluation_and_hasse_orders_match_independent_values() {
    let p: SparsePolynomial<Mersenne31, 3> = SparsePolynomial::from_terms(vec![
        term::<Mersenne31, 3>([2, 1, 0], m31(1)),
        term::<Mersenne31, 3>([0, 0, 1], m31(3)),
    ]);
    assert_eq!(p.evaluate(&[m31(2), m31(3), m31(4)]), m31(24));
    assert_eq!(
        p.evaluate_hasse(&[m31(2), m31(3), m31(4)], &MultiIndex::new([1, 1, 0])),
        m31(4)
    );

    let binary: SparsePolynomial<Gf8B, 3> = SparsePolynomial::from_terms(vec![
        term::<Gf8B, 3>([2, 1, 0], gf8b(1)),
        term::<Gf8B, 3>([0, 0, 1], gf8b(3)),
    ]);
    assert!(
        binary
            .evaluate_hasse(&[gf8b(1), gf8b(1), gf8b(1)], &MultiIndex::new([1, 1, 0]))
            .is_zero()
    );
}

/// `N = 0` is the constant ring: exponents are empty, evaluation ignores
/// the point, and arithmetic degenerates to constants.
#[test]
fn zero_arity_is_the_constant_ring() {
    let index = MultiIndex::<0>::new([]);
    assert_eq!(index.total_degree().expect("degree"), 0);
    let summed = index.checked_add(&index).expect("add");
    assert_eq!(summed.exponents(), &[]);

    let constant: SparsePolynomial<Mersenne31, 0> =
        SparsePolynomial::from_terms(vec![term::<Mersenne31, 0>([], m31(9))]);
    assert_eq!(constant.evaluate(&[]), m31(9));
    let zero: SparsePolynomial<Mersenne31, 0> = SparsePolynomial::zero();
    assert_eq!(zero.evaluate(&[]), m31(0));
}

/// The ordered jet buffer: repeats preserved in caller order, empty
/// requests succeed, and a mismatched output is rejected before writes.
#[test]
fn ordered_jet_requests_hold_their_contracts() {
    let p: SparsePolynomial<Mersenne31, 2> = SparsePolynomial::from_terms(vec![
        term::<Mersenne31, 2>([1, 0], m31(2)),
        term::<Mersenne31, 2>([0, 1], m31(3)),
    ]);
    let point = [m31(4), m31(5)];
    let orders = [
        MultiIndex::new([0, 0]),
        MultiIndex::new([1, 0]),
        MultiIndex::new([0, 0]),
    ];
    let mut output = [m31(0); 3];
    p.evaluate_jet_into(&point, &orders, &mut output)
        .expect("jet");
    // Repeats preserved: positions 0 and 2 both hold f(4,5) = 2·4 + 3·5.
    let value = m31(2 * 4 + 3 * 5);
    assert_eq!(output, [value, m31(2), value]);

    // The empty request succeeds with an empty output buffer.
    p.evaluate_jet_into(&point, &[], &mut [])
        .expect("empty jet");

    let mismatch: Vec<MultiIndex<2>> = vec![MultiIndex::new([0, 0])];
    let mut short = [m31(0); 1];
    short[0] = m31(42);
    assert!(matches!(
        p.evaluate_jet_into(&point, &mismatch, &mut []),
        Err(ConfigError::BufferLength { .. })
    ));
    assert!(matches!(
        p.evaluate_jet_into(&point, &[], &mut short),
        Err(ConfigError::BufferLength { .. })
    ));
}

/// Exponent arithmetic rejects overflow instead of wrapping, and the
/// weighted order `[1, w]` ranks exactly like the dense `WeightedTerm`.
#[test]
fn index_arithmetic_and_weighted_order_hold() {
    let huge = MultiIndex::new([usize::MAX, 1]);
    assert!(matches!(
        huge.checked_add(&MultiIndex::new([1, 1])),
        Err(ConfigError::GeometryOverflow { .. })
    ));
    assert!(matches!(
        huge.total_degree(),
        Err(ConfigError::GeometryOverflow { .. })
    ));

    // The tied case X² + Y at weight 2 chooses Y — the same rule the dense
    // WeightedTerm applies for larger Y degree.
    let tied: SparsePolynomial<Mersenne31, 2> = SparsePolynomial::from_terms(vec![
        term::<Mersenne31, 2>([2, 0], m31(1)),
        term::<Mersenne31, 2>([0, 1], m31(1)),
    ]);
    let leading = tied
        .leading_term(&MonomialOrder::Weighted([1, 2]))
        .expect("leading")
        .expect("nonzero");
    assert_eq!(leading.exponents.exponents(), &[0, 1]);

    // Under [1, 2] the sparse leading term agrees with the dense one.
    let dense = BivariatePolynomial::<Mersenne31>::from_y_coefficients(vec![
        Polynomial::from_coefficients(&[m31(1), m31(1)]).unwrap(),
        Polynomial::from_coefficients(&[m31(1)]).unwrap(),
    ]);
    let sparse = SparsePolynomial::from_bivariate(&dense).expect("sparse");
    let sparse_leading = sparse
        .leading_term(&MonomialOrder::Weighted([1, 4]))
        .expect("leading")
        .expect("nonzero");
    let dense_leading = dense.weighted_leading_term(4).expect("leading").unwrap();
    assert_eq!(
        sparse_leading.exponents.exponents(),
        &[dense_leading.x_degree, dense_leading.y_degree]
    );
    // Storage order never changed because a different order was requested.
    let lex = sparse
        .leading_term(&MonomialOrder::Lex)
        .expect("lex")
        .unwrap();
    assert_eq!(lex.coefficient, m31(1));
    assert_eq!(
        sparse.terms().first().unwrap().exponents.exponents(),
        &[0, 0]
    );
}

/// Dense ↔ sparse conversion round-trips, keeps internal zero `Y` rows as
/// stored coefficients, and refuses unrepresentable exponents instead of
/// wrapping an allocation.
#[test]
fn conversions_round_trip_and_reject_overflow() {
    // Internal zero Y row: row 1 is zero, row 2 is not.
    let dense = BivariatePolynomial::<Mersenne31>::from_y_coefficients(vec![
        Polynomial::from_coefficients(&[m31(1), m31(2)]).unwrap(),
        Polynomial::zero(),
        Polynomial::from_coefficients(&[m31(3)]).unwrap(),
    ]);
    let sparse = SparsePolynomial::from_bivariate(&dense).expect("sparse");
    assert_eq!(sparse.terms().len(), 3);
    let back = sparse.to_bivariate().expect("dense");
    assert_eq!(back, dense);
    assert_eq!(
        back.y_coefficient(1),
        Some(&Polynomial::<Mersenne31>::zero())
    );

    // Univariate conversions round-trip; N = 1.
    let univariate =
        Polynomial::<Mersenne31>::from_coefficients(&[m31(4), m31(5), m31(6)]).unwrap();
    let sparse = SparsePolynomial::from_univariate(&univariate).expect("sparse");
    assert_eq!(sparse.terms().len(), 3);
    assert_eq!(sparse.to_univariate().expect("dense"), univariate);
    let zero: SparsePolynomial<Mersenne31, 1> = SparsePolynomial::zero();
    assert!(zero.to_univariate().expect("dense").is_zero());

    // An unrepresentable dense exponent is an error, not a wrapped
    // allocation: Y-degree usize::MAX cannot become a row count.
    let absurd: SparsePolynomial<Mersenne31, 2> =
        SparsePolynomial::from_terms(vec![term::<Mersenne31, 2>([0, usize::MAX], m31(1))]);
    assert!(matches!(
        absurd.to_bivariate(),
        Err(poly_ring::PolynomialError::Config(
            ConfigError::GeometryOverflow { .. }
        ))
    ));
}
