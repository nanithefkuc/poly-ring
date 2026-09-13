//! The dense bivariate surface against scalar expansion oracles.
//!
//! Every characteristic-dependent rule — the Hasse factor, the `Y`
//! substitution, the root sign — is exercised over both a binary field and
//! a prime field, so a copied binary parity shortcut cannot survive here.

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::{BivariatePolynomial, Polynomial, WeightedTerm, binomial};

#[cfg(feature = "fft")]
use poly_ring::PolynomialProductScratch;

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

fn gf8b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

/// Scalar rows for `Q(X,Y) = Σ rows[j](X)·Y^j`.
fn rows<F: FieldKernels>(rows: &[&[F::Elem]]) -> BivariatePolynomial<F> {
    BivariatePolynomial::from_y_coefficients(
        rows.iter()
            .map(|row| Polynomial::from_coefficients(row).expect("row"))
            .collect(),
    )
}

/// The direct bivariate Hasse value, as the defining binomial sum:
/// `Σ C(k,r)·C(j,s)·q_{j,k}·x^{k−r}·y^{j−s}`.
fn direct_hasse<F: FieldKernels>(
    polynomial: &BivariatePolynomial<F>,
    x: F::Elem,
    y: F::Elem,
    x_order: usize,
    y_order: usize,
) -> F::Elem {
    let mut result = F::Elem::ZERO;
    for (y_degree, row) in polynomial.y_coefficients().iter().enumerate() {
        if y_degree < y_order {
            continue;
        }
        for (x_degree, coefficient) in row.coefficients().enumerate() {
            if x_degree < x_order {
                continue;
            }
            result = result.add(
                coefficient
                    .mul(binomial::<F>(x_degree, x_order))
                    .mul(binomial::<F>(y_degree, y_order))
                    .mul(x.pow((x_degree - x_order) as u64))
                    .mul(y.pow((y_degree - y_order) as u64)),
            );
        }
    }
    result
}

/// `Q = X²Y²` at `(1,1)`: the mixed derivative is `C(2,1)·C(2,1) = 4` over
/// the prime field and zero in characteristic two.
#[test]
fn mixed_characteristic_hasse_discrepancy_holds() {
    // Q = X²·Y²: the Y² row holds X².
    let prime = rows::<Mersenne31>(&[&[], &[], &[m31(0), m31(0), m31(1)]]);
    assert_eq!(prime.hasse_discrepancy(m31(1), m31(1), 1, 1), m31(4));
    let binary = rows::<Gf8B>(&[&[], &[], &[gf8b(0), gf8b(0), gf8b(1)]]);
    assert!(binary.hasse_discrepancy(gf8b(1), gf8b(1), 1, 1).is_zero());
}

/// Substituting `Y = 1 + X·Z` into `X²Y²` gives rows `X²`, `2X³`, `X⁴`
/// over M31 and no middle row over Gf8B — the binomial `C(2,1)` vanishes
/// in characteristic two while the parity rule would have kept it.
#[test]
fn linear_substitution_expands_over_every_characteristic() {
    let prime = rows::<Mersenne31>(&[&[], &[], &[m31(0), m31(0), m31(1)]]);
    let substituted = prime.substitute_y_linear(m31(1)).expect("substitution");
    let middle = substituted.y_coefficient(1).expect("row 1");
    assert_eq!(middle.coefficient(3), m31(2));
    assert_eq!(substituted.y_coefficient(0).unwrap().degree(), Some(2));
    assert_eq!(substituted.y_coefficient(2).unwrap().degree(), Some(4));

    let binary = rows::<Gf8B>(&[&[], &[], &[gf8b(0), gf8b(0), gf8b(1)]]);
    let substituted = binary.substitute_y_linear(gf8b(1)).expect("substitution");
    let middle = substituted.y_coefficient(1).expect("row 1 stored");
    assert!(middle.is_zero(), "the middle row must vanish, not survive");
    assert!(
        substituted.evaluate(gf8b(1), gf8b(1)).is_zero(),
        "Q(X, 1+XZ) at (1,1) is X²(1+X)² = 0 in characteristic two"
    );
}

/// Row-wise `add`, `sub`, `scaled`, `multiply`, and `hasse_derivative`
/// agree with scalar expansion, and orders beyond either support produce
/// the canonical zero.
#[test]
fn ring_arithmetic_matches_scalar_expansion() {
    fn check<F: FieldKernels>() {
        let left = rows::<F>(&[&[F::Elem::ONE, F::Elem::ONE], &[], &[F::Elem::ONE]]);
        let right = rows::<F>(&[&[F::Elem::ONE], &[F::Elem::ONE]]);
        let x = F::GENERATOR;
        let y = F::GENERATOR.pow(3);

        // Sums and differences through evaluation.
        let sum = left.add(&right).expect("add");
        assert_eq!(
            sum.evaluate(x, y),
            left.evaluate(x, y).add(right.evaluate(x, y))
        );
        let difference = left.sub(&right).expect("sub");
        assert_eq!(
            difference.evaluate(x, y),
            left.evaluate(x, y).add(right.evaluate(x, y).neg())
        );
        // Over characteristic two the two coincide; the prime branch of
        // this test separates them through `sub`'s negation above.

        let scale = F::GENERATOR.pow(5);
        let scaled = left.scaled(scale).expect("scaled");
        assert_eq!(scaled.evaluate(x, y), left.evaluate(x, y).mul(scale));

        // The product is the convolution of `Y` rows.
        let product = left.multiply(&right).expect("multiply");
        assert_eq!(
            product.evaluate(x, y),
            left.evaluate(x, y).mul(right.evaluate(x, y))
        );

        // The bivariate Hasse derivative as a polynomial: its evaluation
        // matches the direct binomial sum at every requested order.
        for x_order in 0..=2 {
            for y_order in 0..=2 {
                let derivative = left.hasse_derivative(x_order, y_order).expect("derivative");
                assert_eq!(
                    derivative.evaluate(x, y),
                    direct_hasse(&left, x, y, x_order, y_order)
                );
            }
        }
        // Orders beyond either support produce the zero polynomial.
        assert!(left.hasse_derivative(0, 3).unwrap().is_zero());
        assert!(left.hasse_derivative(2, 0).unwrap().is_zero());
    }
    check::<Gf8B>();
    check::<Mersenne31>();
}

/// The `(1, y_weight)` weighted order and the geometric accessors keep
/// their contracts, including the tie rule and overflow rejections.
#[test]
fn weighted_order_and_geometry_hold() {
    // The tied case `X² + Y` at weight 2 chooses `Y`.
    let mut tied = rows::<Mersenne31>(&[&[], &[m31(1)]]);
    tied.set_y_coefficient(
        0,
        Polynomial::from_coefficients(&[m31(1), m31(0), m31(1)]).unwrap(),
    )
    .unwrap();
    assert_eq!(
        tied.weighted_leading_term(2).expect("leading"),
        Some(WeightedTerm {
            x_degree: 0,
            y_degree: 1,
            weighted_degree: 2,
        })
    );

    // A weighted degree that cannot be represented is an error, never a
    // wrapped value: a row above Y^0 scales by the huge weight.
    let huge = rows::<Mersenne31>(&[&[], &[], &[m31(1)]]);
    let overflow = (usize::MAX / 2) + 2;
    assert!(matches!(
        huge.weighted_leading_term(overflow),
        Err(poly_ring::ConfigError::GeometryOverflow { .. })
    ));

    // Setting the top row to zero re-normalizes; the zero polynomial has
    // no rows.
    let mut q = rows::<Mersenne31>(&[&[m31(1)], &[m31(2)]]);
    q.set_y_coefficient(1, Polynomial::zero()).expect("set");
    assert_eq!(q.y_coefficient_count(), 1);
    assert!(!q.is_zero());
    q.set_y_coefficient(0, Polynomial::zero()).expect("set");
    assert!(q.is_zero());
}

/// `compose_y`/`has_root` mean `Q(X, f(X)) == 0` — divisibility by
/// `Y − f(X)` — so over a prime field the sign of the candidate matters.
#[test]
fn root_predicate_respects_the_sign() {
    let f = Polynomial::<Mersenne31>::from_coefficients(&[m31(5), m31(1)]).expect("f");
    let negative_one = mersenne31::Elem::ZERO.sub(mersenne31::Elem::ONE);
    let minus_f = f.scaled(negative_one);
    // Y − f(X): the Y row is one, the constant row is −f.
    let minus = BivariatePolynomial::from_y_coefficients(vec![minus_f, Polynomial::one().unwrap()]);
    assert!(minus.has_root(&f).expect("has_root"));

    // Y + f(X): identical in characteristic two, different over M31.
    let plus_rows = vec![f.clone(), Polynomial::one().unwrap()];
    let plus = BivariatePolynomial::from_y_coefficients(plus_rows);
    assert!(!plus.has_root(&f).expect("has_root"));
    // …and the predicate is the honest evaluation `Q(X, f(X)) == 0`.
    let value = plus.compose_y(&f).expect("compose");
    assert_eq!(value, f.add(&f).unwrap());
}

/// Bulk packed row ingress: full validation before mutation, canonical
/// rows on the way in, stale high-`Y` rows dropped, empty input resetting
/// to zero, and capacity reused on the second write.
#[test]
fn packed_row_ingress_holds_its_contract() {
    // Overwrite a larger previous polynomial without leaving old high-Y
    // coefficients, and canonicalize prime rows through the ingress.
    let mut q = rows::<Mersenne31>(&[&[m31(1)], &[m31(2)], &[m31(3)], &[m31(4)]]);
    let element = Mersenne31::BYTES;
    let mut small_row = vec![0_u8; 2 * element];
    // A lane at or beyond the prime folds to its field value: `p + 7`
    // arrives through the ingress and reads back as 7.
    let noncanonical = <Mersenne31 as fgf::field::Field>::ORDER as u32 + 7;
    <Mersenne31 as fgf::field::Field>::write(
        &mut small_row[..element],
        mersenne31::Elem::from_raw(noncanonical),
    );
    q.assign_y_coefficients_packed([&small_row[..], &small_row[..2 * element]].into_iter())
        .expect("assign");
    assert_eq!(q.y_coefficient_count(), 2);
    assert_eq!(q.y_coefficient(0).unwrap().coefficient(0), m31(7));
    assert_eq!(q.y_coefficient(1).unwrap().coefficient(0), m31(7));
    assert!(q.y_coefficient(3).is_none());

    // A partial field element is rejected before any write happens.
    let mut intact = rows::<Mersenne31>(&[&[m31(1)]]);
    let partial = vec![0_u8; element - 1];
    let before = intact.clone();
    assert!(matches!(
        intact.assign_y_coefficients_packed([&partial[..]].into_iter()),
        Err(poly_ring::PolynomialError::Config(
            poly_ring::ConfigError::BufferLength { .. }
        ))
    ));
    assert_eq!(intact, before);

    // Empty input resets to zero.
    intact
        .assign_y_coefficients_packed(core::iter::empty::<&[u8]>())
        .expect("empty");
    assert!(intact.is_zero());
}

/// The retained batched-product substitution stays byte-identical to the
/// scalar substitution over the binary fields it is bounded to.
#[cfg(feature = "fft")]
#[test]
fn affine_substitution_scalar_and_batched_agree_over_binary() {
    use butterfly_fft::core::kernel::ButterflyKernels;
    use fgf::Gf16;

    fn check<F>()
    where
        F: FieldKernels + ButterflyKernels,
    {
        let prefix = Polynomial::<F>::from_coefficients(&[
            F::GENERATOR,
            F::GENERATOR.pow(2),
            F::GENERATOR.pow(7),
        ])
        .expect("prefix");
        let q = rows::<F>(&[
            &[F::GENERATOR, F::Elem::ONE, F::GENERATOR.pow(4)],
            &[],
            &[F::GENERATOR.pow(9), F::Elem::ONE],
            &[F::Elem::ONE],
        ]);
        let mut scratch = PolynomialProductScratch::<F>::new();
        for tail_degree in [1_usize, 2, 3] {
            for coefficient_count in [5_usize, 9, 17] {
                let scalar = q
                    .substitute_y_affine_truncated(&prefix, tail_degree, coefficient_count)
                    .expect("scalar");
                let batched = q
                    .substitute_y_affine_truncated_fast(
                        &prefix,
                        tail_degree,
                        coefficient_count,
                        &mut scratch,
                    )
                    .expect("batched");
                assert_eq!(scalar, batched);
            }
        }
    }
    check::<Gf8B>();
    check::<Gf16>();
}
