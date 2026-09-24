//! Ring and roots behavior and error boundaries.

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16, Goldilocks, Mersenne31, gf8b, mersenne31};
use poly_ring::{
    BaseFieldRoots, Polynomial, RootError, base_field_roots, chien_roots, linearized_roots,
};

use crate::oracles;
use oracles::{naive_evaluate, noise, noise_poly};

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

/// `BaseFieldRoots` accessors distinguish the zero polynomial from a finite
/// set, and ordering is the frozen canonical key order.
#[test]
fn base_field_roots_accessors_distinguish_all_from_finite() {
    let all = BaseFieldRoots::<gf8b::Elem>::All;
    assert!(all.as_slice().is_none());
    assert!(all.clone().into_finite().is_none());

    let finite = BaseFieldRoots::Finite(vec![b(2), b(1)]);
    assert_eq!(finite.as_slice(), Some(&[b(2), b(1)][..]));
    assert_eq!(finite.into_finite(), Some(vec![b(2), b(1)]));

    // Empty finite set: a nonzero constant has no roots anywhere.
    let constant = Polynomial::<Gf8B>::from_coefficients(&[b(5)]).expect("constant");
    let roots = chien_roots(&constant).expect("chien");
    assert_eq!(roots.as_slice(), Some(&[][..]));
    assert_eq!(roots.into_finite(), Some(Vec::new()));
}

/// A warmed Chien scratch reports the same roots and reuses its buffers.
#[test]
fn chien_into_reuses_scratch_and_matches_scan() {
    use poly_ring::ChienScratch;

    let polynomial = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1), b(1)]).expect("quadratic");
    let mut scratch = ChienScratch::<Gf8B>::new();
    let mut first = Vec::new();
    let all = chien_roots_into_wrap(&polynomial, &mut scratch, &mut first);
    assert!(!all);
    for root in &first {
        assert!(naive_evaluate(&polynomial, *root).is_zero());
    }
    // Second run over a changed polynomial reuses the same scratch.
    let other = noise_poly::<Gf8B>(5, 0xC101);
    let mut second = Vec::new();
    let all_other = chien_roots_into_wrap(&other, &mut scratch, &mut second);
    assert_eq!(all_other, other.is_zero());
    for root in &second {
        assert!(naive_evaluate(&other, *root).is_zero());
    }
    // Default scratch agrees with a fresh one.
    let mut defaulted = ChienScratch::<Gf8B>::default();
    let mut third = Vec::new();
    let all_third = chien_roots_into_wrap(&polynomial, &mut defaulted, &mut third);
    assert_eq!(all_third, all);
    assert_eq!(first, third);
}

fn chien_roots_into_wrap<F: FieldKernels>(
    polynomial: &Polynomial<F>,
    scratch: &mut poly_ring::ChienScratch<F>,
    roots: &mut Vec<F::Elem>,
) -> bool {
    poly_ring::chien_roots_into(roots, polynomial, scratch).expect("chien into")
}

/// Prime fields remain outside the Chien and linearized binary backends.
#[test]
fn binary_specific_backends_reject_prime_fields() {
    let polynomial =
        Polynomial::<Mersenne31>::from_coefficients(&[m31(1), m31(2)]).expect("linear");
    assert_eq!(
        chien_roots(&polynomial).map(|_| ()),
        Err(RootError::UnsupportedField {
            field_order: Mersenne31::ORDER,
            element_bytes: Mersenne31::BYTES,
        })
    );
    assert_eq!(
        linearized_roots(&polynomial, m31(0)).map(|_| ()),
        Err(RootError::UnsupportedField {
            field_order: Mersenne31::ORDER,
            element_bytes: Mersenne31::BYTES,
        })
    );
}

/// Equal-degree scratch reuse: a warmed extraction matches the one-shot, and
/// the default scratch agrees too.
#[test]
fn equal_degree_into_reuses_scratch() {
    use poly_ring::BinaryRootScratch;

    let polynomial = noise_poly::<Gf8B>(9, 0xC102);
    let one_shot = base_field_roots(&polynomial).expect("one shot");
    let mut scratch = BinaryRootScratch::<Gf8B>::new();
    let mut roots = Vec::new();
    let all =
        poly_ring::binary_field_roots_into(&mut roots, &polynomial, &mut scratch).expect("into");
    assert!(!all);
    assert_eq!(BaseFieldRoots::Finite(roots.clone()), one_shot);
    // Capacity is retained for the next extraction when the split path ran;
    // otherwise the one-shot agreement above is the assertion.
    let _ = scratch.capacity();
    // Default scratch agrees with the explicit one.
    let mut defaulted = BinaryRootScratch::<Gf8B>::default();
    let mut roots2 = Vec::new();
    let all2 =
        poly_ring::binary_field_roots_into(&mut roots2, &polynomial, &mut defaulted).expect("into");
    assert_eq!(all2, all);
    assert_eq!(roots, roots2);

    // A constant polynomial has no roots; the scratch path reports finite.
    let constant = Polynomial::<Gf8B>::from_coefficients(&[b(7)]).expect("constant");
    let mut roots3 = Vec::new();
    let all3 =
        poly_ring::binary_field_roots_into(&mut roots3, &constant, &mut scratch).expect("into");
    assert!(!all3);
    assert!(roots3.is_empty());

    // Zero polynomial reports All on both paths.
    let mut roots4 = Vec::new();
    let all4 =
        poly_ring::binary_field_roots_into(&mut roots4, &Polynomial::<Gf8B>::zero(), &mut scratch)
            .expect("into");
    assert!(all4);
    assert_eq!(
        base_field_roots(&Polynomial::<Gf8B>::zero()).expect("zero"),
        BaseFieldRoots::All
    );
}

/// The linearized solver enumerates the kernel of `X^2 + X` over Gf8B: the
/// two-element subfield `{0, 1}`, in canonical key order, and rejects an
/// inconsistent affine shift with the empty set.
#[test]
fn linearized_kernel_and_inconsistent_shift() {
    // L(X) = X^2 + X: coefficients at degrees 1 and 2 only.
    let linearized =
        Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1), b(1)]).expect("linearized");
    let roots = linearized_roots(&linearized, b(0)).expect("roots");
    assert_eq!(roots, vec![b(0), b(1)]);
    for root in &roots {
        assert!(naive_evaluate(&linearized, *root).is_zero());
    }
    // X^2 + X + 1 may still vanish elsewhere in Gf256: assert the affine
    // shift moves the solution set exactly, against the Chien oracle.
    let shifted_poly = {
        let mut shifted = linearized.clone();
        shifted.set_coefficient(0, b(1)).expect("constant");
        shifted
    };
    let shifted_chien = chien_roots(&shifted_poly)
        .expect("chien")
        .into_finite()
        .expect("finite");
    let shifted_linear = linearized_roots(&linearized, b(1)).expect("shifted");
    assert_eq!(shifted_linear, shifted_chien);

    // A wider affine family: L(X) = X^4 + X over Gf16 still matches Chien.
    let wide = Polynomial::<Gf16>::from_coefficients(&[
        <Gf16 as Field>::Elem::ZERO,
        <Gf16 as Field>::Elem::ONE,
        <Gf16 as Field>::Elem::ZERO,
        <Gf16 as Field>::Elem::ZERO,
        <Gf16 as Field>::Elem::ONE,
    ])
    .expect("wide");
    let wide_roots = linearized_roots(&wide, <Gf16 as Field>::Elem::ZERO).expect("wide roots");
    let ordinary = wide.clone();
    let chien = chien_roots(&ordinary)
        .expect("chien")
        .into_finite()
        .expect("finite");
    assert_eq!(wide_roots, chien);
    assert!(!wide_roots.is_empty());
}

/// Truncation and packed conversion preserve retained coefficients.
#[test]
fn dense_truncation_preserves_retained_coefficients() {
    let polynomial = noise_poly::<Gf8B>(6, 0xC103);
    assert_eq!(polynomial.coefficient_count(), 6);
    assert_eq!(polynomial.degree(), Some(5));
    assert_eq!(
        polynomial.leading_coefficient(),
        Some(polynomial.coefficient(5))
    );
    assert!(Polynomial::<Gf8B>::zero().leading_coefficient().is_none());
    assert_eq!(Polynomial::<Gf8B>::zero().degree(), None);

    // Truncation drops high coefficients; resize grows with zeros.
    let mut truncated = polynomial.clone();
    truncated.truncate(3);
    assert_eq!(truncated.coefficient_count(), 3);
    for degree in 0..3 {
        assert_eq!(
            truncated.coefficient(degree),
            polynomial.coefficient(degree)
        );
    }
    let mut grown = truncated.clone();
    grown.resize_coefficients(6).expect("resize");
    assert_eq!(grown.coefficient_count(), 6);
    for degree in 3..6 {
        assert!(grown.coefficient(degree).is_zero());
    }
    // Packed round trip preserves the value exactly.
    let packed = polynomial.as_packed().to_vec();
    let restored = Polynomial::<Gf8B>::from_packed(packed).expect("packed");
    assert_eq!(restored, polynomial);
    assert!(Polynomial::<Gf8B>::from_packed(vec![0_u8, 1, 2]).is_some());
    // Graded order ranks total degree first: [0,5] outranks [4,0].

    // set_coefficient grows and renormalizes.
    let mut point = Polynomial::<Gf8B>::zero();
    point.set_coefficient(2, b(9)).expect("set");
    assert_eq!(point.degree(), Some(2));
    assert_eq!(point.coefficient(2), b(9));
    point.set_coefficient(2, b(0)).expect("clear");
    assert!(point.is_zero());
}

/// Ring paths: square matches multiply on both characteristics, and the
/// packed-kernel scale path agrees with the scalar one.
#[test]
fn square_and_scale_match_the_oracles() {
    fn check<F: FieldKernels>() {
        let polynomial = noise_poly::<F>(9, 0xC104);
        // square == multiply by self exactly.
        assert_eq!(
            polynomial.square().expect("square"),
            polynomial.multiply(&polynomial).expect("multiply")
        );
        let mut into = Polynomial::<F>::zero();
        polynomial.square_into(&mut into).expect("square into");
        assert_eq!(into, polynomial.square().expect("square"));
        // Scaling by zero/one/two matches scalar evaluation everywhere.
        for raw in [F::Elem::ZERO, F::Elem::ONE, F::Elem::ONE.add(F::Elem::ONE)] {
            let scaled = polynomial.scaled(raw);
            for seed in 0..3 {
                let point = noise::<F>(1, 0xC105 + seed)[0];
                assert_eq!(
                    scaled.evaluate(point),
                    naive_evaluate(&polynomial, point).mul(raw)
                );
            }
        }
        // Shift then evaluate equals x^shift times the value.
        let shifted = polynomial.shifted(2).expect("shift");
        for seed in 0..3 {
            let point = noise::<F>(1, 0xC106 + seed)[0];
            assert_eq!(
                shifted.evaluate(point),
                naive_evaluate(&polynomial, point).mul(point.pow(2))
            );
        }
    }
    check::<Gf8B>();
    check::<Mersenne31>();
}

/// Division errors and the divide-by-X scratch paths.
#[test]
fn division_errors_name_the_violation() {
    use poly_ring::PolynomialError;

    let polynomial = noise_poly::<Gf8B>(7, 0xC107);
    // Zero divisor and zero modulus are DivisionByZero, not a panic.
    assert_eq!(
        polynomial.div_rem(&Polynomial::<Gf8B>::zero()).map(|_| ()),
        Err(PolynomialError::DivisionByZero)
    );
    assert_eq!(
        polynomial
            .remainder(&Polynomial::<Gf8B>::zero())
            .map(|_| ()),
        Err(PolynomialError::DivisionByZero)
    );
    assert_eq!(
        polynomial
            .divide_exact(&Polynomial::<Gf8B>::zero())
            .map(|_| ()),
        Err(PolynomialError::DivisionByZero)
    );
    assert_eq!(
        polynomial
            .multiply_mod(&polynomial, &Polynomial::<Gf8B>::zero())
            .map(|_| ()),
        Err(PolynomialError::DivisionByZero)
    );
    assert_eq!(
        polynomial
            .pow_mod(3, &Polynomial::<Gf8B>::zero())
            .map(|_| ()),
        Err(PolynomialError::DivisionByZero)
    );
    // Exact division by a non-divisor names the failure.
    let divisor = noise_poly::<Gf8B>(3, 0xC108);
    assert_eq!(
        polynomial.divide_exact(&divisor).map(|_| ()),
        Err(PolynomialError::NonExactDivision)
    );
    // div_rem_into matches div_rem, including the zero-dividend path.
    let mut quotient = Polynomial::<Gf8B>::zero();
    let mut remainder = Polynomial::<Gf8B>::zero();
    polynomial
        .div_rem_into(&divisor, &mut quotient, &mut remainder)
        .expect("into");
    let (q, r) = polynomial.div_rem(&divisor).expect("div");
    assert_eq!(quotient, q);
    assert_eq!(remainder, r);
    Polynomial::<Gf8B>::zero()
        .div_rem_into(&divisor, &mut quotient, &mut remainder)
        .expect("zero into");
    assert!(quotient.is_zero() && remainder.is_zero());
    // Dividend below the divisor returns a zero quotient and the dividend.
    let small = noise_poly::<Gf8B>(2, 0xC109);
    let big = noise_poly::<Gf8B>(9, 0xC110);
    let (q2, r2) = small.div_rem(&big).expect("small div");
    assert!(q2.is_zero());
    assert_eq!(r2, small);
}

/// Monomial orders rank storage exponents; weighted overflow is checked.
#[test]
fn monomial_orders_rank_and_reject_overflow() {
    use poly_ring::{ConfigError, MonomialOrder, MultiIndex};
    let _low = MultiIndex::new([1_usize, 0]);
    // checked_add overflows instead of wrapping.
    assert_eq!(
        MultiIndex::new([usize::MAX, 0])
            .checked_add(&MultiIndex::new([1, 0]))
            .map(|_| ()),
        Err(ConfigError::GeometryOverflow {
            context: "multivariate exponent sum",
        })
    );
    assert_eq!(
        MultiIndex::new([usize::MAX, usize::MAX])
            .total_degree()
            .map(|_| ()),
        Err(ConfigError::GeometryOverflow {
            context: "multivariate total degree",
        })
    );
    // Weighted overflow is checked, not wrapped: two terms force a real
    // comparison whose dot product exceeds the address space.
    let order = MonomialOrder::<2>::Weighted([usize::MAX, 1]);
    assert!(
        poly_ring::SparsePolynomial::<Mersenne31, 2>::from_terms(vec![
            poly_ring::Term {
                exponents: MultiIndex::new([2, 0]),
                coefficient: m31(1),
            },
            poly_ring::Term {
                exponents: MultiIndex::new([0, 1]),
                coefficient: m31(1),
            },
        ])
        .leading_term(&order)
        .is_err()
    );
    // Graded order ranks total degree first: [0,5] outranks [4,0], while
    // Lex prefers the variable-0 exponent.
    let both = poly_ring::SparsePolynomial::<Mersenne31, 2>::from_terms(vec![
        poly_ring::Term {
            exponents: MultiIndex::new([4, 0]),
            coefficient: m31(1),
        },
        poly_ring::Term {
            exponents: MultiIndex::new([0, 5]),
            coefficient: m31(1),
        },
    ]);
    assert_eq!(
        both.leading_term(&MonomialOrder::<2>::GradedLex)
            .expect("graded")
            .expect("nonzero")
            .exponents
            .exponents(),
        &[0, 5]
    );
    assert_eq!(
        both.leading_term(&MonomialOrder::<2>::Lex)
            .expect("lex")
            .expect("nonzero")
            .exponents
            .exponents(),
        &[4, 0]
    );
}

/// Karatsuba products at the recursion base agree with schoolbook, and the
/// forced entry point maps both zero orders to zero.
#[test]
fn karatsuba_base_and_mid_sizes_agree_with_schoolbook() {
    use oracles::naive_multiply;
    use poly_ring::internals::karatsuba_multiply;

    for (left_len, right_len) in [(1, 1), (2, 3), (47, 49), (60, 5), (100, 100)] {
        let left = noise_poly::<Gf8B>(left_len, 0xC111 + left_len as u64);
        let right = noise_poly::<Gf8B>(right_len, 0xC112 + right_len as u64);
        assert_eq!(
            karatsuba_multiply(&left, &right).expect("karatsuba"),
            naive_multiply(&left, &right),
            "karatsuba diverged at {left_len}x{right_len}"
        );
    }
    // Odd-characteristic Karatsuba carries real subtraction signs.
    for (left_len, right_len) in [(3, 4), (50, 51)] {
        let left = noise_poly::<Mersenne31>(left_len, 0xC113 + left_len as u64);
        let right = noise_poly::<Mersenne31>(right_len, 0xC114 + right_len as u64);
        assert_eq!(
            karatsuba_multiply(&left, &right).expect("karatsuba"),
            naive_multiply(&left, &right),
            "karatsuba diverged at {left_len}x{right_len}"
        );
    }
}

/// Goldilocks coverage for the scalar ring paths the Gf8B suites skip.
#[test]
fn goldilocks_ring_paths_agree() {
    use oracles::naive_multiply;

    let left = noise_poly::<Goldilocks>(11, 0xC115);
    let right = noise_poly::<Goldilocks>(7, 0xC116);
    assert_eq!(
        left.multiply(&right).expect("product"),
        naive_multiply(&left, &right)
    );
    assert_eq!(
        left.square().expect("square"),
        left.multiply(&left).expect("multiply")
    );
    let (q, r) = left.div_rem(&right).expect("div");
    assert_eq!(
        naive_multiply(&q, &right).add(&r).expect("reconstruct"),
        left
    );
}
