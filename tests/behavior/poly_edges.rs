//! Univariate ring edges: the modular arithmetic entry points, the
//! Karatsuba dispatch tier, accessors, and hand-checked Hasse fixtures.
//!
//! The agreement suite in `poly.rs` covers the small-geometry paths;
//! every test here takes a branch it never reaches: zero dividends,
//! modular powers, the dispatched Karatsuba tier, and binary Hasse
//! values expanded by hand.

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::internals::karatsuba_multiply;
use poly_ring::{Polynomial, PolynomialError};

use crate::oracles;
use oracles::{naive_evaluate, naive_multiply, naive_series_inverse, noise, noise_poly};

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

/// A zero dividend divides to zero quotient and zero remainder.
#[test]
fn zero_dividend_divides_to_zero() {
    let divisor = noise_poly::<Gf8B>(5, 0xF001);
    let (quotient, remainder) = Polynomial::<Gf8B>::zero()
        .div_rem(&divisor)
        .expect("division");
    assert!(quotient.is_zero());
    assert!(remainder.is_zero());
}

/// Modular multiply, square, and power agree with repeated naive
/// multiplication, and the zeroth power is one modulo the modulus.
#[test]
fn modular_arithmetic_matches_repeated_multiplication() {
    fn check<F: FieldKernels>() {
        let base = noise_poly::<F>(7, 0xF010);
        let factor = noise_poly::<F>(5, 0xF011);
        let modulus = noise_poly::<F>(9, 0xF012);

        let product = base.multiply_mod(&factor, &modulus).expect("mul mod");
        let expected = naive_multiply(&base, &factor)
            .div_rem(&modulus)
            .expect("remainder")
            .1;
        assert_eq!(product, expected);

        let square = base.square_mod(&modulus).expect("square mod");
        assert_eq!(square, base.multiply_mod(&base, &modulus).expect("mul mod"));

        // x^13 by square-and-multiply equals thirteen naive folds.
        let power = base.pow_mod(13, &modulus).expect("pow mod");
        let mut folded = Polynomial::<F>::one().expect("one");
        for _ in 0..13 {
            folded = folded.multiply_mod(&base, &modulus).expect("fold");
        }
        assert_eq!(power, folded);

        // The zeroth power is one reduced modulo the modulus.
        let one = Polynomial::<F>::one()
            .expect("one")
            .div_rem(&modulus)
            .expect("remainder")
            .1;
        assert_eq!(base.pow_mod(0, &modulus).expect("pow zero"), one);
    }
    check::<Gf8B>();
    check::<Mersenne31>();
}

/// Dividing by a higher power of X than the valuation allows is a
/// non-exact division, not a silent truncation.
#[test]
fn x_power_division_below_the_valuation_is_not_exact() {
    let x = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1)]).expect("X");
    assert_eq!(
        x.divide_by_x_power(2).map(|_| ()),
        Err(PolynomialError::NonExactDivision)
    );
    assert_eq!(
        x.divide_by_x_power(1).expect("exact").coefficient_count(),
        1
    );
}

/// Coefficients beyond the stored degree read as zero, including at the
/// address-space edge.
#[test]
fn out_of_range_coefficients_read_as_zero() {
    let polynomial = noise_poly::<Gf8B>(4, 0xF020);
    assert!(polynomial.coefficient(100).is_zero());
    assert!(polynomial.coefficient(usize::MAX).is_zero());
}

/// `clone_from` reuses the destination buffer: the capacity survives the
/// copy and the value matches.
#[test]
fn clone_from_reuses_the_destination_buffer() {
    let source = noise_poly::<Gf8B>(9, 0xF030);
    let mut destination = noise_poly::<Gf8B>(24, 0xF031);
    let capacity = destination.retained_capacity_bytes();
    assert!(capacity > 0);
    destination.clone_from(&source);
    assert_eq!(destination, source);
    assert_eq!(destination.retained_capacity_bytes(), capacity);
    assert_eq!(Polynomial::<Gf8B>::zero().retained_capacity_bytes(), 0);
}

/// Composing with `constant + linear·X` evaluates as the substituted
/// point does, and a zero slope collapses to the constant value.
#[test]
fn linear_composition_substitutes_the_point() {
    fn check<F: FieldKernels>() {
        let polynomial = noise_poly::<F>(7, 0xF040);
        let constant = noise::<F>(1, 0xF041)[0];
        let linear = noise::<F>(1, 0xF042)[0];
        let composed = polynomial
            .compose_linear(constant, linear)
            .expect("compose");
        for seed in 0..4 {
            let point = noise::<F>(1, 0xF050 + seed)[0];
            assert_eq!(
                composed.evaluate(point),
                naive_evaluate(&polynomial, constant.add(linear.mul(point)))
            );
        }
        let flat = polynomial
            .compose_linear(constant, F::Elem::ZERO)
            .expect("flat");
        assert_eq!(flat.coefficient_count(), 1);
        assert_eq!(flat.coefficient(0), naive_evaluate(&polynomial, constant));
    }
    check::<Gf8B>();
    check::<Mersenne31>();
}

/// `D^[1](1 + X + X² + X³) = 1 + X²` over Gf8B, evaluated pointwise:
// `C(2,1)` vanishes while `C(1,1)` and `C(3,1)` survive.
#[test]
fn binary_hasse_evaluation_applies_the_parity_mask() {
    let polynomial = Polynomial::<Gf8B>::from_coefficients(&[b(1), b(1), b(1), b(1)]).expect("f");
    for point in [b(0), b(1), <Gf8B as Field>::GENERATOR] {
        assert_eq!(
            polynomial.evaluate_hasse(point, 1),
            b(1).add(point.mul(point))
        );
    }
}

/// The forced Karatsuba entry point maps zero to zero.
#[test]
fn karatsuba_maps_zero_to_zero() {
    let polynomial = noise_poly::<Gf8B>(9, 0xF060);
    assert!(
        karatsuba_multiply(&Polynomial::<Gf8B>::zero(), &polynomial)
            .expect("product")
            .is_zero()
    );
    assert!(
        karatsuba_multiply(&polynomial, &Polynomial::<Gf8B>::zero())
            .expect("product")
            .is_zero()
    );
}

/// Series inversion maps an empty precision to zero and rejects a zero
/// constant term instead of dividing by it.
#[test]
fn series_inversion_edges_hold() {
    let unit = noise_poly::<Gf8B>(6, 0xF090);
    assert!(naive_series_inverse(&unit, 0).is_zero());
    assert!(
        unit.inverse_mod_x_power(0)
            .expect("empty precision")
            .is_zero()
    );
    let mut coefficients: Vec<gf8b::Elem> = noise::<Gf8B>(6, 0xF091);
    coefficients[0] = <Gf8B as Field>::Elem::ZERO;
    let singular = Polynomial::<Gf8B>::from_coefficients(&coefficients).expect("singular");
    assert_eq!(
        singular.inverse_mod_x_power(6).map(|_| ()),
        Err(PolynomialError::ZeroConstantTerm {
            context: "truncated power-series inversion",
        })
    );
}

/// The extended gcd of zero with zero is the zero relation: the Bézout
/// identity holds trivially and every cofactor vanishes.
#[test]
fn extended_gcd_of_zero_with_zero_is_the_zero_relation() {
    let relation = Polynomial::<Gf8B>::zero()
        .extended_gcd(&Polynomial::<Gf8B>::zero())
        .expect("gcd");
    assert!(relation.gcd.is_zero());
    assert!(relation.a_cofactor.is_zero());
    assert!(relation.b_cofactor.is_zero());
}

/// Graded lexicographic order ranks by total degree first: `[0,5]` beats
/// `[4,0]` even though lexicographic storage order prefers `[4,0]`.
#[test]
fn graded_lex_ranks_total_degree_before_storage_order() {
    use poly_ring::{MonomialOrder, MultiIndex, SparsePolynomial, Term};

    let polynomial: SparsePolynomial<Mersenne31, 2> = SparsePolynomial::from_terms(vec![
        Term {
            exponents: MultiIndex::new([4, 0]),
            coefficient: m31(1),
        },
        Term {
            exponents: MultiIndex::new([0, 5]),
            coefficient: m31(1),
        },
    ]);
    let graded = polynomial
        .leading_term(&MonomialOrder::GradedLex)
        .expect("leading")
        .expect("nonzero");
    assert_eq!(graded.exponents.exponents(), &[0, 5]);
    let lex = polynomial
        .leading_term(&MonomialOrder::Lex)
        .expect("lex")
        .expect("nonzero");
    assert_eq!(lex.exponents.exponents(), &[4, 0]);
    // An empty polynomial has no leading term under any order.
    assert!(
        SparsePolynomial::<Mersenne31, 2>::zero()
            .leading_term(&MonomialOrder::GradedLex)
            .expect("empty")
            .is_none()
    );
    assert!(
        SparsePolynomial::<Mersenne31, 2>::default()
            .leading_term(&MonomialOrder::Weighted([1, 1]))
            .expect("default")
            .is_none()
    );
}

/// At and above the crossover the dispatched product takes the Karatsuba
/// tier and stays byte-identical to the naive convolution.
#[test]
fn dispatched_products_above_the_crossover_match_naive() {
    for (left_len, right_len) in [(2048, 2048), (2050, 7)] {
        let left = noise_poly::<Gf8B>(left_len, 0xF070 + left_len as u64);
        let right = noise_poly::<Gf8B>(right_len, 0xF080 + right_len as u64);
        assert_eq!(
            left.multiply(&right).expect("product"),
            naive_multiply(&left, &right)
        );
    }
}
