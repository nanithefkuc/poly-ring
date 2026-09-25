//! Truncated power series and the truncated EEA against their independent
//! oracles.

use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use fgf::{FanPaar8, Gf8B, Gf16, Gf32, Goldilocks, Mersenne31, QuadMersenne31};
use poly_ring::{Polynomial, PolynomialError, truncated_eea};

use crate::oracles;
use oracles::{naive_series_inverse, noise, noise_unit, reference_berlekamp_massey};

fn assert_series_identities<F: FieldKernels>() {
    for (len, t) in [(8, 8), (17, 16), (33, 40), (64, 63)] {
        let unit = noise_unit::<F>(len, 0x5EED_1000 + len as u64);
        let inverse = unit.inverse_mod_x_power(t).expect("series inverse");
        // Defining identity by an independent truncated multiply: the
        // product below x^t is exactly the constant one.
        let identity = unit
            .multiply_truncated(&inverse, t)
            .expect("identity product");
        assert_eq!(
            identity,
            Polynomial::<F>::one().expect("one"),
            "a · a⁻¹ must be 1 mod x^{t}"
        );
        // Newton doubling equals the linear schoolbook solve.
        assert_eq!(inverse, naive_series_inverse(&unit, t));
        // The inverse of a unit is its reciprocal.
        let constant = Polynomial::<F>::constant(unit.coefficient(0)).expect("constant");
        let constant_inverse = constant.inverse_mod_x_power(t).expect("constant inverse");
        assert_eq!(
            constant_inverse,
            Polynomial::constant(unit.coefficient(0).inv()).expect("reciprocal")
        );
        // Series division composes: (a/b)·b ≡ a (mod x^t).
        let numerator = oracles::noise_poly::<F>(len, 0x5EED_1500 + len as u64);
        let quotient = poly_ring::series_divide(&numerator, &unit, t).expect("series division");
        let reconstructed = quotient
            .multiply_truncated(&unit, t)
            .expect("reconstruction");
        let expected = numerator.clone();
        assert!(reconstructed.matches_mod_x_power(&expected, t));
    }

    // A zero constant term is the documented error, not a silent zero (U6).
    let zero_constant = Polynomial::<F>::from_coefficients(&{
        let mut coefficients = noise::<F>(8, 0x5EED_1600);
        coefficients[0] = F::Elem::ZERO;
        coefficients
    })
    .expect("zero-constant polynomial");
    assert_eq!(
        zero_constant.inverse_mod_x_power(4),
        Err(PolynomialError::ZeroConstantTerm {
            context: "truncated power-series inversion"
        })
    );
    assert_eq!(
        Polynomial::<F>::zero().inverse_mod_x_power(4),
        Err(PolynomialError::ZeroConstantTerm {
            context: "truncated power-series inversion"
        })
    );

    // reverse: X^d · p(1/X) has the coefficients in the opposite order.
    let p = oracles::noise_poly::<F>(7, 0x5EED_1700);
    let mut expected: Vec<_> = p.coefficients().collect();
    expected.reverse();
    assert_eq!(
        p.reverse(),
        Polynomial::from_coefficients(&expected).expect("reversed")
    );
    assert!(Polynomial::<F>::zero().reverse().is_zero());
}

trait TruncateCheck<F: FieldKernels> {
    fn matches_mod_x_power(&self, other: &Self, t: usize) -> bool;
}

impl<F: FieldKernels> TruncateCheck<F> for Polynomial<F> {
    fn matches_mod_x_power(&self, other: &Self, t: usize) -> bool {
        for degree in 0..t {
            if self.coefficient(degree) != other.coefficient(degree) {
                return false;
            }
        }
        true
    }
}

#[test]
fn series_identities_hold_across_fields() {
    assert_series_identities::<Gf8B>();
    assert_series_identities::<Gf8B>();
    assert_series_identities::<Gf16>();
    assert_series_identities::<Gf32>();
    assert_series_identities::<FanPaar8>();
}

fn assert_truncated_eea_identities<F: FieldKernels>() {
    // Padé identity on (x^{2t}, S): remainder ≡ b_cofactor · S (mod x^{2t})
    // with deg remainder < t — the key equation — checked by the truncated
    // multiply rather than a rerun.
    for t in [4, 8, 12] {
        let syndrome = oracles::noise_poly::<F>(2 * t, 0x5EED_2000 + t as u64);
        let x_to_t = monomial::<F>(2 * t);
        let step = truncated_eea(&x_to_t, &syndrome, t).expect("truncated eea");
        let product = step
            .b_cofactor
            .multiply_truncated(&syndrome, t)
            .expect("padé product");
        assert!(product.matches_mod_x_power(&step.remainder, 2 * t));
        assert!(
            step.remainder.degree().is_none_or(|degree| degree < t),
            "the stop condition was not honored"
        );

        // Dornstetter equivalence: the normalized b_cofactor equals the
        // reference Berlekamp–Massey connection polynomial on the same
        // sequence.
        let sequence: Vec<_> = syndrome.coefficients().take(2 * t).collect();
        let (connection, l) = reference_berlekamp_massey::<F>(&sequence);
        let mut normalized = step.b_cofactor.clone();
        let constant = step.b_cofactor.coefficient(0);
        if !constant.is_zero() {
            normalized = normalized.scaled(constant.inv());
        }
        let expected = Polynomial::from_coefficients(&connection).expect("connection");
        assert_eq!(
            normalized, expected,
            "truncated EEA and Berlekamp–Massey disagree at t = {t}"
        );
        assert_eq!(normalized.degree(), Some(l));

        // LFSR reproduction: the sequence satisfies the recurrence.
        for i in l..sequence.len() {
            let mut value = F::Elem::ZERO;
            for (j, coefficient) in connection.iter().enumerate() {
                value = value.add(coefficient.mul(sequence[i - j]));
            }
            assert!(
                value.is_zero(),
                "connection polynomial does not reproduce the sequence"
            );
        }
    }

    // Generic (a, b) form of the identity: r = u·a + v·b exactly.
    let a = oracles::noise_poly::<F>(9, 0x5EED_2100);
    let b = oracles::noise_poly::<F>(6, 0x5EED_2101);
    let step = truncated_eea(&a, &b, 2).expect("truncated eea");
    let identity = step
        .a_cofactor
        .multiply(&a)
        .expect("u·a")
        .add(&step.b_cofactor.multiply(&b).expect("v·b"))
        .expect("sum");
    assert_eq!(identity, step.remainder);

    // b already below the stop returns (b, 0, 1).
    let small = oracles::noise_poly::<F>(2, 0x5EED_2102);
    let step = truncated_eea(&a, &small, 4).expect("truncated eea");
    assert_eq!(step.remainder, small);
    assert!(step.a_cofactor.is_zero());
    assert_eq!(step.b_cofactor, Polynomial::one().expect("one"));
}

fn monomial<F: FieldKernels>(degree: usize) -> Polynomial<F> {
    let mut coefficients = vec![F::Elem::ZERO; degree + 1];
    coefficients[degree] = F::Elem::ONE;
    Polynomial::from_coefficients(&coefficients).expect("monomial")
}

#[test]
fn truncated_eea_matches_berlekamp_massey_across_fields() {
    assert_truncated_eea_identities::<Gf8B>();
    assert_truncated_eea_identities::<Gf8B>();
    assert_truncated_eea_identities::<Gf16>();
    assert_truncated_eea_identities::<Gf32>();
}

// ---------------------------------------------------------------------------
// Odd characteristic

#[test]
fn prime_series_inverse_matches_the_geometric_series() {
    // (1 − X)^{-1} = 1 + X + X² + … The characteristic-two update
    // `b ← a·b²` is wrong here; the general Newton step `b ← b·(2 − a·b)` is
    // what produces the all-ones prefix.
    fn check<F: FieldKernels>() {
        let one = F::Elem::ONE;
        let unit = Polynomial::<F>::from_coefficients(&[one, one.neg()]).expect("1 − X");
        let inverse = unit.inverse_mod_x_power(5).expect("inverse");
        assert_eq!(
            inverse,
            Polynomial::<F>::from_coefficients(&[one; 5]).expect("geometric series")
        );
        assert_eq!(inverse, naive_series_inverse(&unit, 5));
    }
    check::<Mersenne31>();
    check::<Goldilocks>();
    check::<QuadMersenne31>();
}

#[test]
fn prime_series_identities_hold() {
    fn check<F: FieldKernels>() {
        for (len, t) in [(8, 8), (17, 16), (33, 40), (64, 63)] {
            let unit = noise_unit::<F>(len, 0x5EED_3000 + len as u64);
            let inverse = unit.inverse_mod_x_power(t).expect("series inverse");
            assert_eq!(
                unit.multiply_truncated(&inverse, t).expect("identity"),
                Polynomial::<F>::one().expect("one"),
                "a · a⁻¹ must be 1 mod x^{t} over {}",
                F::NAME
            );
            assert_eq!(inverse, naive_series_inverse(&unit, t));

            // Series division agrees with multiply-by-inverse.
            let numerator = oracles::noise_poly::<F>(len, 0x5EED_3100 + len as u64);
            assert_eq!(
                poly_ring::series_divide(&numerator, &unit, t).expect("series divide"),
                numerator.multiply_truncated(&inverse, t).expect("product")
            );
        }
        // t = 0 and a zero constant term keep their contracts.
        let unit = noise_unit::<F>(4, 0x5EED_3200);
        assert!(unit.inverse_mod_x_power(0).expect("t = 0").is_zero());
        let shifted = unit.shifted(1).expect("X · a");
        assert!(matches!(
            shifted.inverse_mod_x_power(4),
            Err(PolynomialError::ZeroConstantTerm { .. })
        ));
    }
    check::<Mersenne31>();
    check::<Goldilocks>();
    check::<QuadMersenne31>();
}
