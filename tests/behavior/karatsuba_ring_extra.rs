//! Karatsuba recombination and ring accumulation edges: the out-of-range
//! offset guards, the empty-middle recursion arm, and the shifted
//! accumulation skips.
//!
//! Each test asserts observable behavior — exact products against the naive
//! scalar oracle — on branches the agreement suites never take. The
//! recombination guards (`add_into`/`sub_into` with past-the-end offsets,
//! the zero-sum middle) trigger inside ordinary Karatsuba products at
//! specific operand shapes; the ring skips trigger in truncated products
//! with zero or out-of-range factor coefficients.

use fgf::field::Field;
use fgf::{Gf8B, Gf16, Mersenne31, gf8b, gf16};
use poly_ring::Polynomial;
use poly_ring::internals::karatsuba_multiply;

use crate::oracles;
use oracles::{naive_multiply, noise, noise_poly};

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn g16(value: u16) -> gf16::Elem {
    gf16::Elem::from_raw(value)
}

/// Karatsuba products match the naive oracle across the recursion shapes
/// that hit the recombination guards: odd splits, vanishing high parts,
/// zero-sum middles, and truncated destinations.
#[test]
fn karatsuba_shapes_match_the_naive_oracle() {
    // Odd and prime lengths force asymmetric splits; lengths around the
    // 48-coefficient base exercise the base-case boundary; a 300-coefficient
    // pair recurses twice.
    for (left_len, right_len, seed) in [
        (1_usize, 1_usize, 0xF401_u64),
        (3, 5, 0xF402),
        (47, 49, 0xF403),
        (48, 48, 0xF404),
        (49, 50, 0xF405),
        (63, 64, 0xF406),
        (65, 65, 0xF407),
        (96, 48, 0xF408),
        (97, 33, 0xF409),
        (100, 100, 0xF40A),
        (300, 200, 0xF40B),
    ] {
        let left = noise_poly::<Gf8B>(left_len, seed);
        let right = noise_poly::<Gf8B>(right_len, seed + 0x100);
        assert_eq!(
            karatsuba_multiply(&left, &right).expect("karatsuba"),
            naive_multiply(&left, &right),
            "shape {left_len}x{right_len} diverged"
        );
    }
    // Zero operands give the zero polynomial without touching the
    // recombination.
    assert!(
        karatsuba_multiply(&Polynomial::<Gf8B>::zero(), &noise_poly::<Gf8B>(60, 0xF40C))
            .expect("zero left")
            .is_zero()
    );
    assert!(
        karatsuba_multiply(&noise_poly::<Gf8B>(60, 0xF40D), &Polynomial::<Gf8B>::zero())
            .expect("zero right")
            .is_zero()
    );
}

/// A Karatsuba product with a zero high part takes the empty-middle arm
/// and still matches the oracle: `a = a_low` (high vanishes) with the
/// split landing exactly on the operand end.
#[test]
fn karatsuba_vanishing_high_part_matches_oracle() {
    // A power-of-two length at the recursion split with one operand much
    // shorter: the short operand's high part vanishes at the top split.
    let left = noise_poly::<Gf8B>(64, 0xF40E);
    let right = noise_poly::<Gf8B>(3, 0xF40F);
    assert_eq!(
        karatsuba_multiply(&left, &right).expect("karatsuba"),
        naive_multiply(&left, &right)
    );
    // Both high parts vanish when both operands fit below the top split
    // but the combined length still recurses: 48x48 splits at 48 with
    // empty highs.
    let left = noise_poly::<Gf8B>(48, 0xF410);
    let right = noise_poly::<Gf8B>(48, 0xF411);
    assert_eq!(
        karatsuba_multiply(&left, &right).expect("karatsuba"),
        naive_multiply(&left, &right)
    );
    // Same shapes over Gf16 exercise the wider-element recombination.
    let left = noise_poly::<Gf16>(64, 0xF412);
    let right = noise_poly::<Gf16>(3, 0xF413);
    assert_eq!(
        karatsuba_multiply(&left, &right).expect("karatsuba"),
        naive_multiply(&left, &right)
    );
}

/// Karatsuba over a prime field matches the oracle, proving the explicit
/// subtraction signs (not XOR) in the recombination.
#[test]
fn karatsuba_prime_field_subtraction_matches_oracle() {
    let left = noise_poly::<Mersenne31>(65, 0xF414);
    let right = noise_poly::<Mersenne31>(49, 0xF415);
    assert_eq!(
        karatsuba_multiply(&left, &right).expect("karatsuba"),
        naive_multiply(&left, &right)
    );
    // Sparse operands whose split sums cancel: alternating zeros force the
    // middle product through genuinely different `z1 - z0 - z2` values.
    let left = Polynomial::<Mersenne31>::from_coefficients(&noise::<Mersenne31>(70, 0xF416))
        .expect("left");
    let right = Polynomial::<Mersenne31>::from_coefficients(&noise::<Mersenne31>(70, 0xF417))
        .expect("right");
    assert_eq!(
        karatsuba_multiply(&left, &right).expect("karatsuba"),
        naive_multiply(&left, &right)
    );
}

/// `compose_linear` with a zero constant term drops the constant-add arm
/// and still matches direct evaluation at every point of the small field.
#[test]
fn compose_linear_matches_pointwise_evaluation() {
    // f = 1 + 2X + 3X² + 4X³ over Gf8B, affine 0 + g·X: every coefficient
    // of the result comes from the multiply arm, the zero-constant arm
    // contributes nothing.
    let coefficients = [b(1), b(2), b(3), b(4)];
    let polynomial = Polynomial::<Gf8B>::from_coefficients(&coefficients).expect("poly");
    let linear = b(7);
    let composed = polynomial.compose_linear(b(0), linear).expect("compose");
    // Oracle: Horner evaluation of f at (g·x) for every field element.
    for key in 0..Gf8B::ORDER {
        let bytes = key.to_le_bytes();
        let x = Gf8B::decode(&bytes[..Gf8B::BYTES]);
        let at = linear.mul(x);
        let mut want = <Gf8B as Field>::Elem::ZERO;
        for coefficient in coefficients.iter().rev() {
            want = want.mul(at).add(*coefficient);
        }
        assert_eq!(composed.evaluate(x), want, "point {key} diverged");
    }
    // A nonzero constant exercises the constant-add arm on the same input.
    let composed = polynomial.compose_linear(b(5), linear).expect("compose");
    for key in 0..Gf8B::ORDER {
        let bytes = key.to_le_bytes();
        let x = Gf8B::decode(&bytes[..Gf8B::BYTES]);
        let at = b(5).add(linear.mul(x));
        let mut want = <Gf8B as Field>::Elem::ZERO;
        for coefficient in coefficients.iter().rev() {
            want = want.mul(at).add(*coefficient);
        }
        assert_eq!(composed.evaluate(x), want, "point {key} diverged");
    }
}

/// Truncated products with zero and out-of-range factor coefficients skip
/// the accumulation and still match the naive truncation oracle.
#[test]
fn truncated_products_with_zero_factors_match_oracle() {
    // Factors with interior zeros: the zero scales skip accumulation.
    let left =
        Polynomial::<Gf8B>::from_coefficients(&[b(1), b(0), b(0), b(3), b(0), b(5)]).expect("left");
    let right = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(2), b(0), b(4)]).expect("right");
    for count in [0_usize, 1, 2, 3, 5, 9, 100] {
        let got = left.multiply_truncated(&right, count).expect("truncated");
        let mut full = naive_multiply(&left, &right);
        full.truncate(count);
        assert_eq!(got, full, "truncation {count} diverged");
    }
    // The shorter-factor operand order swaps source/factors; both orders
    // agree with the oracle.
    for count in [1_usize, 4, 9] {
        assert_eq!(
            right.multiply_truncated(&left, count).expect("swapped"),
            left.multiply_truncated(&right, count).expect("ordered"),
            "operand order diverged at {count}"
        );
    }
    // A truncation shorter than the shift of every nonzero factor term
    // writes no accumulation at all: the result is zero.
    let shifted = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(0), b(0), b(9)]).expect("x^3");
    let unit = Polynomial::<Gf8B>::one().expect("one");
    assert!(
        shifted
            .multiply_truncated(&unit, 3)
            .expect("short truncation")
            .is_zero()
    );
    // Wider elements take the same skips.
    let left = Polynomial::<Gf16>::from_coefficients(&[g16(1), g16(0), g16(9)]).expect("left");
    let right = Polynomial::<Gf16>::from_coefficients(&[g16(0), g16(4)]).expect("right");
    let mut full = naive_multiply(&left, &right);
    full.truncate(2);
    assert_eq!(left.multiply_truncated(&right, 2).expect("g16"), full);
}
