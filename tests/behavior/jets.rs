//! Exact jet fixtures: Hasse derivatives, not repeated formal derivatives
//! and not factorial-normalized derivatives.
//!
//! `f(a + T) = Σ_j D^[j]f(a) T^j` is the definition every test here checks,
//! at and above the characteristic, over binary and prime fields.

use fgf::field::Elem as _;
use fgf::field::Field;
use fgf::{Gf8B, Goldilocks, Mersenne31, gf8b, goldilocks, mersenne31};
use poly_ring::{DerivativePlan, JetPlan, MultiplicityPlan};

fn m31(n: u64) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(n as u32)
}

fn goldi(n: u64) -> goldilocks::Elem {
    goldilocks::Elem::from_raw(n)
}

#[test]
fn binary_fixture_matches_the_taylor_definition() {
    // f = 1 + X^2 over GF(2^8): f(T) = 1 + T^2 and
    // f(1 + T) = 1 + (1 + T)^2 = T^2 in characteristic two.
    let f = [gf8b::Elem::ONE, gf8b::Elem::ZERO, gf8b::Elem::ONE];
    let points = [gf8b::Elem::ZERO, gf8b::Elem::ONE];
    let plan = MultiplicityPlan::<Gf8B>::new(&points, &[3, 4], 8).expect("plan");
    let mut scratch = plan.scratch(1).expect("scratch");
    let mut output = [gf8b::Elem::ZERO; 7];
    plan.evaluate_into(&f, &mut scratch, &mut output)
        .expect("evaluate");
    assert_eq!(
        output,
        [
            gf8b::Elem::ONE,  // D^[0]f(0)
            gf8b::Elem::ZERO, // D^[1]f(0)
            gf8b::Elem::ONE,  // D^[2]f(0)
            gf8b::Elem::ZERO, // D^[0]f(1)
            gf8b::Elem::ZERO, // D^[1]f(1)
            gf8b::Elem::ONE,  // D^[2]f(1)
            gf8b::Elem::ZERO, // D^[3]f(1)
        ]
    );
}

#[test]
fn prime_fixture_matches_the_taylor_definition() {
    // f = 1 + 2X + 3X^2 + 4X^3: f(T) = 1 + 2T, and
    // f(2 + T) = 49 + 62T + 27T^2 + 4T^3 (mod 2^31 − 1 and mod Goldilocks).
    let f_m31 = [m31(1), m31(2), m31(3), m31(4)];
    let points_m31 = [m31(0), m31(2)];
    let plan = MultiplicityPlan::<Mersenne31>::new(&points_m31, &[2, 4], 8).expect("plan");
    let mut scratch = plan.scratch(1).expect("scratch");
    let mut output = [m31(0); 6];
    plan.evaluate_into(&f_m31, &mut scratch, &mut output)
        .expect("evaluate");
    assert_eq!(output, [m31(1), m31(2), m31(49), m31(62), m31(27), m31(4)]);

    let f_goldi = [goldi(1), goldi(2), goldi(3), goldi(4)];
    let points_goldi = [goldi(0), goldi(2)];
    let plan = MultiplicityPlan::<Goldilocks>::new(&points_goldi, &[2, 4], 8).expect("plan");
    let mut scratch = plan.scratch(1).expect("scratch");
    let mut output = [goldi(0); 6];
    plan.evaluate_into(&f_goldi, &mut scratch, &mut output)
        .expect("evaluate");
    assert_eq!(
        output,
        [
            goldi(1),
            goldi(2),
            goldi(49),
            goldi(62),
            goldi(27),
            goldi(4)
        ]
    );
}

#[test]
fn order_two_derivative_is_three_twelve_not_six_twentyfour() {
    // D^[2](1 + 2X + 3X^2 + 4X^3) = [C(2,2)·3, C(3,2)·4] = [3, 12]: the
    // twice-repeated formal derivative would be [6, 24] and the factorial
    // normalization [3/2, 6] — neither may appear.
    let plan = DerivativePlan::<Mersenne31>::new(2, 8).expect("plan");
    let f = [m31(1), m31(2), m31(3), m31(4)];
    let mut output = [m31(0); 2];
    plan.apply_into(&f, &mut output).expect("apply");
    assert_eq!(output, [m31(3), m31(12)]);

    let plan = DerivativePlan::<Goldilocks>::new(2, 8).expect("plan");
    let f = [goldi(1), goldi(2), goldi(3), goldi(4)];
    let mut output = [goldi(0); 2];
    plan.apply_into(&f, &mut output).expect("apply");
    assert_eq!(output, [goldi(3), goldi(12)]);

    // The batch form agrees row for row.
    let plan = DerivativePlan::<Mersenne31>::new(2, 8).expect("plan");
    let mut packed = [0_u8; 4 * 3 * 4];
    for degree in 0..4 {
        for lane in 0..3 {
            let value = m31((degree as u64 + lane as u64 + 1) % 7 + 1);
            let offset = (degree * 3 + lane) * 4;
            <Mersenne31 as Field>::encode(&mut packed[offset..offset + 4], value);
        }
    }
    let mut batch_output = [0_u8; 2 * 3 * 4];
    plan.apply_batch_into(&packed, 4, 3, &mut batch_output)
        .expect("batch apply");
    for lane in 0..3 {
        let mut scalar = [m31(0); 4];
        for (degree, slot) in scalar.iter_mut().enumerate() {
            let offset = (degree * 3 + lane) * 4;
            *slot = <Mersenne31 as Field>::decode(&packed[offset..offset + 4]);
        }
        let mut scalar_output = [m31(0); 2];
        plan.apply_into(&scalar, &mut scalar_output).expect("apply");
        for (degree, expected) in scalar_output.iter().enumerate() {
            let offset = (degree * 3 + lane) * 4;
            assert_eq!(
                <Mersenne31 as Field>::decode(&batch_output[offset..offset + 4]),
                *expected
            );
        }
    }
}

#[test]
fn small_characteristic_orders_survive_the_binomial() {
    // Over GF(2^8), (1 + T)^4 = 1 + T^4: every middle binomial of the
    // fourth row is even. An implementation that substitutes ordinary
    // derivatives would divide by 2! or 4! and diverge here.
    let plan = JetPlan::<Gf8B>::new(gf8b::Elem::ONE, 5, 5).expect("jet");
    let mut scratch = plan.scratch(1).expect("scratch");
    let f = [
        gf8b::Elem::ZERO,
        gf8b::Elem::ZERO,
        gf8b::Elem::ZERO,
        gf8b::Elem::ZERO,
        gf8b::Elem::ONE,
    ];
    let mut output = [gf8b::Elem::ZERO; 5];
    plan.evaluate_into(&f, &mut scratch, &mut output)
        .expect("jet");
    assert_eq!(
        output,
        [
            gf8b::Elem::ONE,
            gf8b::Elem::ZERO,
            gf8b::Elem::ZERO,
            gf8b::Elem::ZERO,
            gf8b::Elem::ONE,
        ]
    );
}

#[test]
fn truncated_translation_matches_the_oracle_across_the_recursion() {
    // Sizes crossing the Horner base in every direction: multiplicities
    // below, at, and above the coefficient count, so every divide-and-
    // conquer combine runs with truncated live orders.
    fn noise<F: fgf::kernel::FieldKernels>(len: usize, seed: u64) -> Vec<F::Elem> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let bytes = state.to_le_bytes();
                F::decode(&bytes[..F::BYTES]).add(F::Elem::ZERO)
            })
            .collect()
    }

    fn check<F: poly_ring::PolynomialField>(point: F::Elem) {
        for count in [17_usize, 33, 64, 100] {
            let coefficients = noise::<F>(count, 0x5EED_1000 + count as u64);
            for multiplicity in [1_usize, 3, 16, 17, 40, count, count + 7] {
                let plan = JetPlan::<F>::new(point, multiplicity, count + 7).expect("plan");
                let mut scratch = plan.scratch(1).expect("scratch");
                let mut output = vec![F::Elem::ZERO; multiplicity];
                plan.evaluate_into(&coefficients, &mut scratch, &mut output)
                    .expect("jet");
                let mut oracle = vec![F::Elem::ZERO; multiplicity];
                horner_scalar(&coefficients, point, multiplicity, &mut oracle);
                assert_eq!(output, oracle, "count={count} s={multiplicity}");
            }
        }
    }

    fn horner_scalar<E: fgf::field::Elem>(
        coefficients: &[E],
        point: E,
        multiplicity: usize,
        output: &mut [E],
    ) {
        output.fill(E::ZERO);
        for &coefficient in coefficients.iter().rev() {
            for order in (1..multiplicity).rev() {
                output[order] = output[order].mul(point).add(output[order - 1]);
            }
            output[0] = output[0].mul(point).add(coefficient);
        }
    }

    check::<Gf8B>(gf8b::Elem::ONE);
    check::<Mersenne31>(m31(2));
    check::<Goldilocks>(goldi(7));
    // The zero point runs the copy path with the same size spread.
    check::<Gf8B>(gf8b::Elem::ZERO);
}

#[test]
fn degenerate_jet_inputs_hold_their_contracts() {
    // The empty polynomial is the empty jet, at every point.
    for point in [gf8b::Elem::ZERO, gf8b::Elem::ONE] {
        let plan = JetPlan::<Gf8B>::new(point, 4, 8).expect("plan");
        let mut scratch = plan.scratch(1).expect("scratch");
        let mut output = [gf8b::Elem::ONE; 4];
        plan.evaluate_into(&[], &mut scratch, &mut output)
            .expect("jet");
        assert!(output.iter().all(|value| value.is_zero()));
    }
    // An oversized multiplicity is a geometry error, not a panic.
    assert!(matches!(
        JetPlan::<Goldilocks>::new(goldi(1), usize::MAX / 8 + 1, 2),
        Err(poly_ring::HasseError::GeometryOverflow { .. })
    ));
}
